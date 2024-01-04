#
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MDR Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import os
import time, glob
import torch
import logging
import itertools
from pathlib import Path
from collections import OrderedDict
from typing import Any, Dict, List, Set

import motmetrics as mm



import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.engine import AutogradProfiler, DefaultTrainer, default_argument_parser, default_setup, launch, hooks
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.solver.build import maybe_add_gradient_clipping
from fvcore.nn.precise_bn import get_bn_modules
from sparsercnn import add_MDR_config, MOTtrainset, MOTEvaluator, ClipHook
from register_test_data import *
logger = logging.getLogger("detectron2")

class Trainer(DefaultTrainer):
#     """
#     Extension of the Trainer class adapted to MDR.
#     """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        mot_train_dataset = MOTtrainset(cfg)
        mot_train_loader = build_detection_train_loader(
            mot_train_dataset,
            mapper = None,
            total_batch_size = cfg.SOLVER.IMS_PER_BATCH,
            num_workers = cfg.DATALOADER.NUM_WORKERS,
            aspect_ratio_grouping=False if cfg.MODEL.MDR.IS_TRAIN else True,
        )
        return mot_train_loader
    
    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        ret.append(ClipHook(cfg))
        return ret
        
    @classmethod
    def build_optimizer(cls, cfg, model):
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
      
        # fix detector weights
        if cfg.MODEL.MDR.IS_FIX:
            freeze = ['backbone', 'head', 'init_proposal_features', 'init_proposal_boxes'] 
            for k, v in model.named_parameters():
                if k.split('.')[0] in freeze:
                    logger.info('freezing %s' % k)
                    v.requires_grad = False
         
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "backbone" in key or 'head' in key or 'init_proposal_features' in key or 'init_proposal_boxes' in key :
                lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER

            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer
    
    @classmethod
    def track(cls, cfg, model):
        def compare_dataframes(gts, ts):
            accs = []
            names = []
            for k, tsacc in ts.items():
                if k in gts:            
                    logger.info('Comparing {}...'.format(k))
                    accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
                    names.append(k)
                else:
                    logger.warning('No ground truth for {}, skipping.'.format(k))

            return accs, names
        
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            file_name = os.path.join(cfg.OUTPUT_DIR, dataset_name)
            if comm.is_main_process():
                os.makedirs(file_name, exist_ok=True)
            results_folder = os.path.join(file_name, "track_results")    
            os.makedirs(results_folder, exist_ok=True) 
            
            data_loader = cls.build_test_loader(cfg, dataset_name)
            metadata = MetadataCatalog.get(dataset_name) 

            # build evaluator for specific dataloader
            evaluator = MOTEvaluator(
                cfg=cfg,
                dataloader=data_loader,
            )
            
            # start evaluate
            evaluator.evaluate(
                model, result_folder = results_folder
            )
            # evaluate MOTA
            mm.lap.default_solver = 'lap'

            if 'val_half.json' in metadata.json_file:
                gt_type = '_val_half'
            else:
                gt_type = ''
            logger.info(f'gt_type: {gt_type}')
         
            gtfiles = glob.glob(os.path.join(metadata.image_root, '*/gt/gt{}.txt'.format(gt_type)))
           
            print('gt_files', gtfiles)
            tsfiles = [f for f in glob.glob(os.path.join(results_folder, '*.txt')) if not os.path.basename(f).startswith('eval')]

            logger.info('Found {} groundtruths and {} test files.'.format(len(gtfiles), len(tsfiles)))
            logger.info('Available LAP solvers {}'.format(mm.lap.available_solvers))
            logger.info('Default LAP solver \'{}\''.format(mm.lap.default_solver))
            logger.info('Loading files.')
            
            gt = OrderedDict([(Path(f).parts[-3], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=1)) for f in gtfiles])
            ts = OrderedDict([(os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=-1)) for f in tsfiles])    
            
            mh = mm.metrics.create()    
            accs, names = compare_dataframes(gt, ts)
            
            logger.info('Running metrics')
            metrics = ['recall', 'precision', 'num_unique_objects', 'mostly_tracked',
                    'partially_tracked', 'mostly_lost', 'num_false_positives', 'num_misses',
                    'num_switches', 'num_fragmentations', 'mota', 'motp', 'num_objects']
            summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)

            div_dict = {
                'num_objects': ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations'],
                'num_unique_objects': ['mostly_tracked', 'partially_tracked', 'mostly_lost']}
            for divisor in div_dict:
                for divided in div_dict[divisor]:
                    summary[divided] = (summary[divided] / summary[divisor])
            fmt = mh.formatters
            change_fmt_list = ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations', 'mostly_tracked',
                            'partially_tracked', 'mostly_lost']
            for k in change_fmt_list:
                fmt[k] = fmt['mota']
            print(mm.io.render_summary(summary, formatters=fmt, namemap=mm.io.motchallenge_metric_names))

            metrics = mm.metrics.motchallenge_metrics + ['num_objects']
            summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
            logger.info('\n'+mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
            logger.info('Completed')
    

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_MDR_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        if cfg.MODEL.MDR.TRACKING:
            Trainer.track(cfg, model)
            return 0
        else:
            res = Trainer.test(cfg, model)
            if comm.is_main_process():
                verify_results(cfg, res)
            return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
    '''
    CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_net.py  --num-gpus 4  --config-file configs/MDR-r50-500pro-36e.yaml
    
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train_net.py  --num-gpus 4  --config-file configs/MDR-r50-500pro-36e-track.yaml
    
    CUDA_VISIBLE_DEVICES=0  python train_net.py  --num-gpus 1  --eval-only  --config-file configs/MDR-r50-500pro-36e-track-infer.yaml
    
    '''
