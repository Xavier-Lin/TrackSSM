import logging
from detectron2.engine.train_loop import HookBase
from detectron2.data import MetadataCatalog, build_detection_train_loader
class ClipHook(HookBase):
    "Update sampler length of input video clip"
    def __init__(self, cfg):
        self._cfg = cfg
  
    def after_step(self):
        if self.trainer.iter + 1 in self._cfg.MODEL.MDR.SAMPLER_STEPS:
            self.trainer.data_loader.dataset.dataset.update_sampler_len(self.trainer.iter + 1)
            self.trainer.data_loader = build_detection_train_loader(
                self.trainer.data_loader.dataset.dataset,
                mapper = None,
                total_batch_size = self._cfg.SOLVER.IMS_PER_BATCH,
                num_workers = self._cfg.DATALOADER.NUM_WORKERS,
                aspect_ratio_grouping=False if self._cfg.MODEL.MDR.IS_TRAIN else True,
            )
 
    def before_train(self):
        self.trainer.data_loader.dataset.dataset.update_sampler_len(self.trainer.iter + 1)
        self.trainer.data_loader = build_detection_train_loader(
            self.trainer.data_loader.dataset.dataset,
            mapper = None,
            total_batch_size = self._cfg.SOLVER.IMS_PER_BATCH,
            num_workers = self._cfg.DATALOADER.NUM_WORKERS,
            aspect_ratio_grouping=False if self._cfg.MODEL.MDR.IS_TRAIN else True,
        )
   