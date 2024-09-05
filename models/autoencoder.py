import torch
from torch.nn import Module
import models.diffusion as diffusion
from models.diffusion import VarianceSchedule, D2MP_OB
from models.motion_decoder import Time_info_decoder
import numpy as np

class D2MP(Module):
    def __init__(self, config, encoder=None, device="cuda"):
        super().__init__()
        self.config = config
        self.device = device
        self.encoder = encoder
        if config.use_diffmot:
            self.diffnet = getattr(diffusion, config.diffnet)
            self.diffusion = D2MP_OB(
                # net = self.diffnet(point_dim=2, context_dim=config.encoder_dim, tf_layer=config.tf_layer, residual=False),
                net=self.diffnet(point_dim=4, context_dim=config.encoder_dim, tf_layer=config.tf_layer, residual=False),
                var_sched = VarianceSchedule(
                    num_steps=100,
                    beta_T=5e-2,
                    mode='linear'
                ),
                config=self.config
            )
        else:
            self.ssm_decoder = Time_info_decoder()

    def generate(self, conds, sample, bestof, flexibility=0.0, ret_traj=False, img_w=None, img_h=None):
        cond_encodeds = []
        for i in range(len(conds)):
            tmp_c = conds[i]
            tmp_c = np.array(tmp_c)
            tmp_c[:, 0::2] = tmp_c[:, 0::2] / img_w
            tmp_c[:, 1::2] = tmp_c[:, 1::2] / img_h
            tmp_conds = torch.tensor(tmp_c, dtype=torch.float)
            if len(tmp_conds) != self.config.interval:
                pad_conds = tmp_conds[-1].repeat((self.config.interval, 1))
                tmp_conds = torch.cat((tmp_conds, pad_conds), dim=0)[:self.config.interval]
            cond_encodeds.append(tmp_conds.unsqueeze(0))
        cond_encodeds = torch.cat(cond_encodeds).cuda()
        # import pdb;pdb.set_trace()
        cond_flow = self.encoder(cond_encodeds)
        if self.config.use_diffmot:
            track_pred = self.diffusion.sample(cond_flow, sample, bestof, flexibility=flexibility, ret_traj=ret_traj)
        else:
            track_pred = self.ssm_decoder(cond_encodeds[:, -1, :4], cond_flow)
        return track_pred.cpu().detach().numpy()

    def forward(self, batch):
        cond_encoded = self.encoder(batch["condition"]) # input： batch["condition"] b 5 8。 out： b 1 256
        loss = self.ssm_decoder(batch["condition"][:, -1, :4], cond_encoded, curr_gt = batch["cur_bbox"])
        return loss