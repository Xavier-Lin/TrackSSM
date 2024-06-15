import torch.nn.functional as F
from .common import *

class VarianceSchedule(Module):

    def __init__(self, num_steps, mode='linear',beta_1=1e-4, beta_T=5e-2,cosine_s=8e-3):
        super().__init__()
        assert mode in ('linear', 'cosine')
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)
        elif mode == 'cosine':
            timesteps = (
            torch.arange(num_steps + 1) / num_steps + cosine_s
            )
            alphas = timesteps / (1 + cosine_s) * math.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding
        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()
        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)
        # self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alpha_bars))
        # self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alpha_bars - 1))

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas




class HMINet(Module):

    def __init__(self, point_dim=4, context_dim=256, tf_layer=3, residual=False):
        super().__init__()
        self.residual = residual
        self.pos_emb = PositionalEncoding(d_model=2*context_dim, dropout=0.1, max_len=24)
        self.pos_emb2= PositionalEncoding(d_model=context_dim, dropout=0.1, max_len=24)
        self.concat1 = MFL(4, context_dim // 2, context_dim+3)
        self.concat1_2 = MFL(context_dim // 2, context_dim, context_dim + 3)
        self.concat1_3 = MFL(context_dim, 2 * context_dim, context_dim + 3)
        self.layer = nn.TransformerEncoderLayer(d_model=2*context_dim, nhead=4, dim_feedforward=4*context_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=tf_layer)
        self.layer2 = nn.TransformerEncoderLayer(d_model=context_dim, nhead=4, dim_feedforward=2 * context_dim)
        self.transformer_encoder2 = nn.TransformerEncoder(self.layer2, num_layers=tf_layer)
        self.concat3 = MFL(2*context_dim,context_dim, context_dim+3)
        self.concat4 = MFL(context_dim,context_dim//2, context_dim+3)
        self.linear = MFL(context_dim//2, 4, context_dim+3)
        #self.linear = nn.Linear(128,2)

    def forward(self, x, beta, context):
        #当前帧相对于前一帧motion_dispartment 的 噪声扩散结果 -- b 4
        #采样 时间点 t的 取对数结果 t.log() / 4 -- b
        #聚合前5帧的位置和运动信息的embedding -- b 1 256 --- 相当于 是对 当前帧目标位置和运动信息embedding的预测！！！
        batch_size = x.size(0)# b
        beta = beta.view(batch_size, 1)          # (b, 1) t.log() / 4 
        context = context.view(batch_size, -1)   # (b, 256)
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (b, 3)对 每条轨迹 均匀分布采样时间点 进行 正余弦编码
        ctx_emb = torch.cat([time_emb, context], dim=-1)    # (b, 256+3) -- 对 预测的 当前帧位置 和 运动信息的 embedding 进行 采样时间点编码
        x = self.concat1_3(ctx_emb, self.concat1_2(ctx_emb, self.concat1(ctx_emb,x)))# b 512
        # 这一步 就是对 当前帧 相对于前一帧motion_dispartment噪声扩散结果 的反向去噪过程！！！
        # 使用什么来去噪呢？ 使用 的是 带采样时间编码 的 流信息！！！--- 来对 运动位移的 噪声扩散结果 进行 反向去噪
        final_emb = x.unsqueeze(0)# 1 b 512 --- 获得 反向去噪 之后的 运动位移结果
        # import pdb;pdb.set_trace()
        final_emb = self.pos_emb(final_emb)# 对运动位移进行正余弦编码 -- 得到 运动位移的 高维度表示. 1 b 512
        trans = self.transformer_encoder(final_emb).permute(1,0,2).squeeze(1)# 对 反向去噪的 运动位移表示 进行特征增强 --- b 512
        trans = self.concat3(ctx_emb, trans)# 再进行去噪！--- b 256
        
        final_emb = trans.unsqueeze(0)# 得到 反向去噪的 运动位移表示 --- 1 b 256
        final_emb = self.pos_emb2(final_emb)# 进行位置编码 --- 1 b 256
        trans = self.transformer_encoder2(final_emb).permute(1, 0, 2).squeeze(1)# b 256
        trans = self.concat4(ctx_emb, trans)# 再进行去噪！--- b 128
        
        return self.linear(ctx_emb, trans)




class D2MP_OB(Module):

    def __init__(self, net, var_sched:VarianceSchedule, config):
        super().__init__()
        self.config = config
        self.net = net
        self.var_sched = var_sched
        self.eps = self.config.eps
        self.weight = True

    def q_sample(self, x_start, noise, t, C):
        time = t.reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
        x_noisy = x_start + C * time + torch.sqrt(time) * noise # 对于 每条 轨迹样本的 单步噪声 扩散过程！！重点 -- 这里对应的就是论文中的噪声扩散公式：M_t = (1 - t) * M_0 + sqrt(t) * N
        return x_noisy

    def pred_x0_from_xt(self, xt, noise, C, t):
        time = t.reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
        x0 = xt - C * time - torch.sqrt(time) * noise
        return x0

    def pred_C_from_xt(self, xt, noise, t):
        time = t.reshape(noise.shape[0], *((1,) * (len(noise.shape) - 1)))
        C = (xt - torch.sqrt(time) * noise) / (time - 1)
        return C

    def pred_xtms_from_xt(self, xt, noise, C, t, s):
        time = t.reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
        s = s.reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
        mean = xt + C * (time-s) - C * time - s / torch.sqrt(time) * noise
        epsilon = torch.randn_like(mean, device=xt.device)
        sigma = torch.sqrt(s * (time-s) / time)
        xtms = mean + sigma * epsilon
        return xtms

    def forward(self, x_0, context, t=None):# 轨迹当前帧相对于前一帧的bbox motion -- b 4， 聚合 前5帧的 位置和运动信息的 embedding -- b 1 256
        batch_size, point_dim = x_0.size()# b 4
        if t == None:# T
            t = torch.rand(x_0.shape[0], device=x_0.device) * (1. - self.eps) + self.eps# 0.999 * t + 0.001 --- 这种做法 是为了确保 采样的时间点t 不等于 0，为了 避免数值计算中 的错误！！！
        # torch.rand(x_0.shape[0], device=x_0.device) --- 对于每一个轨迹样本，都从均匀分布中采样，从采样batch个样本
        # 对 这些 时间采样 的样本 进行缩放取对数，以用于 下一步的 时间点余弦编码
        beta = t.log() / 4 # 生成 采样时间点 对应的beta 数值 这是扩散模型中的常见操作
        e_rand = torch.randn_like(x_0).cuda()  # (B, N, d)# 对于 每一个轨迹样本 从 正态分布 噪声中采样 得到噪声样本
        C = -1 * x_0 # 将运动偏移量detla转向 从当前帧到前一帧
        x_noisy = self.q_sample(x_start=x_0, noise=e_rand, t=t, C=C)# 得到 当前帧相对于前一帧motion_dispartment 的 噪声扩散结果
        t = t.reshape(-1, 1)# b 1

        pred = self.net(x_noisy, beta=beta, context=context)# 这一步 主要进行了 前向扩散 以及 反向去噪 的过程 -- b 4
        C_pred = pred
        noise_pred = (x_noisy - (t - 1) * C_pred) / t.sqrt()
        if not self.weight:
            loss_C = F.smooth_l1_loss(C_pred.view(-1, point_dim), C.view(-1, point_dim), reduction='mean')
            # loss_x0 = F.smooth_l1_loss(x_rec.view(-1, point_dim), x_0.view(-1, point_dim), reduction='mean')
            loss_noise = F.smooth_l1_loss(noise_pred.view(-1, point_dim), e_rand.view(-1, point_dim), reduction='mean')
            loss = 0.5 * loss_C + 0.5 * loss_noise
        else:
            simple_weight1 = (t ** 2 - t + 1) / t
            simple_weight2 = (t ** 2 - t + 1) / (1 - t + self.eps)
            loss_C = F.smooth_l1_loss(C_pred.view(-1, point_dim), C.view(-1, point_dim), reduction='none')
            loss_noise = F.smooth_l1_loss(noise_pred.view(-1, point_dim), e_rand.view(-1, point_dim), reduction='none')
            loss = simple_weight1 * loss_C + simple_weight2 * loss_noise
            loss = loss.mean()
            # 损失这部分 使用的都是 最正常的l1损失
        return loss

    def sample(self, context, sample, bestof, point_dim=4, flexibility=0.0, ret_traj=False):
        traj_list = []
        # context = context.to(self.var_sched.betas.device)
        for i in range(sample):
            batch_size = context.size(0)
            if bestof:
                x_T = torch.randn([batch_size, point_dim]).to(context.device)
            else:
                x_T = torch.zeros([batch_size, point_dim]).to(context.device)

            self.var_sched.num_steps = 1
            traj = {self.var_sched.num_steps: x_T}

            cur_time = torch.ones((batch_size,), device=x_T.device)
            step = 1. / self.var_sched.num_steps
            for t in range(self.var_sched.num_steps, 0, -1):
                s = torch.full((batch_size,), step, device=x_T.device)
                if t == 1:
                    s = cur_time

                x_t = traj[t]
                beta = cur_time.log() / 4
                t_tmp = cur_time.reshape(-1, 1)
                pred = self.net(x_t, beta=beta, context=context)
                C_pred = pred
                noise_pred = (x_t - (t_tmp - 1) * C_pred) / t_tmp.sqrt()

                x0 = self.pred_x0_from_xt(x_t, noise_pred, C_pred, cur_time)
                x0.clamp_(-1., 1.)
                C_pred = -1 * x0
                x_next = self.pred_xtms_from_xt(x_t, noise_pred, C_pred, cur_time, s)
                cur_time = cur_time - s
                traj[t-1] = x_next.detach()     # Stop gradient and save trajectory.
                if not ret_traj:
                   del traj[t]

            if ret_traj:
                traj_list.append(traj)
            else:
                traj_list.append(traj[0])

        return torch.stack(traj_list)
