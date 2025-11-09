import torch
from torch import nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate

class TemporalAwareSpikeAttention(nn.Module):
    """
    Temporal-aware Spiking Self-Attention with multi-scale temporal activity
    - Q/K/V SNN化 (Linear → LIF → LN)
    - 支持基于 rate 的调制
    - 新增: 多尺度时间活跃度 (早期/晚期/斜率)，通过可学习融合调制 Q
    """

    def __init__(self, d_model, n_heads=6, num_steps=4, use_first_spike=False, alpha=1.0, tau=1.0):
        super().__init__()
        self.num_steps = num_steps
        self.n_heads = max(1, n_heads)
        self.use_first_spike = use_first_spike
        self.alpha = alpha
        self.tau = tau

        # 可调参数
        self.lambda_mix = 0.0
        self.detach_weight = True
        self.use_spike_qkv = True
        self.use_temporal_mod = True
        self.use_multi_scale = True   # 新增：是否启用多尺度特征

        # 线性投影
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # LIF 神经元
        mk_lif = lambda: snn.Leaky(beta=0.99, spike_grad=surrogate.atan(2.0),
                                   init_hidden=True, threshold=1.0)
        self.lif_q, self.lif_k, self.lif_v = mk_lif(), mk_lif(), mk_lif()

        # LayerNorm
        self.q_ln = nn.LayerNorm(d_model)
        self.k_ln = nn.LayerNorm(d_model)
        self.v_ln = nn.LayerNorm(d_model)
        self.out_ln = nn.LayerNorm(d_model)

        # Dropout
        self.attn_drop = nn.Dropout(p=0.0)
        self.proj_drop = nn.Dropout(p=0.0)

        # 残差调制强度
        self.gamma_temporal = nn.Parameter(torch.tensor(0.2))

        # 多尺度融合 (3 → 1)
        self.w_fuse = nn.Linear(3, 1, bias=False)

    def forward(self, x, spikes_time_series):
        """
        x: [B, L, d_model]
        spikes_time_series: [B, H, C, L, T]
        """
        B, L, d_model = x.shape
        T = spikes_time_series.shape[-1]
        device = x.device

        # ===== Q/K/V with LIF =====
        Q_lin, K_lin, V_lin = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        if self.use_spike_qkv:
            Q, K, V = self.lif_q(Q_lin), self.lif_k(K_lin), self.lif_v(V_lin)
        else:
            Q, K, V = Q_lin, K_lin, V_lin

        # ===== Multi-scale temporal activity =====
        if self.use_temporal_mod:
            mask_spk = (spikes_time_series > 0).float()  # [B,H,C,L,T]

            if self.use_first_spike:
                t_idx = torch.arange(T, device=device).view(1,1,1,1,-1)
                first_t = (mask_spk * (t_idx+1))
                first_t[first_t == 0] = self.num_steps
                first_t = first_t.min(dim=-1).values.float()  # [B,H,C,L]
                w_base = torch.exp(-self.alpha * first_t)
            else:
                w_base = mask_spk.mean(dim=-1)  # [B,H,C,L]

            if self.use_multi_scale:
                # early / late / slope
                early_rate = mask_spk[..., :T//2].mean(dim=-1)   # [B,H,C,L]
                late_rate  = mask_spk[..., T//2:].mean(dim=-1)   # [B,H,C,L]
                slope = late_rate - early_rate                   # [B,H,C,L]

                early = early_rate.mean(dim=(1,2))   # [B,L]
                late  = late_rate.mean(dim=(1,2))    # [B,L]
                slp   = slope.mean(dim=(1,2))        # [B,L]

                w_multi = torch.stack([early, late, slp], dim=-1)  # [B,L,3]
                w_L = self.w_fuse(w_multi).squeeze(-1)             # [B,L]
            else:
                # 原始版本：直接平均
                w_L = w_base.mean(dim=(1,2))  # [B,L]

            # 归一化
            w_L = (w_L - w_L.mean(dim=-1, keepdim=True)) / (w_L.std(dim=-1, keepdim=True) + 1e-6)
            if self.detach_weight:
                w_L = w_L.detach()

            # 残差式调制 Q
            Q = Q * (1.0 + torch.clamp(self.gamma_temporal, min=0.0) * w_L.unsqueeze(-1))

        # ===== LN =====
        Q, K, V = self.q_ln(Q), self.k_ln(K), self.v_ln(V)

        # ===== Multi-head split =====
        nH = self.n_heads
        assert d_model % nH == 0
        dh = d_model // nH
        def split_heads(t): return t.view(B, L, nH, dh).permute(0,2,1,3)
        Qh, Kh, Vh = map(split_heads, (Q,K,V))

        # ===== Similarity =====
        dot = torch.matmul(Qh, Kh.transpose(-2,-1))  # [B,nH,L,L]
        neg_l1 = -(Qh.unsqueeze(3)-Kh.unsqueeze(2)).abs().sum(dim=-1)
        score = self.lambda_mix*dot + (1.0-self.lambda_mix)*neg_l1
        score = score / max(self.tau, 1e-6)
        score = score - score.max(dim=-1, keepdim=True).values

        # ===== Softmax =====
        attn = F.softmax(score, dim=-1)
        attn = self.attn_drop(attn)

        # ===== Aggregate =====
        Oh = torch.matmul(attn, Vh)   # [B,nH,L,dh]
        O = Oh.permute(0,2,1,3).contiguous().view(B,L,d_model)
        O = self.proj_drop(self.out_proj(O))

        # Residual + LN
        return self.out_ln(O + x)
