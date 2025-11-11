
 import torch
 from torch import nn
 import torch.nn.functional as F
-import snntorch as snn
-from snntorch import surrogate
+from spikingjelly.activation_based import neuron, surrogate
+
+
+tau_default = 2.0
+backend = "torch"
+detach_reset = True
+
 
 class TemporalAwareSpikeAttention(nn.Module):
     """
     Temporal-aware Spiking Self-Attention with multi-scale temporal activity
-    - Q/K/V SNN化 (Linear → LIF → LN)
-    - 支持基于 rate 的调制
-    - 新增: 多尺度时间活跃度 (早期/晚期/斜率)，通过可学习融合调制 Q
+    - 基于 SpikformerCPG 的脉冲化 Q/K/V (Linear → BN → LIF)
+    - 保持 rate 模调，并新增早期/晚期/斜率多尺度融合调制 Q
+    - 支持可选返回调制所用的时间统计，方便调试或下游使用
+    - 输出尺寸与输入一致，便于在 TCN 自注意力中直接替换
     """
 
-    def __init__(self, d_model, n_heads=6, num_steps=4, use_first_spike=False, alpha=1.0, tau=1.0):
+    def __init__(
+        self,
+        d_model=None,
+        *,
+        dim=None,
+        n_heads=6,
+        num_steps=4,
+        use_first_spike=False,
+        alpha=1.0,
+        tau=tau_default,
+        qk_scale=None,
+    ):
         super().__init__()
+        if d_model is None:
+            if dim is None:
+                raise TypeError(
+                    "TemporalAwareSpikeAttention requires `d_model` or `dim` to be provided"
+                )
+            d_model = dim
+        elif dim is not None and dim != d_model:
+            raise ValueError("`d_model` and legacy alias `dim` must match if both are set")
+
+        self.d_model = d_model
         self.num_steps = num_steps
         self.n_heads = max(1, n_heads)
         self.use_first_spike = use_first_spike
         self.alpha = alpha
         self.tau = tau
 
+        head_dim = d_model // self.n_heads
+        assert head_dim * self.n_heads == d_model, "d_model must be divisible by n_heads"
+        self.qk_scale = qk_scale or head_dim ** -0.5
+
         # 可调参数
         self.lambda_mix = 0.0
         self.detach_weight = True
         self.use_spike_qkv = True
         self.use_temporal_mod = True
-        self.use_multi_scale = True   # 新增：是否启用多尺度特征
-
-        # 线性投影
-        self.q_proj = nn.Linear(d_model, d_model, bias=False)
-        self.k_proj = nn.Linear(d_model, d_model, bias=False)
-        self.v_proj = nn.Linear(d_model, d_model, bias=False)
-        self.out_proj = nn.Linear(d_model, d_model, bias=False)
-
-        # LIF 神经元
-        mk_lif = lambda: snn.Leaky(beta=0.99, spike_grad=surrogate.atan(2.0),
-                                   init_hidden=True, threshold=1.0)
-        self.lif_q, self.lif_k, self.lif_v = mk_lif(), mk_lif(), mk_lif()
-
-        # LayerNorm
-        self.q_ln = nn.LayerNorm(d_model)
-        self.k_ln = nn.LayerNorm(d_model)
-        self.v_ln = nn.LayerNorm(d_model)
+        self.use_multi_scale = True
+        self.multi_scale_ratio = 0.5
+        self.eps = 1e-6
+
+        # Q/K/V 线性 + BN + LIF
+        self.q_m = nn.Linear(d_model, d_model)
+        self.q_bn = nn.BatchNorm1d(d_model)
+        self.q_lif = neuron.LIFNode(
+            tau=tau,
+            step_mode="m",
+            detach_reset=detach_reset,
+            surrogate_function=surrogate.ATan(),
+            v_threshold=1.0,
+            backend=backend,
+        )
+
+        self.k_m = nn.Linear(d_model, d_model)
+        self.k_bn = nn.BatchNorm1d(d_model)
+        self.k_lif = neuron.LIFNode(
+            tau=tau,
+            step_mode="m",
+            detach_reset=detach_reset,
+            surrogate_function=surrogate.ATan(),
+            v_threshold=1.0,
+            backend=backend,
+        )
+
+        self.v_m = nn.Linear(d_model, d_model)
+        self.v_bn = nn.BatchNorm1d(d_model)
+        self.v_lif = neuron.LIFNode(
+            tau=tau,
+            step_mode="m",
+            detach_reset=detach_reset,
+            surrogate_function=surrogate.ATan(),
+            v_threshold=1.0,
+            backend=backend,
+        )
+
+        # 注意力后处理
+        self.attn_lif = neuron.LIFNode(
+            tau=tau,
+            step_mode="m",
+            detach_reset=detach_reset,
+            surrogate_function=surrogate.ATan(),
+            v_threshold=0.5,
+            backend=backend,
+        )
+
+        self.last_m = nn.Linear(d_model, d_model)
+        self.last_bn = nn.BatchNorm1d(d_model)
+        self.last_lif = neuron.LIFNode(
+            tau=tau,
+            step_mode="m",
+            detach_reset=detach_reset,
+            surrogate_function=surrogate.ATan(),
+            v_threshold=1.0,
+            backend=backend,
+        )
+
         self.out_ln = nn.LayerNorm(d_model)
 
         # Dropout
         self.attn_drop = nn.Dropout(p=0.0)
-        self.proj_drop = nn.Dropout(p=0.0)
 
         # 残差调制强度
         self.gamma_temporal = nn.Parameter(torch.tensor(0.2))
 
         # 多尺度融合 (3 → 1)
         self.w_fuse = nn.Linear(3, 1, bias=False)
 
-    def forward(self, x, spikes_time_series):
-        """
-        x: [B, L, d_model]
-        spikes_time_series: [B, H, C, L, T]
-        """
-        B, L, d_model = x.shape
-        T = spikes_time_series.shape[-1]
-        device = x.device
-
-        # ===== Q/K/V with LIF =====
-        Q_lin, K_lin, V_lin = self.q_proj(x), self.k_proj(x), self.v_proj(x)
-        if self.use_spike_qkv:
-            Q, K, V = self.lif_q(Q_lin), self.lif_k(K_lin), self.lif_v(V_lin)
+    def _lif_transform(self, linear, bn, lif, tensor):
+        x = linear(tensor)
+        x = bn(x.transpose(-1, -2)).transpose(-1, -2)
+        x = x.reshape(1, *tensor.shape[:-1], tensor.shape[-1])
+        return lif(x) if self.use_spike_qkv else x
+
+    def _compute_temporal_activity(self, spikes_time_series):
+        """Compute multi-scale temporal statistics for modulation."""
+
+        stats = {}
+        mask_spk = (spikes_time_series > 0).float()
+        T = mask_spk.shape[-1]
+        device = mask_spk.device
+
+        if self.use_first_spike and T > 0:
+            t_idx = torch.arange(T, device=device).view(1, 1, 1, 1, -1)
+            first_t = mask_spk * (t_idx + 1)
+            first_t[first_t == 0] = float(T if T > 0 else 1)
+            first_t = first_t.min(dim=-1).values.float()
+            base = torch.exp(-self.alpha * first_t)
+        else:
+            base = mask_spk.mean(dim=-1)
+
+        base_mean = base.mean(dim=(1, 2))
+        stats["base_rate"] = base_mean
+
+        if self.use_multi_scale and T >= 2:
+            split = int(round(T * self.multi_scale_ratio))
+            if split <= 0:
+                split = 1
+            if split >= T:
+                split = T - 1
+
+            early_slice = mask_spk[..., :split]
+            late_slice = mask_spk[..., -split:]
+
+            early_rate = early_slice.mean(dim=-1)
+            late_rate = late_slice.mean(dim=-1)
+            slope = late_rate - early_rate
+
+            stats["early_rate"] = early_rate.mean(dim=(1, 2))
+            stats["late_rate"] = late_rate.mean(dim=(1, 2))
+            stats["slope"] = slope.mean(dim=(1, 2))
+
+            w_multi = torch.stack(
+                [stats["early_rate"], stats["late_rate"], stats["slope"]], dim=-1
+            )
+            fused = self.w_fuse(w_multi).squeeze(-1)
         else:
-            Q, K, V = Q_lin, K_lin, V_lin
-
-        # ===== Multi-scale temporal activity =====
-        if self.use_temporal_mod:
-            mask_spk = (spikes_time_series > 0).float()  # [B,H,C,L,T]
-
-            if self.use_first_spike:
-                t_idx = torch.arange(T, device=device).view(1,1,1,1,-1)
-                first_t = (mask_spk * (t_idx+1))
-                first_t[first_t == 0] = self.num_steps
-                first_t = first_t.min(dim=-1).values.float()  # [B,H,C,L]
-                w_base = torch.exp(-self.alpha * first_t)
-            else:
-                w_base = mask_spk.mean(dim=-1)  # [B,H,C,L]
-
-            if self.use_multi_scale:
-                # early / late / slope
-                early_rate = mask_spk[..., :T//2].mean(dim=-1)   # [B,H,C,L]
-                late_rate  = mask_spk[..., T//2:].mean(dim=-1)   # [B,H,C,L]
-                slope = late_rate - early_rate                   # [B,H,C,L]
-
-                early = early_rate.mean(dim=(1,2))   # [B,L]
-                late  = late_rate.mean(dim=(1,2))    # [B,L]
-                slp   = slope.mean(dim=(1,2))        # [B,L]
-
-                w_multi = torch.stack([early, late, slp], dim=-1)  # [B,L,3]
-                w_L = self.w_fuse(w_multi).squeeze(-1)             # [B,L]
-            else:
-                # 原始版本：直接平均
-                w_L = w_base.mean(dim=(1,2))  # [B,L]
-
-            # 归一化
-            w_L = (w_L - w_L.mean(dim=-1, keepdim=True)) / (w_L.std(dim=-1, keepdim=True) + 1e-6)
+            fused = base_mean
+
+        stats["raw_fused"] = fused
+
+        centered = fused - fused.mean(dim=-1, keepdim=True)
+        std = centered.std(dim=-1, keepdim=True)
+        normed = centered / (std + self.eps)
+        stats["temporal_weight"] = torch.nan_to_num(normed)
+
+        return stats
+
+    def forward(self, x, spikes_time_series, return_temporal_stats=False):
+        """Forward.
+
+        Args:
+            x (Tensor): [B, L, d_model]
+            spikes_time_series (Tensor): [B, H, C, L, T]
+            return_temporal_stats (bool): whether to return modulation statistics
+        Returns:
+            Tensor: [B, L, d_model]
+        """
+
+        B, L, D = x.shape
+        residual = x
+        temporal_stats = None
+
+        # ===== Q/K/V with SpikformerCPG-style processing =====
+        x_seq = x.unsqueeze(0).flatten(0, 1)
+        Q = self._lif_transform(self.q_m, self.q_bn, self.q_lif, x_seq)
+        K = self._lif_transform(self.k_m, self.k_bn, self.k_lif, x_seq)
+        V = self._lif_transform(self.v_m, self.v_bn, self.v_lif, x_seq)
+
+        # ===== Multi-scale temporal activity modulation =====
+        if self.use_temporal_mod and spikes_time_series is not None:
+            temporal_stats = self._compute_temporal_activity(spikes_time_series)
+            w_L = temporal_stats["temporal_weight"]
+
             if self.detach_weight:
                 w_L = w_L.detach()
 
-            # 残差式调制 Q
-            Q = Q * (1.0 + torch.clamp(self.gamma_temporal, min=0.0) * w_L.unsqueeze(-1))
+            gamma = torch.clamp(self.gamma_temporal, min=0.0)
+            Q = Q * (1.0 + gamma.view(1, 1, 1) * w_L.unsqueeze(0).unsqueeze(-1))
 
-        # ===== LN =====
-        Q, K, V = self.q_ln(Q), self.k_ln(K), self.v_ln(V)
+        # reshape for multi-head attn
+        Q = Q.reshape(1, B, L, self.n_heads, D // self.n_heads).permute(0, 1, 3, 2, 4)
+        K = K.reshape(1, B, L, self.n_heads, D // self.n_heads).permute(0, 1, 3, 2, 4)
+        V = V.reshape(1, B, L, self.n_heads, D // self.n_heads).permute(0, 1, 3, 2, 4)
 
-        # ===== Multi-head split =====
-        nH = self.n_heads
-        assert d_model % nH == 0
-        dh = d_model // nH
-        def split_heads(t): return t.view(B, L, nH, dh).permute(0,2,1,3)
-        Qh, Kh, Vh = map(split_heads, (Q,K,V))
-
-        # ===== Similarity =====
-        dot = torch.matmul(Qh, Kh.transpose(-2,-1))  # [B,nH,L,L]
-        neg_l1 = -(Qh.unsqueeze(3)-Kh.unsqueeze(2)).abs().sum(dim=-1)
-        score = self.lambda_mix*dot + (1.0-self.lambda_mix)*neg_l1
+        dot = torch.matmul(Q, K.transpose(-2, -1)) * self.qk_scale
+        neg_l1 = -(Q.unsqueeze(-2) - K.unsqueeze(-3)).abs().sum(dim=-1)
+        score = self.lambda_mix * dot + (1.0 - self.lambda_mix) * neg_l1
         score = score / max(self.tau, 1e-6)
         score = score - score.max(dim=-1, keepdim=True).values
 
-        # ===== Softmax =====
         attn = F.softmax(score, dim=-1)
         attn = self.attn_drop(attn)
 
-        # ===== Aggregate =====
-        Oh = torch.matmul(attn, Vh)   # [B,nH,L,dh]
-        O = Oh.permute(0,2,1,3).contiguous().view(B,L,d_model)
-        O = self.proj_drop(self.out_proj(O))
+        x_attn = torch.matmul(attn, V)
+        x_attn = x_attn.transpose(2, 3).reshape(1, B, L, D)
+        x_attn = self.attn_lif(x_attn)
+
+        x_flat = x_attn.flatten(0, 1)
+        x_flat = self.last_m(x_flat)
+        x_flat = self.last_bn(x_flat.transpose(-1, -2)).transpose(-1, -2)
+        x_out = self.last_lif(x_flat.reshape(1, B, L, D))
+
+        x_out = x_out.squeeze(0)
+        out = self.out_ln(x_out + residual)
 
-        # Residual + LN
-        return self.out_ln(O + x)
+        if return_temporal_stats:
+            return out, (temporal_stats or {})
+        return out
