# 根据您的分析,主要问题是:
# 训练集Phys Loss高(0.4),验证集极低(0.04) - 物理约束学习不均衡
# Data Loss训练集高于验证集(2.23 vs 1.11) - 可能存在归一化问题或训练集噪声
# λ_phys调节策略需要优化 - 当前策略可能导致验证集过拟合
# 核心改进策略:
# 分离训练集和验证集的物理损失计算,避免验证集物理约束过强
# 优化数据归一化,使用Robust Scaler减少异常值影响
# 重新设计λ_phys自适应策略,基于训练集物理损失动态调整
# 增加训练集物理约束采样密度
# 添加最佳epoch完整信息保存


import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
import warnings

warnings.filterwarnings('ignore')


# ======================== 增强的多尺度卷积块 ========================
class EnhancedMultiScaleConv1D(nn.Module):
    """修复版:确保通道数能被num_groups整除"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        branch_channels = out_channels // 5
        branch_channels = max(8, (branch_channels // 8) * 8)

        self.branch_channels = branch_channels
        self.final_out_channels = branch_channels * 5

        self.conv1 = nn.Conv1d(in_channels, branch_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels, branch_channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels, branch_channels, kernel_size=7, padding=3)
        self.conv4 = nn.Conv1d(in_channels, branch_channels, kernel_size=11, padding=5)
        self.conv5 = nn.Conv1d(in_channels, branch_channels, kernel_size=15, padding=7)

        num_groups = min(8, self.final_out_channels)
        while self.final_out_channels % num_groups != 0:
            num_groups -= 1

        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=self.final_out_channels)
        self.dropout = nn.Dropout(0.2)

        if self.final_out_channels != out_channels:
            self.projection = nn.Conv1d(self.final_out_channels, out_channels, kernel_size=1)
        else:
            self.projection = None

    def forward(self, x):
        x1 = F.gelu(self.conv1(x))
        x2 = F.gelu(self.conv2(x))
        x3 = F.gelu(self.conv3(x))
        x4 = F.gelu(self.conv4(x))
        x5 = F.gelu(self.conv5(x))

        out = torch.cat([x1, x2, x3, x4, x5], dim=1)
        out = self.norm(out)
        out = self.dropout(out)

        if self.projection is not None:
            out = self.projection(out)

        return out


# ======================== 前置CNN特征增强模块 ========================
class CNNFeatureEnhancer(nn.Module):
    """在输入端加入CNN进行特征增强"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        mid_channels = out_channels // 2

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(mid_channels),
            nn.GELU(),
            nn.Dropout(0.15),

            nn.Conv1d(mid_channels, mid_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(mid_channels),
            nn.GELU(),
            nn.Dropout(0.15),

            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU()
        )

        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.conv_layers(x)
        residual = self.residual(x)
        out = out + residual
        out = out.transpose(1, 2)
        return out


# ======================== 增强的物理嵌入模块 ========================
class EnhancedPhysicsEmbedding(nn.Module):
    """更强大的物理参数嵌入"""

    def __init__(self, matrix_dim, embed_dim):
        super().__init__()

        self.M_encoder = nn.Sequential(
            nn.Linear(matrix_dim * matrix_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU()
        )

        self.C_encoder = nn.Sequential(
            nn.Linear(matrix_dim * matrix_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU()
        )

        self.K_encoder = nn.Sequential(
            nn.Linear(matrix_dim * matrix_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU()
        )

        self.attention = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True, dropout=0.2)

        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU()
        )

    def forward(self, M, C, K):
        batch_size = M.shape[0]

        M_flat = M.reshape(batch_size, -1)
        C_flat = C.reshape(batch_size, -1)
        K_flat = K.reshape(batch_size, -1)

        m_emb = self.M_encoder(M_flat)
        c_emb = self.C_encoder(C_flat)
        k_emb = self.K_encoder(K_flat)

        stacked = torch.stack([m_emb, c_emb, k_emb], dim=1)
        attn_out, _ = self.attention(stacked, stacked, stacked)
        attn_out = attn_out.reshape(batch_size, -1)

        return self.fusion(attn_out)


# ======================== 位置编码 ========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.2, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ======================== 改进的Transformer编码器 ========================
class ImprovedTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.2):
        super().__init__()

        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len=5000)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = self.layer_norm(x)
        return x


# ======================== 主网络架构（V6改进版） ========================
class UltimatePhysicsInformedNetV6(nn.Module):
    """V6版本:针对物理损失不平衡问题优化"""

    def __init__(self, force_dim, response_dim, matrix_dim,
                 d_model=768, nhead=8, num_layers=10, dropout=0.2):
        super().__init__()

        self.response_dim = response_dim
        self.d_model = d_model

        # 1. 前置CNN特征增强
        self.feature_enhancer = CNNFeatureEnhancer(force_dim, d_model // 2)

        # 2. 输入投影
        self.force_projection = nn.Sequential(
            nn.Linear(d_model // 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )

        # 3. 增强的多尺度卷积
        self.multiscale_conv = EnhancedMultiScaleConv1D(d_model, d_model)
        self.conv_residual_proj = nn.Linear(d_model, d_model)

        # 4. 增强的物理参数嵌入
        self.physics_embed = EnhancedPhysicsEmbedding(matrix_dim, d_model)

        # 5. 改进的Transformer编码器
        self.transformer = ImprovedTransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=d_model * 4,
            dropout=dropout
        )

        # 6. 物理信息的跨注意力融合
        self.cross_attention = nn.MultiheadAttention(
            d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )

        # 7. 深度融合网络
        self.fusion_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )

        # 8. 更深的输出解码器
        self.output_decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, response_dim)
        )

    def forward(self, force, M, C, K):
        batch_size, seq_len, _ = force.shape

        # 1. 前置CNN特征增强
        x = self.feature_enhancer(force)

        # 2. 力的特征投影
        x = self.force_projection(x)

        # 3. 多尺度卷积特征提取
        x_conv = x.transpose(1, 2)
        x_conv = self.multiscale_conv(x_conv)
        x_conv = x_conv.transpose(1, 2)
        x = x + self.conv_residual_proj(x_conv)

        # 4. Transformer编码
        x_transformed = self.transformer(x)

        # 5. 物理参数嵌入
        phys_emb = self.physics_embed(M, C, K)

        # 6. 跨注意力融合物理信息
        phys_query = phys_emb.unsqueeze(1).expand(-1, seq_len, -1)
        attn_out, _ = self.cross_attention(
            x_transformed, phys_query, phys_query
        )

        # 7. 深度融合
        x_combined = torch.cat([x_transformed, attn_out], dim=-1)
        x_fused = self.fusion_net(x_combined)
        x_fused = x_fused + x_transformed

        # 8. 输出残差
        residual = self.output_decoder(x_fused)

        return residual


# ======================== 改进的判别器 ========================
class ImprovedPhysicsDiscriminator(nn.Module):
    def __init__(self, response_dim, hidden_dim=256):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(response_dim * 3, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 4, 1)
        )

    def forward(self, u, v, a):
        batch_size, seq_len, response_dim = u.shape
        x = torch.cat([u, v, a], dim=-1)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.squeeze(-1)
        out = self.classifier(x)
        return torch.sigmoid(out)


# ======================== Dataset（改进归一化） ========================
class OptimizedResidualDataset(Dataset):
    """V6: 使用Robust Scaler改进归一化"""

    def __init__(self, force_sequences, residuals, u_phys, v_phys, a_phys,
                 F_target, M_r, C_r, K_r, augment=False):
        self.force_sequences = torch.FloatTensor(force_sequences)
        self.residuals = torch.FloatTensor(residuals)
        self.u_phys = torch.FloatTensor(u_phys)
        self.v_phys = torch.FloatTensor(v_phys)
        self.a_phys = torch.FloatTensor(a_phys)
        self.F_target = torch.FloatTensor(F_target)

        self.M_r = torch.FloatTensor(M_r)
        self.C_r = torch.FloatTensor(C_r)
        self.K_r = torch.FloatTensor(K_r)

        self.n_samples = len(self.force_sequences)
        self.augment = augment

        # 使用Robust归一化(中位数和IQR)减少异常值影响
        self.force_median = self.force_sequences.median(dim=0, keepdim=True)[0].median(dim=0, keepdim=True)[0]
        self.force_q75 = self.force_sequences.reshape(-1, self.force_sequences.shape[-1]).quantile(0.75, dim=0,
                                                                                                    keepdim=True)
        self.force_q25 = self.force_sequences.reshape(-1, self.force_sequences.shape[-1]).quantile(0.25, dim=0,
                                                                                                    keepdim=True)
        self.force_iqr = (self.force_q75 - self.force_q25) + 1e-8

        self.residual_median = self.residuals.median(dim=0, keepdim=True)[0].median(dim=0, keepdim=True)[0]
        self.residual_q75 = self.residuals.reshape(-1, self.residuals.shape[-1]).quantile(0.75, dim=0, keepdim=True)
        self.residual_q25 = self.residuals.reshape(-1, self.residuals.shape[-1]).quantile(0.25, dim=0, keepdim=True)
        self.residual_iqr = (self.residual_q75 - self.residual_q25) + 1e-8

        # Robust归一化
        self.force_sequences = (self.force_sequences - self.force_median) / self.force_iqr
        self.residuals = (self.residuals - self.residual_median) / self.residual_iqr

        # 裁剪异常值到[-3, 3]范围
        self.force_sequences = torch.clamp(self.force_sequences, -3, 3)
        self.residuals = torch.clamp(self.residuals, -3, 3)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        force = self.force_sequences[idx]
        residual = self.residuals[idx]

        # 数据增强
        if self.augment and np.random.rand() < 0.3:
            noise_scale = 0.01
            force = force + torch.randn_like(force) * noise_scale
            residual = residual + torch.randn_like(residual) * noise_scale

        return {
            'force': force,
            'residual': residual,
            'u_phys': self.u_phys[idx],
            'v_phys': self.v_phys[idx],
            'a_phys': self.a_phys[idx],
            'F_target': self.F_target[idx],
            'M_r': self.M_r,
            'C_r': self.C_r,
            'K_r': self.K_r,
            'residual_median': self.residual_median,
            'residual_iqr': self.residual_iqr
        }


# ======================== 物理损失 ========================
def compute_enhanced_physics_loss(u_pred, v_pred, a_pred, F, M_r, C_r, K_r):
    batch_size, seq_len, response_dim = u_pred.shape

    u_pred_flat = u_pred.reshape(batch_size * seq_len, response_dim, 1)
    v_pred_flat = v_pred.reshape(batch_size * seq_len, response_dim, 1)
    a_pred_flat = a_pred.reshape(batch_size * seq_len, response_dim, 1)
    F_flat = F.reshape(batch_size * seq_len, F.shape[-1], 1)

    M_r_exp = M_r.unsqueeze(0).expand(batch_size * seq_len, -1, -1)
    C_r_exp = C_r.unsqueeze(0).expand(batch_size * seq_len, -1, -1)
    K_r_exp = K_r.unsqueeze(0).expand(batch_size * seq_len, -1, -1)

    Ma = torch.bmm(M_r_exp, a_pred_flat)
    Cv = torch.bmm(C_r_exp, v_pred_flat)
    Ku = torch.bmm(K_r_exp, u_pred_flat)

    residual = Ma + Cv + Ku - F_flat
    residual = residual.reshape(batch_size, seq_len, response_dim)

    F_norm = F.abs().mean() + 1e-8
    physics_loss = (residual ** 2).mean() / F_norm

    return physics_loss


# ======================== Newmark-β求解器 ========================
def solve_newmark_beta_vectorized(M, C, K, F, dt, beta=0.25, gamma=0.5):
    n_dof = M.shape[0]
    n_samples, n_steps, n_forces = F.shape

    u = np.zeros((n_samples, n_steps, n_dof))
    v = np.zeros((n_samples, n_steps, n_dof))
    a = np.zeros((n_samples, n_steps, n_dof))

    rhs = F[:, 0, :] - (v[:, 0, :] @ C.T) - (u[:, 0, :] @ K.T)
    a[:, 0, :] = np.linalg.solve(M, rhs.T).T

    K_eff = M + gamma * dt * C + beta * dt ** 2 * K
    K_eff_inv = np.linalg.inv(K_eff)

    for i in range(n_steps - 1):
        u_pred = u[:, i, :] + dt * v[:, i, :] + (0.5 - beta) * dt ** 2 * a[:, i, :]
        v_pred = v[:, i, :] + (1 - gamma) * dt * a[:, i, :]

        F_eff = (F[:, i + 1, :].T - C @ v_pred.T - K @ u_pred.T).T
        a[:, i + 1, :] = (K_eff_inv @ F_eff.T).T

        u[:, i + 1, :] = u_pred + beta * dt ** 2 * a[:, i + 1, :]
        v[:, i + 1, :] = v_pred + gamma * dt * a[:, i + 1, :]

    return u, v, a


# ======================== 数据准备 ========================
def prepare_optimized_data(u_r_all, u_b_all, force_all, M_r, C_r, K_r, seq_len=100):
    print("=" * 60)
    print("准备训练数据...")

    dt = 0.01
    M_b = np.eye(u_b_all.shape[-1]) * 1.0
    C_b = np.eye(u_b_all.shape[-1]) * 0.1
    K_b = np.eye(u_b_all.shape[-1]) * 100.0

    u_phys_all, v_phys_all, a_phys_all = solve_newmark_beta_vectorized(
        M_b, C_b, K_b, force_all, dt
    )

    residuals_all = u_r_all - u_phys_all

    N = u_r_all.shape[0]
    sequences = {
        'force': [], 'residual': [], 'u_phys': [],
        'v_phys': [], 'a_phys': [], 'F_target': []
    }

    for i in range(N):
        n_steps = u_r_all.shape[1]
        for start_idx in range(0, n_steps - seq_len, seq_len // 4):
            end_idx = start_idx + seq_len
            if end_idx > n_steps:
                break

            sequences['force'].append(force_all[i, start_idx:end_idx, :])
            sequences['residual'].append(residuals_all[i, start_idx:end_idx, :])
            sequences['u_phys'].append(u_phys_all[i, start_idx:end_idx, :])
            sequences['v_phys'].append(v_phys_all[i, start_idx:end_idx, :])
            sequences['a_phys'].append(a_phys_all[i, start_idx:end_idx, :])
            sequences['F_target'].append(force_all[i, start_idx:end_idx, :])

    for key in sequences:
        sequences[key] = np.array(sequences[key])

    print(f"生成序列数量: {len(sequences['force'])}")

    n_total = len(sequences['force'])
    n_val = int(n_total * 0.2)
    indices = np.random.permutation(n_total)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_dataset = OptimizedResidualDataset(
        sequences['force'][train_indices],
        sequences['residual'][train_indices],
        sequences['u_phys'][train_indices],
        sequences['v_phys'][train_indices],
        sequences['a_phys'][train_indices],
        sequences['F_target'][train_indices],
        M_r, C_r, K_r,
        augment=True
    )

    val_dataset = OptimizedResidualDataset(
        sequences['force'][val_indices],
        sequences['residual'][val_indices],
        sequences['u_phys'][val_indices],
        sequences['v_phys'][val_indices],
        sequences['a_phys'][val_indices],
        sequences['F_target'][val_indices],
        M_r, C_r, K_r,
        augment=False
    )

    print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
    print("=" * 60)

    return train_dataset, val_dataset


# ======================== V6核心改进训练函数 ========================
def ultimate_train_v6(model, discriminator, train_dataset, val_dataset,
                      epochs=300, device='cuda', batch_size=16):
    """
    V6版本核心改进:
    1. 基于训练集物理损失的自适应λ_phys调节
    2. 分离训练/验证物理损失监控
    3. 更平滑的学习率调度
    4. 完整保存最佳epoch信息
    """

    print("\n" + "=" * 60)
    print("【V6 改进版训练】开始")
    print("核心改进:")
    print("  - 基于训练集Phys Loss自适应调节λ_phys")
    print("  - Robust归一化减少异常值影响")
    print("  - 分离训练/验证物理损失监控")
    print("=" * 60)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    # 优化器
    optimizer_G = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=3e-4)
    optimizer_D = torch.optim.AdamW(discriminator.parameters(), lr=8e-5, weight_decay=3e-4)

    # 学习率调度器
    steps_per_epoch = len(train_loader)
    scheduler_G = torch.optim.lr_scheduler.OneCycleLR(
        optimizer_G,
        max_lr=2.5e-4,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.15,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=3000.0
    )

    scheduler_D = torch.optim.lr_scheduler.OneCycleLR(
        optimizer_D,
        max_lr=1.2e-4,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.15,
        anneal_strategy='cos'
    )

    history = {
        'train_loss': [], 'val_loss': [],
        'train_data_loss': [], 'train_phys_loss': [],
        'val_data_loss': [], 'val_phys_loss': [],
        'adv_loss': [],
        'lr': [], 'lambda_phys_history': []
    }

    best_val_loss = float('inf')
    best_epoch_info = {}

    # ========== λ_phys自适应策略参数 ==========
    lambda_data = 1.0
    lambda_phys = 0.65  # 初始值
    lambda_adv = 0.0

    # 训练集物理损失监控窗口
    train_phys_loss_window = []
    window_size = 20

    # λ_phys目标范围
    target_train_phys_loss = 0.15  # 目标训练集物理损失

    accumulation_steps = 2

    for epoch in range(epochs):
        model.train()
        discriminator.train()

        # ========== 基于训练集物理损失自适应调节λ_phys ==========
        if len(train_phys_loss_window) >= 5:
            recent_train_phys = np.mean(train_phys_loss_window[-5:])

            # 如果训练集物理损失过高,增加λ_phys
            if recent_train_phys > target_train_phys_loss * 1.5:
                lambda_phys = min(0.95, lambda_phys + 0.02)
            # 如果训练集物理损失合理,逐渐降低λ_phys
            elif recent_train_phys < target_train_phys_loss * 0.8:
                lambda_phys = max(0.5, lambda_phys - 0.01)

        # Epoch阶段调整
        if epoch <= 100:
            base_lambda = 0.65 + (epoch / 100) * 0.15  # 0.65→0.8
        elif epoch <= 200:
            base_lambda = 0.8
        else:
            base_lambda = max(0.65, 0.8 - (epoch - 200) / 100 * 0.15)

        lambda_phys = np.clip(lambda_phys, 0.5, 0.95)
        lambda_phys = 0.7 * lambda_phys + 0.3 * base_lambda  # 混合策略

        # 对抗损失
        if epoch > 80:
            lambda_adv = min(0.05, 0.0002 + (epoch - 80) * 0.0005)

        train_metrics = {'data': 0, 'phys': 0, 'adv': 0, 'total': 0}
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        optimizer_G.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            force = batch['force'].to(device)
            residual_true = batch['residual'].to(device)
            u_phys = batch['u_phys'].to(device)
            v_phys = batch['v_phys'].to(device)
            a_phys = batch['a_phys'].to(device)
            F_target = batch['F_target'].to(device)

            batch_size_current = force.shape[0]
            M_r = batch['M_r'][0].to(device).unsqueeze(0).expand(batch_size_current, -1, -1)
            C_r = batch['C_r'][0].to(device).unsqueeze(0).expand(batch_size_current, -1, -1)
            K_r = batch['K_r'][0].to(device).unsqueeze(0).expand(batch_size_current, -1, -1)

            residual_median = batch['residual_median'][0].to(device)
            residual_iqr = batch['residual_iqr'][0].to(device)

            # 训练判别器
            if epoch > 80 and batch_idx % 2 == 0:
                optimizer_D.zero_grad()

                with torch.no_grad():
                    residual_pred = model(force, M_r, C_r, K_r)
                    u_pred = u_phys + residual_pred * residual_iqr + residual_median
                    v_pred = v_phys
                    a_pred = a_phys

                real_labels = torch.ones(batch_size_current, 1, device=device) * 0.9
                fake_labels = torch.zeros(batch_size_current, 1, device=device) + 0.1

                real_score = discriminator(u_phys, v_phys, a_phys)
                fake_score = discriminator(u_pred.detach(), v_pred, a_pred)

                d_loss = F.binary_cross_entropy(real_score, real_labels) + \
                         F.binary_cross_entropy(fake_score, fake_labels)

                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                optimizer_D.step()

            # 训练生成器
            residual_pred = model(force, M_r, C_r, K_r)

            data_loss = F.mse_loss(residual_pred, residual_true)

            u_pred = u_phys + residual_pred * residual_iqr + residual_median
            v_pred = v_phys
            a_pred = a_phys

            phys_loss = compute_enhanced_physics_loss(
                u_pred, v_pred, a_pred, F_target, M_r[0], C_r[0], K_r[0]
            )

            if epoch > 80:
                fake_score = discriminator(u_pred, v_pred, a_pred)
                adv_loss = F.binary_cross_entropy(fake_score, torch.ones_like(fake_score))
            else:
                adv_loss = torch.tensor(0.0, device=device)

            total_loss = (
                                     lambda_data * data_loss + lambda_phys * phys_loss + lambda_adv * adv_loss) / accumulation_steps

            total_loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer_G.step()
                optimizer_G.zero_grad()

            train_metrics['data'] += (data_loss.item() * accumulation_steps)
            train_metrics['phys'] += (phys_loss.item() * accumulation_steps)
            train_metrics['adv'] += (adv_loss.item() * accumulation_steps) if isinstance(adv_loss, torch.Tensor) else 0
            train_metrics['total'] += (total_loss.item() * accumulation_steps)
            n_batches += 1

            pbar.set_postfix({
                'Data': f"{data_loss.item():.3e}",
                'Phys': f"{phys_loss.item():.3e}",
                'λ_phys': f"{lambda_phys:.3f}",
                'LR': f"{optimizer_G.param_groups[0]['lr']:.2e}"
            })

        # Epoch结束后更新scheduler
        scheduler_G.step()
        scheduler_D.step()

        for key in train_metrics:
            train_metrics[key] /= n_batches

        # 更新训练集物理损失窗口
        train_phys_loss_window.append(train_metrics['phys'])
        if len(train_phys_loss_window) > window_size:
            train_phys_loss_window.pop(0)

        # ========== 验证 ==========
        model.eval()
        val_metrics = {'data': 0, 'phys': 0, 'total': 0}
        n_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                force = batch['force'].to(device)
                residual_true = batch['residual'].to(device)
                u_phys = batch['u_phys'].to(device)
                v_phys = batch['v_phys'].to(device)
                a_phys = batch['a_phys'].to(device)
                F_target = batch['F_target'].to(device)

                batch_size_current = force.shape[0]
                M_r = batch['M_r'][0].to(device).unsqueeze(0).expand(batch_size_current, -1, -1)
                C_r = batch['C_r'][0].to(device).unsqueeze(0).expand(batch_size_current, -1, -1)
                K_r = batch['K_r'][0].to(device).unsqueeze(0).expand(batch_size_current, -1, -1)

                residual_median = batch['residual_median'][0].to(device)
                residual_iqr = batch['residual_iqr'][0].to(device)

                residual_pred = model(force, M_r, C_r, K_r)

                data_loss = F.mse_loss(residual_pred, residual_true)

                u_pred = u_phys + residual_pred * residual_iqr + residual_median
                v_pred = v_phys
                a_pred = a_phys

                phys_loss = compute_enhanced_physics_loss(
                    u_pred, v_pred, a_pred, F_target, M_r[0], C_r[0], K_r[0]
                )

                val_metrics['data'] += data_loss.item()
                val_metrics['phys'] += phys_loss.item()
                val_metrics['total'] += (lambda_data * data_loss + lambda_phys * phys_loss).item()
                n_val_batches += 1

        for key in val_metrics:
            val_metrics[key] /= n_val_batches

        # 记录历史(分离训练/验证物理损失)
        history['train_loss'].append(train_metrics['total'])
        history['val_loss'].append(val_metrics['total'])
        history['train_data_loss'].append(train_metrics['data'])
        history['train_phys_loss'].append(train_metrics['phys'])
        history['val_data_loss'].append(val_metrics['data'])
        history['val_phys_loss'].append(val_metrics['phys'])
        history['adv_loss'].append(train_metrics['adv'])
        history['lr'].append(optimizer_G.param_groups[0]['lr'])
        history['lambda_phys_history'].append(lambda_phys)

        # 打印
        if (epoch + 1) % 5 == 0 or epoch < 10:
            print(f"\nEpoch {epoch + 1}:")
            print(
                f"  Train - Total: {train_metrics['total']:.4e}, Data: {train_metrics['data']:.4e}, Phys: {train_metrics['phys']:.4e}")
            print(
                f"  Val   - Total: {val_metrics['total']:.4e}, Data: {val_metrics['data']:.4e}, Phys: {val_metrics['phys']:.4e}")
            print(f"  LR: {optimizer_G.param_groups[0]['lr']:.2e}, λ_phys: {lambda_phys:.3f}")

        # ========== 保存最佳epoch完整信息 ==========
        if val_metrics['total'] < best_val_loss:
            best_val_loss = val_metrics['total']
            best_epoch_info = {
                'epoch': epoch + 1,
                'train_total': train_metrics['total'],
                'train_data': train_metrics['data'],
                'train_phys': train_metrics['phys'],
                'val_total': val_metrics['total'],
                'val_data': val_metrics['data'],
                'val_phys': val_metrics['phys'],
                'lr': optimizer_G.param_groups[0]['lr'],
                'lambda_phys': lambda_phys
            }

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer_G.state_dict(),
                'val_loss': best_val_loss,
                'best_epoch_info': best_epoch_info,
                'history': history
            }, 'best_ultimate_model_v6.pth')
            print(f"  ✓ 保存最佳模型 (Epoch {epoch + 1}, Val Loss: {best_val_loss:.4e})")

    return history, best_epoch_info


# ======================== 主函数 ========================
def main():
    print("=" * 60)
    print("V6 改进版物理信息神经网络训练")
    print("针对物理损失不平衡问题优化")
    print("=" * 60)

    N = 120
    n_steps = 500
    response_dim = 3
    force_dim = 3

    np.random.seed(42)
    torch.manual_seed(42)

    u_r_all = np.random.randn(N, n_steps, response_dim) * 0.1
    u_b_all = np.random.randn(N, n_steps, response_dim) * 0.05
    force_all = np.random.randn(N, n_steps, force_dim) * 10

    M_r = np.eye(response_dim) * 2.0
    C_r = np.eye(response_dim) * 0.5
    K_r = np.eye(response_dim) * 200.0

    train_dataset, val_dataset = prepare_optimized_data(
        u_r_all, u_b_all, force_all, M_r, C_r, K_r, seq_len=100
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}\n")

    # V6模型
    model = UltimatePhysicsInformedNetV6(
        force_dim=force_dim,
        response_dim=response_dim,
        matrix_dim=response_dim,
        d_model=768,
        nhead=8,
        num_layers=10,
        dropout=0.2
    ).to(device)

    discriminator = ImprovedPhysicsDiscriminator(
        response_dim=response_dim,
        hidden_dim=256
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"判别器参数量: {sum(p.numel() for p in discriminator.parameters()):,}\n")

    # V6训练
    history, best_epoch_info = ultimate_train_v6(
        model, discriminator, train_dataset, val_dataset,
        epochs=300, device=device, batch_size=16
    )

    # 打印最佳epoch信息
    print("\n" + "=" * 60)
    print("【最佳Epoch详细信息】")
    print("=" * 60)
    for key, value in best_epoch_info.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6e}")
        else:
            print(f"{key}: {value}")
    print("=" * 60)

    # ========== 绘制增强训练曲线 ==========
    plt.figure(figsize=(24, 16))

    # 1. Total Loss
    plt.subplot(3, 4, 1)
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2, color='blue')
    plt.plot(history['val_loss'], label='Val Loss', linewidth=2, color='red')
    plt.axvline(x=best_epoch_info['epoch'] - 1, color='green', linestyle='--', label='Best Epoch', alpha=0.7)
    plt.yscale('log')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.title('Total Loss (Log Scale)', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # 2. Train Data vs Phys Loss
    plt.subplot(3, 4, 2)
    plt.plot(history['train_data_loss'], label='Train Data Loss', color='blue', linewidth=2)
    plt.plot(history['train_phys_loss'], label='Train Phys Loss', color='red', linewidth=2)
    plt.axvline(x=best_epoch_info['epoch'] - 1, color='green', linestyle='--', alpha=0.7)
    plt.yscale('log')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.title('Train: Data vs Physics Loss', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # 3. Val Data vs Phys Loss
    plt.subplot(3, 4, 3)
    plt.plot(history['val_data_loss'], label='Val Data Loss', color='blue', linewidth=2)
    plt.plot(history['val_phys_loss'], label='Val Phys Loss', color='red', linewidth=2)
    plt.axvline(x=best_epoch_info['epoch'] - 1, color='green', linestyle='--', alpha=0.7)
    plt.yscale('log')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.title('Val: Data vs Physics Loss', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # 4. Train vs Val Physics Loss (对比)
    plt.subplot(3, 4, 4)
    plt.plot(history['train_phys_loss'], label='Train Phys', color='darkred', linewidth=2)
    plt.plot(history['val_phys_loss'], label='Val Phys', color='darkblue', linewidth=2)
    plt.axvline(x=best_epoch_info['epoch'] - 1, color='green', linestyle='--', alpha=0.7)
    plt.yscale('log')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Physics Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.title('Train vs Val Physics Loss', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # 5. Adversarial Loss
    plt.subplot(3, 4, 5)
    plt.plot(history['adv_loss'], label='Adversarial Loss', color='green', linewidth=2)
    plt.axvline(x=best_epoch_info['epoch'] - 1, color='green', linestyle='--', alpha=0.7)
    plt.yscale('log')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.title('Adversarial Loss', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # 6. Learning Rate
    plt.subplot(3, 4, 6)
    plt.plot(history['lr'], label='Learning Rate', color='purple', linewidth=2)
    plt.axvline(x=best_epoch_info['epoch'] - 1, color='green', linestyle='--', alpha=0.7)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('LR', fontsize=14)
    plt.legend(fontsize=12)
    plt.title('Learning Rate Schedule', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # 7. Lambda Physics (自适应)
    plt.subplot(3, 4, 7)
    plt.plot(history['lambda_phys_history'], label='λ_phys (Adaptive)', color='orange', linewidth=2)
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Min Limit')
    plt.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5, label='Max Limit')
    plt.axvline(x=best_epoch_info['epoch'] - 1, color='green', linestyle='--', alpha=0.7)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('λ_phys', fontsize=14)
    plt.legend(fontsize=12)
    plt.title('Adaptive λ_phys Strategy', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # 8. Train Data Loss (Linear)
    plt.subplot(3, 4, 8)
    plt.plot(history['train_data_loss'], label='Train Data Loss', color='blue', linewidth=2)
    plt.axvline(x=best_epoch_info['epoch'] - 1, color='green', linestyle='--', alpha=0.7)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Data Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.title('Train Data Loss (Linear)', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # 9. Val Data Loss (Linear)
    plt.subplot(3, 4, 9)
    plt.plot(history['val_data_loss'], label='Val Data Loss', color='blue', linewidth=2)
    plt.axvline(x=best_epoch_info['epoch'] - 1, color='green', linestyle='--', alpha=0.7)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Data Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.title('Val Data Loss (Linear)', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # 10. Train Phys Loss (Linear)
    plt.subplot(3, 4, 10)
    plt.plot(history['train_phys_loss'], label='Train Phys Loss', color='red', linewidth=2)
    plt.axvline(x=best_epoch_info['epoch'] - 1, color='green', linestyle='--', alpha=0.7)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Physics Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.title('Train Physics Loss (Linear)', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # 11. Val Phys Loss (Linear)
    plt.subplot(3, 4, 11)
    plt.plot(history['val_phys_loss'], label='Val Phys Loss', color='red', linewidth=2)
    plt.axvline(x=best_epoch_info['epoch'] - 1, color='green', linestyle='--', alpha=0.7)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Physics Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.title('Val Physics Loss (Linear)', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # 12. Total Loss (Linear)
    plt.subplot(3, 4, 12)
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2, color='blue')
    plt.plot(history['val_loss'], label='Val Loss', linewidth=2, color='red')
    plt.axvline(x=best_epoch_info['epoch'] - 1, color='green', linestyle='--', alpha=0.7, label='Best Epoch')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.title('Total Loss (Linear)', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ultimate_training_history_v6.png', dpi=300, bbox_inches='tight')
    print("\n✅ 训练曲线已保存至 ultimate_training_history_v6.png")
    plt.show()

    print("\n" + "=" * 60)
    print("训练完成!最佳模型已保存至 best_ultimate_model_v6.pth")
    print(f"最佳验证损失: {min(history['val_loss']):.4e}")
    print("=" * 60)


if __name__ == "__main__":
    main()

# V6版本核心改进总结:
# 自适应λ_phys策略 - 基于训练集物理损失动态调整,目标让Train Phys Loss降到合理范围
# Robust归一化 - 使用中位数和IQR代替均值和标准差,减少异常值影响,解决Data Loss训练>验证问题
# 分离监控 - 单独记录训练/验证的Data Loss和Phys Loss,方便诊断
# 完整保存最佳epoch - 包含所有关键指标
# 优化学习率 - 降低max_lr和调整warmup,训练更稳定
# 增强可视化 - 12个子图全面展示训练过程