"""
Sign Language Recognition (SLR) Model
Based on: "From Scarcity to Understanding: Transfer Learning for the
Extremely Low Resource Irish Sign Language" (Holmes et al., ICCVW 2023)

Architecture:
  1. Depthwise 1D CNN  – local temporal patterns per keypoint coordinate
  2. Frame Embeddings  – non-linear relationships across keypoints per frame
  3. Depthwise 1D CNN  – second-pass local temporal patterns on embeddings
  4. Self-Attention     – global temporal context
  5. Classifier         – final linear head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResidualDepthwiseConv1d(nn.Module):
    """
    A single residual depthwise 1D convolution block.
    Depthwise means each channel is convolved independently (groups=channels),
    keeping per-feature temporal patterns separate.
    """
    def __init__(self, channels: int, kernel_size: int, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2  # 'same' padding to preserve sequence length
        self.conv = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,   # depthwise
        )
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        residual = x
        # Conv1d expects (B, C, T)
        out = self.conv(x.transpose(1, 2)).transpose(1, 2)
        out = self.dropout(F.gelu(self.norm(out)))
        return out + residual


class Stage1LocalCNN(nn.Module):
    """
    Stage 1: Four depthwise 1D conv layers with increasing kernel sizes
    (3, 5, 7, 9) applied independently to each input feature (coordinate).

    Input  x: (B, T, input_dim)   where input_dim = num_keypoints * 2 (x,y)
    Output  : (B, T, input_dim)
    """
    def __init__(self, input_dim: int, dropout: float = 0.1):
        super().__init__()
        kernel_sizes = [3, 5, 7, 9]
        self.layers = nn.ModuleList([
            ResidualDepthwiseConv1d(input_dim, k, dropout)
            for k in kernel_sizes
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class FrameEmbeddingBlock(nn.Module):
    """
    One block of the frame embedding sub-network.
    Linear → LayerNorm → GELU → Dropout
    The last block omits GELU and instead adds a residual from Stage 1.
    """
    def __init__(self, in_dim: int, out_dim: int,
                 use_gelu: bool = True, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm   = nn.LayerNorm(out_dim)
        self.drop   = nn.Dropout(dropout)
        self.use_gelu = use_gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.norm(self.linear(x))
        if self.use_gelu:
            out = F.gelu(out)
        out = self.drop(out)
        return out


class Stage2FrameEmbedding(nn.Module):
    """
    Stage 2: Four embedding blocks applied frame-by-frame (same weights
    across time — like a shared MLP over each frame independently).

    The final block has no GELU and adds a residual from Stage 1 output
    (projected to embedding_dim if needed).

    Input  : (B, T, input_dim)   — Stage 1 output
    Output : (B, T, embedding_dim)
    """
    def __init__(self, input_dim: int, embedding_dim: int, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            FrameEmbeddingBlock(input_dim,    embedding_dim, use_gelu=True,  dropout=dropout),
            FrameEmbeddingBlock(embedding_dim, embedding_dim, use_gelu=True,  dropout=dropout),
            FrameEmbeddingBlock(embedding_dim, embedding_dim, use_gelu=True,  dropout=dropout),
            FrameEmbeddingBlock(embedding_dim, embedding_dim, use_gelu=False, dropout=dropout),
        ])
        # Project Stage 1 output to embedding_dim for the residual connection
        self.residual_proj = (
            nn.Linear(input_dim, embedding_dim)
            if input_dim != embedding_dim else nn.Identity()
        )

    def forward(self, x: torch.Tensor, stage1_out: torch.Tensor) -> torch.Tensor:
        # First three blocks
        out = x
        for block in self.blocks[:3]:
            out = block(out)
        # Final block: no GELU + residual from Stage 1
        out = self.blocks[3](out) + self.residual_proj(stage1_out)
        return out


class Stage3LocalCNN(nn.Module):
    """
    Stage 3: Second stack of residual depthwise 1D CNNs over the embeddings.
    Each block now has TWO conv layers with GELU activations in between.

    Input / Output: (B, T, embedding_dim)
    """
    def __init__(self, embedding_dim: int, num_blocks: int = 4, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            self._make_block(embedding_dim, dropout) for _ in range(num_blocks)
        ])

    @staticmethod
    def _make_block(channels: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.LayerNorm(channels),   # applied after transpose inside forward
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.LayerNorm(channels),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            residual = x
            # Each block is a Sequential; we need manual forward for the transposes
            out = x.transpose(1, 2)                          # (B, C, T)
            conv1, norm1, gelu, drop1, conv2, norm2, drop2 = block
            out = conv1(out).transpose(1, 2)                 # back to (B, T, C)
            out = drop1(gelu(norm1(out)))
            out = conv2(out.transpose(1, 2)).transpose(1, 2) # (B, T, C)
            out = drop2(norm2(out))
            x = out + residual
        return x


class Stage4SelfAttention(nn.Module):
    """
    Stage 4: Multi-head self-attention over the full sequence for global
    temporal context. Includes a feedforward sub-layer (standard Transformer
    encoder layer).

    Input / Output: (B, T, embedding_dim)
    """
    def __init__(self, embedding_dim: int, num_heads: int,
                 num_layers: int, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,    # (B, T, C) convention
            norm_first=True,     # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor,
                src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.transformer(x, src_key_padding_mask=src_key_padding_mask)


# ---------------------------------------------------------------------------
# Full SLR Model
# ---------------------------------------------------------------------------

class SLRModel(nn.Module):
    """
    Full Sign Language Recognition model as described in Holmes et al. 2023.

    Args:
        num_keypoints   : number of pose keypoints (paper uses 67)
        coords_per_kp   : coordinates per keypoint, default 2 (x, y)
        num_classes     : number of sign classes for the classification head
        embedding_dim   : size of frame embeddings (paper: 192)
        num_attn_layers : number of Transformer encoder layers (paper: 4)
        num_attn_heads  : number of attention heads (paper: 8)
        dropout         : dropout probability

    Input shape:
        x : (batch, time, num_keypoints * coords_per_kp)
        padding_mask : (batch, time) bool mask — True where frames are padding

    Output:
        logits : (batch, num_classes)
    """
    def __init__(
        self,
        num_keypoints: int   = 67,
        coords_per_kp: int   = 2,
        num_classes: int     = 224,
        embedding_dim: int   = 192,
        num_attn_layers: int = 4,
        num_attn_heads: int  = 8,
        dropout: float       = 0.1,
    ):
        super().__init__()
        input_dim = num_keypoints * coords_per_kp   # 134 with default args

        # Stage 1 – per-coordinate local temporal CNN
        self.stage1 = Stage1LocalCNN(input_dim, dropout=dropout)

        # Stage 2 – frame-level embedding network
        self.stage2 = Stage2FrameEmbedding(input_dim, embedding_dim, dropout=dropout)

        # Stage 3 – second local temporal CNN on embeddings
        self.stage3 = Stage3LocalCNN(embedding_dim, num_blocks=num_attn_layers, dropout=dropout)

        # Stage 4 – global self-attention
        self.stage4 = Stage4SelfAttention(
            embedding_dim, num_attn_heads, num_attn_layers, dropout=dropout
        )

        # Stage 5 – classifier head
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x            : (B, T, input_dim) — normalised keypoint sequences
            padding_mask : (B, T) bool — True for padded (invalid) frames
        Returns:
            logits       : (B, num_classes)
        """
        # Stage 1: local temporal patterns per coordinate
        s1 = self.stage1(x)                          # (B, T, input_dim)

        # Stage 2: frame embeddings with residual from Stage 1
        s2 = self.stage2(s1, stage1_out=s1)          # (B, T, embedding_dim)

        # Stage 3: second local temporal CNN
        s3 = self.stage3(s2)                         # (B, T, embedding_dim)

        # Stage 4: global self-attention
        s4 = self.stage4(s3, src_key_padding_mask=padding_mask)  # (B, T, embedding_dim)

        # Aggregate: mean-pool over valid (non-padded) time steps
        if padding_mask is not None:
            # Invert mask: True = valid frame
            valid = (~padding_mask).float().unsqueeze(-1)        # (B, T, 1)
            pooled = (s4 * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        else:
            pooled = s4.mean(dim=1)                              # (B, embedding_dim)

        # Stage 5: classify
        logits = self.classifier(pooled)                         # (B, num_classes)
        return logits

    def get_features(self, x: torch.Tensor,
                     padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return the pooled embedding vector before the classifier head.
        Useful for extracting transferable representations."""
        s1 = self.stage1(x)
        s2 = self.stage2(s1, stage1_out=s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3, src_key_padding_mask=padding_mask)
        if padding_mask is not None:
            valid  = (~padding_mask).float().unsqueeze(-1)
            pooled = (s4 * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        else:
            pooled = s4.mean(dim=1)
        return pooled


# ---------------------------------------------------------------------------
# Transfer learning helpers
# ---------------------------------------------------------------------------

def build_model_for_pretraining(source_num_classes: int, **kwargs) -> SLRModel:
    """Create a fresh model for pre-training on a source dataset."""
    return SLRModel(num_classes=source_num_classes, **kwargs)


def adapt_for_finetuning(
    pretrained_model: SLRModel,
    target_num_classes: int,
    freeze_backbone: bool = False,
) -> SLRModel:
    """
    Replace the classifier head for a new target dataset.

    Args:
        pretrained_model  : model with weights loaded from pre-training
        target_num_classes: number of classes in the target (fine-tune) dataset
        freeze_backbone   : if True, freeze everything except the classifier head
                            (matches the 'without fine-tuning' condition in the paper)
    Returns:
        model ready for fine-tuning
    """
    embedding_dim = pretrained_model.classifier.in_features
    pretrained_model.classifier = nn.Linear(embedding_dim, target_num_classes)

    if freeze_backbone:
        for name, param in pretrained_model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False

    return pretrained_model


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def build_optimizer(model: SLRModel, lr: float = 3e-4) -> torch.optim.Optimizer:
    """Adam optimizer as specified in the paper (lr=0.0003)."""
    return torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )


def build_schedulers(optimizer: torch.optim.Optimizer):
    """
    Returns the two LR schedulers used in the paper:
      - ReduceLROnPlateau monitoring val accuracy (patience=5, factor=0.1)
    Early stopping logic should be handled in your training loop (patience=20
    on validation loss).
    """
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',       # monitor validation accuracy (higher = better)
        factor=0.1,
        patience=5
    )
    return plateau_scheduler


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    BATCH   = 4
    TIME    = 64     # sequence length (frames)
    KP      = 67     # MediaPipe Holistic keypoints (hands + upper body)
    COORDS  = 2      # x, y per keypoint
    CLASSES = 224    # ISL classes

    model = SLRModel(
        num_keypoints=KP,
        coords_per_kp=COORDS,
        num_classes=CLASSES,
        embedding_dim=192,
        num_attn_layers=4,
        num_attn_heads=8,
        dropout=0.1,
    )

    x    = torch.randn(BATCH, TIME, KP * COORDS)
    mask = torch.zeros(BATCH, TIME, dtype=torch.bool)  # no padding in this test

    logits = model(x, padding_mask=mask)
    print(f"Input  shape : {x.shape}")
    print(f"Output shape : {logits.shape}")   # expected: (4, 224)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params : {total_params:,}")

    # ---- Transfer learning demo ----
    # 1. Pre-train on VGT (292 classes)
    pretrain_model = build_model_for_pretraining(source_num_classes=292)
    print("\nPre-training model classifier:", pretrain_model.classifier)

    # 2. Adapt for ISL fine-tuning (224 classes), full fine-tuning
    finetune_model = adapt_for_finetuning(pretrain_model, target_num_classes=224, freeze_backbone=False)
    print("Fine-tune model classifier  :", finetune_model.classifier)

    # 3. Alternatively, frozen backbone (head-only)
    frozen_model = adapt_for_finetuning(
        build_model_for_pretraining(source_num_classes=292),
        target_num_classes=224,
        freeze_backbone=True,
    )
    trainable = sum(p.numel() for p in frozen_model.parameters() if p.requires_grad)
    print(f"Trainable params (frozen backbone): {trainable:,}")