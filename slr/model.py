import math

import torch
import torch.nn as nn

from PoseFeatureExtractor import PoseFeatureExtractor

class SLRModel(nn.Module):
    def __init__(
        self, 
        input_dim=134, 
        embed_dim=192, 
        num_classes=250,
        n_heads=8, 
        n_attn_layers=4, 
        dropout=0.1,
        stochastic_drop_start_prob = 1.0,
        stochastic_drop_end_prob = 1.0
    ):
        super().__init__()

        total_residual_layers = 4 + 4

        probs = torch.linspace(
            stochastic_drop_start_prob, 
            stochastic_drop_end_prob, 
            total_residual_layers
        ).tolist()

        self.extractor = PoseFeatureExtractor()
        input_dim += 4

        # --- Stage 1 & 3: Temporal Convolutions ---
        s1_probs = probs[:4]
        self.stage1_blocks = nn.ModuleList([
            self._make_conv_block(input_dim, k, layers=1, keep_prob=s1_probs[i]) for i, k in enumerate([3, 5, 7, 9])
        ])
        
        s3_probs = probs[4:8]
        self.stage3_blocks = nn.ModuleList([
            self._make_conv_block(embed_dim, k, layers=2, keep_prob=s3_probs[i]) for i, k in enumerate([3, 5, 7, 9])
        ])

        # --- Stage 2: Frame Embedder ---
        self.stage2_blocks = nn.ModuleList()
        for i in range(4):
            in_f = input_dim if i == 0 else embed_dim # Immediate projection
            not_last = (i < 3)
            self.stage2_blocks.append(self._make_embed_block(in_f, embed_dim, dropout, not_last))
        
        self.stage2_residual_proj = nn.Linear(input_dim, embed_dim)

        # --- Stage 4: Transformer with CLS Token ---
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim)) # Random starting CLS
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=n_heads, 
            dropout=dropout,
            batch_first=True, 
            dim_feedforward=embed_dim * 4, # Common practice to increase this by 4x
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_attn_layers)
        # Also handles the residuals between self attention (included in transformer encoder) and feed forward

        # --- Stage 5: Classifier ---
        self.classifier = nn.Linear(embed_dim, num_classes)

    # --- Building Blocks ---

    class TransposedConvLayer(nn.Module):
        """Internal helper to bridge Conv1d (channel-first) and LayerNorm (channel-last)"""
        def __init__(self, dim, kernel):
            super().__init__()
            self.conv = nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=dim)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()

        def forward(self, x):
            # x is (B, T, C)
            # Conv layer filters across the last dimension and we want it to be time T
            x = self.conv(x.transpose(1, 2)).transpose(1, 2)
            return self.act(self.norm(x))
    
    # Based on this repo https://github.com/FrancescoSaverioZuppichini/DropPath
    class DropPath(nn.Module):
        def __init__(self, p: float = 0.5, inplace: bool = False):
            super().__init__()
            self.p = p
            self.inplace = inplace

        def drop_path(self, x, keep_prob: float = 1.0, inplace: bool = False):
            mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1) 
            # remember tuples have the * operator -> (1,) * 3 = (1,1,1)
            mask = x.new_empty(mask_shape).bernoulli_(keep_prob)
            mask.div_(keep_prob)
            if inplace:
                x.mul_(mask)
            else:
                x = x * mask
            return x

        def forward(self, x):
            if self.training and self.p > 0:
                x = self.drop_path(x, self.p, self.inplace)
            return x

    def _make_conv_block(self, dim, kernel, layers, keep_prob = 1.0):
        """
        Returns a block containing 'layers' number of depthwise convs.
        Each matches the DepthwiseConvBlock behavior: (B, T, C) -> (B, T, C)
        """
        blocks = []
        for _ in range(layers):
            blocks.append(self.TransposedConvLayer(dim, kernel))
        if keep_prob != 1.0:
            blocks.append(self.DropPath(p = keep_prob))
        return nn.Sequential(*blocks)

    def _make_embed_block(self, in_dim, out_dim, dropout, not_last):
        layers = [nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim)]
        if not_last:
            layers.extend([nn.GELU(), nn.Dropout(dropout)])
        return nn.Sequential(*layers)

    # --- Stage Functions ---

    def run_stage1(self, x):
        # x: (B, T, D) -> Conv1d needs (B, D, T)
        for block in self.stage1_blocks:
            x = x + block(x) # residuals
        return x

    def run_stage2(self, x):
        residual = self.stage2_residual_proj(x)
        out = x
        for block in self.stage2_blocks:
            out = block(out)
        return out + residual # residual projection across all of Stage 2

    def run_stage3(self, x):
        for block in self.stage3_blocks:
            x = x + block(x) # residuals
        return x

    # --- Positional Encoding ---
    # Taken from https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py
    @staticmethod
    def positionalencoding1d(d_model, length, device):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :param device: the device (cuda/cpu) to place the tensor on
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model, device=device)
        position = torch.arange(0, length, device=device).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float, device=device) *
                             -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe
    
    def stage_params(self):
        """
        Returns an ordered dict mapping stage name → list of parameters.
        Stage numbering matches the architecture description (1–5).
        """
        return {
            "stage1": list(self.stage1_blocks.parameters()),
            "stage2": list(self.stage2_blocks.parameters())
                    + list(self.stage2_residual_proj.parameters()),
            "stage3": list(self.stage3_blocks.parameters()),
            "stage4": list(self.transformer.parameters())
                    + [self.cls_token],
            "stage5": list(self.classifier.parameters()),
        }

    def forward(self, x, padding_mask=None):
        x = self.extractor(x)

        # Stage 1: Temporal patterns per feature
        s1 = self.run_stage1(x)
        
        # Stage 2: Spatial frame embedding
        s2 = self.run_stage2(s1)
        
        # Stage 3: Local motion/position refinement
        s3 = self.run_stage3(s2)

        # Stage 4: Global Attention (CLS Token Logic)
        B, T, E = s3.shape
        pe = self.positionalencoding1d(E, T, x.device)
        s3 = s3 + pe

        cls_tokens = self.cls_token.expand(B, -1, -1) # Expand CLS token to fit all batches
        s4_input = torch.cat((cls_tokens, s3), dim=1) # Add CLS to front of stage 3 output

        # Handle mask for the extra token
        if padding_mask is not None:
            cls_mask = torch.zeros((B, 1), device=x.device, dtype=torch.bool)
            padding_mask = torch.cat((cls_mask, padding_mask), dim=1)

        s4_out = self.transformer(s4_input, src_key_padding_mask=padding_mask)
        
        # Extract CLS token output
        cls_final = s4_out[:, 0]

        # Stage 5: Classification
        return self.classifier(cls_final)
    
# Stage ordering — freeze_up_to=N freezes stages 1 … N
_STAGE_ORDER = ["stage1", "stage2", "stage3", "stage4", "stage5"]
 
 
def load_backbone(model: SLRModel, checkpoint_path: str, device="cpu"):
    """
    Load all weights from a checkpoint except the classifier head.
    The new model can have a different num_classes.
    """
    ckpt       = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model_state_dict"]
    backbone   = {k: v for k, v in state_dict.items() if "classifier" not in k}
    missing, unexpected = model.load_state_dict(backbone, strict=False)
    print(f"  Backbone loaded from '{checkpoint_path}'")
    if missing:
        print(f"    Missing keys  : {missing}")
    if unexpected:
        print(f"    Unexpected    : {unexpected}")
    print(f"  Classifier head ({model.classifier.out_features} classes) initialised fresh.")
 
 
def freeze_stages(model: SLRModel, freeze_up_to: int):
    """
    Freeze stages 1 … freeze_up_to (inclusive), leave the rest trainable.
 
    freeze_up_to=0 → nothing frozen (full fine-tune)
    freeze_up_to=1 → stage 1 frozen
    freeze_up_to=2 → stages 1-2 frozen
    freeze_up_to=3 → stages 1-3 frozen
    freeze_up_to=4 → stages 1-4 frozen  (only classifier trains)
 
    Args:
        model        : SLRModel instance
        freeze_up_to : int 0-4
    """
    if not (0 <= freeze_up_to <= 4):
        raise ValueError(f"freeze_up_to must be 0-4, got {freeze_up_to}")
 
    groups = model.stage_params()
 
    # First unfreeze everything so this is idempotent
    for p in model.parameters():
        p.requires_grad = True
 
    frozen_stages = _STAGE_ORDER[:freeze_up_to]
    for stage_name in frozen_stages:
        for p in groups[stage_name]:
            p.requires_grad = False
 
    # Summary
    frozen_count   = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_label   = f"stages 1-{freeze_up_to}" if freeze_up_to > 0 else "nothing"
    print(f"  Frozen  : {frozen_label}  ({frozen_count:,} params)")
    print(f"  Trainable: {trainable_count:,} params")