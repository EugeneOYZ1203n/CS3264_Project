import math

import torch
import torch.nn as nn

class SLRModel(nn.Module):
    def __init__(
        self, 
        input_dim=134, 
        embed_dim=192, 
        num_classes=250,
        n_heads=8, 
        n_attn_layers=4, 
        dropout=0.1
    ):
        super().__init__()

        # --- Stage 1 & 3: Temporal Convolutions ---
        self.stage1_blocks = nn.ModuleList([
            self._make_conv_block(input_dim, k, layers=1) for k in [3, 5, 7, 9]
        ])
        
        self.stage3_blocks = nn.ModuleList([
            self._make_conv_block(embed_dim, k, layers=2) for k in [3, 5, 7, 9]
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

    def _make_conv_block(self, dim, kernel, layers):
        convs = []
        for i in range(layers):
            convs.append(nn.Conv1d(dim, dim, kernel, padding=kernel//2, groups=dim))
            convs.append(nn.LayerNorm(dim))
            convs.append(nn.GELU())
        return nn.Sequential(*convs)

    def _make_embed_block(self, in_dim, out_dim, dropout, not_last):
        layers = [nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim)]
        if not_last:
            layers.extend([nn.GELU(), nn.Dropout(dropout)])
        return nn.Sequential(*layers)

    # --- Stage Functions ---

    def run_stage1(self, x):
        # x: (B, T, D) -> Conv1d needs (B, D, T)
        # Conv layer filters across the last dimension and we want it to be time T
        x = x.transpose(1, 2)
        for block in self.stage1_blocks:
            x = x + block(x) # residuals
        return x.transpose(1, 2)

    def run_stage2(self, x):
        residual = self.stage2_residual_proj(x)
        out = x
        for block in self.stage2_blocks:
            out = block(out)
        return out + residual # residual projection across all of Stage 2

    def run_stage3(self, x):
        x = x.transpose(1, 2)
        for block in self.stage3_blocks:
            x = x + block(x) # residuals
        return x.transpose(1, 2)

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

    def forward(self, x, padding_mask=None):
        # Stage 1: Temporal patterns per feature
        s1 = self.run_stage1(x)
        
        # Stage 2: Spatial frame embedding
        s2 = self.run_stage2(s1, s1)
        
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