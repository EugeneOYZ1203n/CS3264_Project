import torch
import torch.nn as nn
import torch.nn.functional as F

class PosePreprocessing(nn.Module):
    def __init__(self, max_len=128):
        super().__init__()
        self.max_len = max_len

    def forward(self, x):
        B, T, F = x.shape

        if self.max_len is not None and T > self.max_len:
            x = x[:, :self.max_len, :]
            T = self.max_len

        dx = torch.zeros_like(x)
        if T > 1:
            # Result at index t is the movement toward t+1
            dx[:, :-1, :] = x[:, 1:, :] - x[:, :-1, :]

        dx2 = torch.zeros_like(x)
        if T > 2:
            dx2[:, :-2, :] = x[:, 2:, :] - x[:, :-2, :]

        out = torch.cat([x, dx, dx2], dim=-1)

        out = torch.where(torch.isnan(out), torch.tensor(0.0, device=x.device), out)

        return out

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

class Conv1DBlock(nn.Module):
    def __init__(self, dim, kernel_size, drop_rate=0.2, drop_path_rate=0.2):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.BatchNorm1d(dim, momentum=0.05)
        self.dropout = nn.Dropout(drop_rate)
        self.act = nn.GELU()
        self.drop_path = DropPath(p=1 - drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x):
        # x: (B, C, T)
        residual = x
        out = self.conv(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.dropout(out)
        return self.drop_path(out) + residual # Keras Conv1DBlock usually implies a residual

class TransformerBlock(nn.Module):
    def __init__(self, dim, expand=2, nhead=8):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=nhead, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * expand),
            nn.GELU(),
            nn.Linear(dim * expand, dim)
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, padding_mask=None):
        # x: (B, T, C)
        x = self.norm(x)
        attn_out, _ = self.mha(x, x, x, key_padding_mask=padding_mask)
        x = x + self.dropout(attn_out)
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        return x

class KerasStyleSLRModel(nn.Module):
    def __init__(self, input_dim, num_classes, dim=192, ksize=17):
        super().__init__()
        self.preprocess = PosePreprocessing(max_len=128)
        input_dim *= 3
        self.stem_conv = nn.Linear(input_dim, dim, bias=False)
        self.stem_bn = nn.BatchNorm1d(dim, momentum=0.05)
        
        # Interleaved Sections
        self.block1 = nn.ModuleList([
            Conv1DBlock(dim, ksize),
            Conv1DBlock(dim, ksize),
            Conv1DBlock(dim, ksize)
        ])
        self.trans1 = TransformerBlock(dim)

        self.block2 = nn.ModuleList([
            Conv1DBlock(dim, ksize),
            Conv1DBlock(dim, ksize),
            Conv1DBlock(dim, ksize)
        ])
        self.trans2 = TransformerBlock(dim)

        self.top_conv = nn.Linear(dim, dim * 2)
        self.late_dropout = nn.Dropout(0.8)
        self.classifier = nn.Linear(dim * 2, num_classes)

    def forward(self, x, padding_mask=None):
        # x: (B, T, C_in)
        x = self.preprocess(x)
        x = self.stem_conv(x) 
        
        # Switch to Channel-First for BN and Conv1D
        x = x.transpose(1, 2) # (B, C, T)
        x = self.stem_bn(x)
        
        # First group
        for conv in self.block1:
            x = conv(x)
        
        # Switch to Channel-Last for Transformer
        x = x.transpose(1, 2) # (B, T, C)
        x = self.trans1(x, padding_mask=padding_mask)
        
        # Second group (Switch back to C-First)
        x = x.transpose(1, 2)
        for conv in self.block2:
            x = conv(x)
            
        x = x.transpose(1, 2)
        x = self.trans2(x, padding_mask=padding_mask)
        
        # Global Average Pooling
        x = self.top_conv(x)
        if padding_mask is not None:
            valid = (~padding_mask).float().unsqueeze(-1)          # (B, T, 1)
            x = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)
        
        x = self.late_dropout(x)
        return self.classifier(x)
    
def freeze_stages(model, stage_idx):
    """
    Freezes stages of the model for transfer learning.
    """
    # Define stage mappings
    stages = [
        [model.preprocess, model.stem_conv, model.stem_bn],
        [model.block1, model.trans1],
        [model.block2, model.trans2],
        [model.top_conv],
    ]
    
    for i in range(min(stage_idx, len(stages))):
        for module in stages[i]:
            for param in module.parameters():
                param.requires_grad = False
    print(f"  Froze up to stage {stage_idx}")

def load_backbone(model, ckpt_path, device):
    """
    Loads weights from a checkpoint, skipping the classifier head 
    if classes don't match (e.g., ASL -> SgSL).
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    # Check if we need to extract from 'model_state_dict' or if it's raw state_dict
    state_dict = ckpt.get("model_state_dict", ckpt)
    
    model_dict = model.state_dict()
    # Filter out the classifier layer
    pretrained_dict = {k: v for k, v in state_dict.items() 
                       if k in model_dict and "classifier" not in k}
    
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print(f"  Loaded {len(pretrained_dict)} layers from backbone.")