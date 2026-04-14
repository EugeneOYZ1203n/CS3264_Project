import torch
import torch.nn as nn

# Used to extract additional features

class PoseFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Indices based on 67-point layout:
        # L: Sh(53), El(55), Wr(57) | R: Sh(54), El(56), Wr(58)
        # L_Hip(65), R_Hip(66)
        self.pairs = {
            'left_elbow_angle':  (53, 55, 57), # Shoulder-Elbow-Wrist
            'right_elbow_angle': (54, 56, 58),
            'left_shoulder_angle': (65, 53, 55), # Hip-Shoulder-Elbow
            'right_shoulder_angle':(66, 54, 56)
        }

    def forward(self, x):
        """
        x: (Batch, Time, 134) -> Reshaped to (B, T, 67, 2)
        Returns: (Batch, Time, 134 + num_angles)
        """
        B, T, D = x.shape
        coords = x.view(B, T, 67, 2)
        angles = []

        for name, (idx_a, idx_b, idx_c) in self.pairs.items():
            # Get vectors
            vec_ba = coords[:, :, idx_a] - coords[:, :, idx_b]
            vec_bc = coords[:, :, idx_c] - coords[:, :, idx_b]

            # Dot product and magnitudes
            dot = torch.sum(vec_ba * vec_bc, dim=-1)
            mag_ba = torch.norm(vec_ba, dim=-1) + 1e-6
            mag_bc = torch.norm(vec_bc, dim=-1) + 1e-6

            # Calculate cosine and angle
            cos_theta = dot / (mag_ba * mag_bc)
            # Clamp to avoid NaN in acos due to floating point precision
            angle = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))
            angles.append(angle.unsqueeze(-1))

        # Concatenate original features with new angles
        angles_tensor = torch.cat(angles, dim=-1) # (B, T, 4)
        
        return torch.cat([x, angles_tensor], dim=-1)