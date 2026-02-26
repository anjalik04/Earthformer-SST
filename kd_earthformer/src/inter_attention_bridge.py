"""
Inter-Attention Bridge for KDRL (Knowledge Distillation-based Representation Learning).

Implements cross-attention modules that bridge teacher and student Earthformer
internal representations at selected encoder and decoder block outputs.

The bridge computes L_kt (Eq. 7 in the KDRL paper):
    L_kt = || u_t(x_t; Θ_t) − u_s(x_s; Θ_s) ||

where u_t and u_s are the teacher/student transform functions at selected layers,
and the inter-attention mechanism aligns the representations before computing
the distance.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class InterAttentionLayer(nn.Module):
    """Single cross-attention bridge between one teacher and one student feature map.

    Given teacher feature t and student feature s, both of shape (B, T, H, W, C):
      - Project s → Q, t → K, V
      - Compute multi-head cross-attention: attn_out = Attn(Q_s, K_t, V_t)
      - L_kt_block = MSE(attn_out, proj_s)

    This encourages the student's representation to become linearly
    predictable from the teacher's representation via attention.
    """

    def __init__(
        self,
        feat_dim: int,
        proj_dim: int = 64,
        num_heads: int = 4,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        """
        Parameters
        ----------
        feat_dim : int
            Channel dimension of the feature maps (C in (B, T, H, W, C)).
        proj_dim : int
            Projection dimension for Q, K, V.
        num_heads : int
            Number of attention heads.
        attn_drop : float
            Dropout on attention weights.
        proj_drop : float
            Dropout on output projection.
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.proj_dim = proj_dim

        # Projections: student → Q, teacher → K, V
        self.proj_q = nn.Linear(feat_dim, proj_dim)
        self.proj_k = nn.Linear(feat_dim, proj_dim)
        self.proj_v = nn.Linear(feat_dim, proj_dim)

        # Multi-head attention (Q from student, K/V from teacher)
        self.mha = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True,
        )

        # Output projection back to proj_dim for distance computation
        self.out_proj = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.Dropout(proj_drop),
        )

        # Student reference projection (for computing distance against attn output)
        self.student_ref_proj = nn.Linear(feat_dim, proj_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        for m in [self.proj_q, self.proj_k, self.proj_v, self.student_ref_proj]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        for m in self.out_proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        teacher_feat: torch.Tensor,
        student_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Compute inter-attention loss for one block pair.

        Parameters
        ----------
        teacher_feat : Tensor
            Shape (B, T, H, W, C_teacher)
        student_feat : Tensor
            Shape (B, T', H', W', C_student)

        Returns
        -------
        loss : Tensor
            Scalar MSE loss between attention-aligned representation and student ref.
        """
        B = teacher_feat.shape[0]

        # Align spatial dims if needed (interpolate teacher to match student)
        if teacher_feat.shape[1:4] != student_feat.shape[1:4]:
            # (B, T, H, W, C) → (B, C, T, H, W) for interpolation
            t_perm = teacher_feat.permute(0, 4, 1, 2, 3)
            target_size = student_feat.shape[1:4]  # (T', H', W')
            t_perm = F.interpolate(
                t_perm,
                size=target_size,
                mode="trilinear",
                align_corners=False,
            )
            teacher_feat = t_perm.permute(0, 2, 3, 4, 1)  # back to (B, T', H', W', C)

        T, H, W = student_feat.shape[1], student_feat.shape[2], student_feat.shape[3]
        seq_len = T * H * W

        # Flatten spatial dims: (B, T, H, W, C) → (B, T*H*W, C)
        t_flat = teacher_feat.reshape(B, seq_len, -1)
        s_flat = student_feat.reshape(B, seq_len, -1)

        # Project
        Q = self.proj_q(s_flat)    # (B, seq_len, proj_dim)
        K = self.proj_k(t_flat)    # (B, seq_len, proj_dim)
        V = self.proj_v(t_flat)    # (B, seq_len, proj_dim)

        # Cross-attention: student queries, teacher keys/values
        attn_out, _ = self.mha(Q, K, V)  # (B, seq_len, proj_dim)
        attn_out = self.out_proj(attn_out)

        # Student reference (what the student representation should align to)
        s_ref = self.student_ref_proj(s_flat)  # (B, seq_len, proj_dim)

        # L_kt for this block: L2 distance (Eq. 7)
        loss = F.mse_loss(attn_out, s_ref)
        return loss


class InterAttentionBridge(nn.Module):
    """Multi-block inter-attention bridge between teacher and student Earthformers.

    Wraps multiple InterAttentionLayer modules — one per selected encoder block
    and optionally one for the decoder.

    The total bridge loss is the mean of all per-block losses:
        L_kt = mean(L_kt_enc_block_0, L_kt_enc_block_1, ..., L_kt_dec)
    """

    def __init__(
        self,
        enc_block_dims: List[int],
        dec_block_dims: Optional[List[int]] = None,
        proj_dim: int = 64,
        num_heads: int = 4,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        """
        Parameters
        ----------
        enc_block_dims : list of int
            Channel dimensions for each selected encoder block feature.
        dec_block_dims : list of int, optional
            Channel dimensions for each decoder block feature to bridge.
            If None, no decoder bridging is done.
        proj_dim : int
            Projection dimension for cross-attention.
        num_heads : int
            Number of attention heads in each bridge layer.
        attn_drop : float
            Attention dropout.
        proj_drop : float
            Output projection dropout.
        """
        super().__init__()
        self.num_enc_bridges = len(enc_block_dims)

        self.enc_bridges = nn.ModuleList([
            InterAttentionLayer(
                feat_dim=dim,
                proj_dim=proj_dim,
                num_heads=num_heads,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
            )
            for dim in enc_block_dims
        ])

        self.dec_bridges = nn.ModuleList()
        if dec_block_dims is not None:
            for dim in dec_block_dims:
                self.dec_bridges.append(
                    InterAttentionLayer(
                        feat_dim=dim,
                        proj_dim=proj_dim,
                        num_heads=num_heads,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                    )
                )

    def forward(
        self,
        teacher_enc_feats: List[torch.Tensor],
        student_enc_feats: List[torch.Tensor],
        teacher_dec_feats: Optional[List[torch.Tensor]] = None,
        student_dec_feats: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Compute total inter-attention bridge loss.

        Parameters
        ----------
        teacher_enc_feats : list of Tensor
            Teacher encoder block outputs, matching enc_block_dims.
        student_enc_feats : list of Tensor
            Student encoder block outputs, matching enc_block_dims.
        teacher_dec_feats : list of Tensor, optional
            Teacher decoder block outputs for decoder bridging.
        student_dec_feats : list of Tensor, optional
            Student decoder block outputs for decoder bridging.

        Returns
        -------
        loss : Tensor
            Scalar — mean of all per-block inter-attention losses.
        """
        losses = []

        # Encoder bridges
        assert len(teacher_enc_feats) == len(student_enc_feats) == self.num_enc_bridges, (
            f"Expected {self.num_enc_bridges} encoder features, "
            f"got teacher={len(teacher_enc_feats)}, student={len(student_enc_feats)}"
        )
        for bridge, t_feat, s_feat in zip(
            self.enc_bridges, teacher_enc_feats, student_enc_feats
        ):
            losses.append(bridge(t_feat, s_feat))

        # Decoder bridges
        if (
            teacher_dec_feats is not None
            and student_dec_feats is not None
            and len(self.dec_bridges) > 0
        ):
            n_dec = min(len(self.dec_bridges), len(teacher_dec_feats), len(student_dec_feats))
            for i in range(n_dec):
                losses.append(self.dec_bridges[i](teacher_dec_feats[i], student_dec_feats[i]))

        if len(losses) == 0:
            return torch.tensor(0.0, requires_grad=True)

        return torch.stack(losses).mean()
