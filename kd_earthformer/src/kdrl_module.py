"""
CuboidKDRLModule — PyTorch Lightning module for KDRL Earthformer-to-Earthformer
knowledge distillation with inter-attention bridging.

Implements the Joint Training Framework (Section V, Eq. 10):
    L_joint = α·L_hard(student_pred, ground_truth)
            + β·L_soft(student_pred, teacher_pred)
            + γ·L_kt(teacher_feats, student_feats)

where L_kt uses the InterAttentionBridge for encoder + decoder features.
"""
import os
import sys
import warnings
from shutil import copyfile
from typing import List, Optional

# --- Ensure Earthformer root and kd_earthformer root are on path ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_KD_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))  # kd_earthformer/
_EF_ROOT = os.path.abspath(os.path.join(_KD_ROOT, ".."))   # Earthformer/
for _p in [_KD_ROOT, _EF_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import torchmetrics
import pytorch_lightning as pl
from omegaconf import OmegaConf

# Earthformer imports (Earthformer root is on sys.path)
from src.earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel
from src.earthformer.utils.optim import SequentialLR, warmup_lambda
from src.earthformer.utils.utils import get_parameter_names

# Local imports (kd_earthformer root is on sys.path)
from src.inter_attention_bridge import InterAttentionBridge

# Import teacher module loader
from scripts.cuboid_transformer.sst.train_cuboid_sst import CuboidSSTPLModule


def build_cuboid_from_oc(oc):
    """Build a CuboidTransformerModel from a merged OmegaConf config.

    Sets input/target spatial dims to 21×28 (SST patch size).
    """
    oc.model.input_shape[1] = 21
    oc.model.input_shape[2] = 28
    oc.model.target_shape[1] = 21
    oc.model.target_shape[2] = 28
    model_cfg = OmegaConf.to_object(oc.model)
    num_blocks = len(model_cfg["enc_depth"])
    enc_attn_patterns = [model_cfg["self_pattern"]] * num_blocks
    dec_self_attn_patterns = [model_cfg["cross_self_pattern"]] * num_blocks
    dec_cross_attn_patterns = [model_cfg["cross_pattern"]] * num_blocks
    return CuboidTransformerModel(
        input_shape=tuple(model_cfg["input_shape"]),
        target_shape=tuple(model_cfg["target_shape"]),
        base_units=model_cfg["base_units"],
        block_units=model_cfg["block_units"],
        scale_alpha=model_cfg["scale_alpha"],
        enc_depth=model_cfg["enc_depth"],
        dec_depth=model_cfg["dec_depth"],
        enc_use_inter_ffn=model_cfg["enc_use_inter_ffn"],
        dec_use_inter_ffn=model_cfg["dec_use_inter_ffn"],
        dec_hierarchical_pos_embed=model_cfg["dec_hierarchical_pos_embed"],
        downsample=model_cfg["downsample"],
        downsample_type=model_cfg["downsample_type"],
        enc_attn_patterns=enc_attn_patterns,
        dec_self_attn_patterns=dec_self_attn_patterns,
        dec_cross_attn_patterns=dec_cross_attn_patterns,
        num_heads=model_cfg["num_heads"],
        attn_drop=model_cfg["attn_drop"],
        proj_drop=model_cfg["proj_drop"],
        ffn_drop=model_cfg["ffn_drop"],
        upsample_type=model_cfg["upsample_type"],
        ffn_activation=model_cfg["ffn_activation"],
        gated_ffn=model_cfg["gated_ffn"],
        norm_layer=model_cfg["norm_layer"],
        padding_type=model_cfg["padding_type"],
        checkpoint_level=model_cfg["checkpoint_level"],
    )


def _get_encoder_mem_dims(model: CuboidTransformerModel, indices: List[int]) -> List[int]:
    """Extract channel dimensions for selected encoder block memories."""
    mem_shapes = model.encoder.get_mem_shapes()
    return [mem_shapes[i][-1] for i in indices]


def _get_decoder_block_dims(model: CuboidTransformerModel) -> List[int]:
    """Extract channel dimensions for each decoder block output.

    The decoder processes blocks deepest-first. After upsampling, the
    channel dim comes from the target mem_shape for that level.
    """
    mem_shapes = model.encoder.get_mem_shapes()
    # Decoder blocks process in reverse: block[-1] → block[0], with
    # upsampling between them. The output dims match the mem_shapes at
    # progressively lower resolutions.
    dims = []
    for i in range(len(mem_shapes) - 1, -1, -1):
        if i > 0:
            # After upsample, dim becomes mem_shapes[i-1][-1]
            dims.append(mem_shapes[i - 1][-1])
        else:
            dims.append(mem_shapes[0][-1])
    return dims


class CuboidKDRLModule(pl.LightningModule):
    """Lightning module for KDRL Earthformer-to-Earthformer distillation.

    Joint loss: L = α·L_hard + β·L_soft + γ·L_kt
    """

    def __init__(
        self,
        teacher_ckpt_path: str,
        oc_file: str,
        save_dir: str,
        # KDRL loss weights (Eq. 10)
        alpha_hard: float = 1.0,
        beta_soft: float = 0.5,
        gamma_kt: float = 0.5,
        # Bridge config
        enc_block_indices: Optional[List[int]] = None,
        use_decoder_bridge: bool = True,
        bridge_num_heads: int = 4,
        bridge_proj_dim: int = 64,
        bridge_attn_drop: float = 0.0,
        bridge_proj_drop: float = 0.0,
    ):
        super().__init__()
        if enc_block_indices is None:
            enc_block_indices = [-2, -1]

        self.alpha_hard = alpha_hard
        self.beta_soft = beta_soft
        self.gamma_kt = gamma_kt
        self.enc_block_indices = enc_block_indices
        self.use_decoder_bridge = use_decoder_bridge
        self.save_dir = save_dir

        # --- Load and freeze teacher ---
        teacher_pl = CuboidSSTPLModule.load_from_checkpoint(
            teacher_ckpt_path,
            oc_file=oc_file,
            save_dir=os.path.dirname(teacher_ckpt_path),
        )
        self.teacher = teacher_pl.torch_nn_module
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        # --- Build student (same architecture) ---
        oc = teacher_pl.oc
        self.student = build_cuboid_from_oc(oc)
        self.oc = oc

        # --- Build inter-attention bridge ---
        enc_dims = _get_encoder_mem_dims(self.student, enc_block_indices)

        dec_dims = None
        if use_decoder_bridge:
            dec_dims = _get_decoder_block_dims(self.student)

        self.bridge = InterAttentionBridge(
            enc_block_dims=enc_dims,
            dec_block_dims=dec_dims,
            proj_dim=bridge_proj_dim,
            num_heads=bridge_num_heads,
            attn_drop=bridge_attn_drop,
            proj_drop=bridge_proj_drop,
        )

        # --- Save config ---
        if not os.path.isabs(self.save_dir):
            _curr_dir = os.path.dirname(os.path.abspath(__file__))
            self.save_dir = os.path.join(_curr_dir, "..", "experiments", self.save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        if oc_file:
            cfg_target = os.path.join(self.save_dir, "cfg.yaml")
            if not os.path.exists(cfg_target) or not os.path.samefile(oc_file, cfg_target):
                try:
                    copyfile(oc_file, cfg_target)
                except Exception:
                    pass

        # --- Metrics ---
        self.valid_mse = torchmetrics.MeanSquaredError()
        self.valid_mae = torchmetrics.MeanAbsoluteError()
        self.valid_mape = torchmetrics.MeanAbsolutePercentageError()

    def _forward_with_features(self, x, model):
        """Forward through a model with feature extraction for KDRL.

        Parameters
        ----------
        x : Tensor
            Input (B, T, H, W, C) in Earthformer layout.
        model : CuboidTransformerModel
            The teacher or student model.

        Returns
        -------
        out : Tensor
            Prediction (B, T_out, H, W, C_out).
        enc_feats : list of Tensor
            Selected encoder block outputs.
        dec_feats : list of Tensor or None
            Decoder block outputs (if use_decoder_bridge).
        """
        out, features = model.forward_with_features(
            x,
            enc_block_indices=self.enc_block_indices,
            return_dec_pre_proj=True,
            return_dec_block_outputs=self.use_decoder_bridge,
        )
        enc_feats = features["enc_mem_l"]
        dec_feats = features.get("dec_block_outputs", None)
        return out, enc_feats, dec_feats

    def training_step(self, batch, batch_idx):
        """Joint training step: L = α·L_hard + β·L_soft + γ·L_kt"""
        teacher_x, teacher_y, student_x, student_y, _ = batch

        # Permute to Earthformer layout: (B, T, C, H, W) → (B, T, H, W, C)
        teacher_x = teacher_x.permute(0, 1, 3, 4, 2)
        student_x = student_x.permute(0, 1, 3, 4, 2)
        student_y = student_y.permute(0, 1, 3, 4, 2)

        # --- Teacher forward (frozen, no grad) ---
        with torch.no_grad():
            teacher_pred, teacher_enc_feats, teacher_dec_feats = \
                self._forward_with_features(teacher_x, self.teacher)

        # --- Student forward ---
        student_pred, student_enc_feats, student_dec_feats = \
            self._forward_with_features(student_x, self.student)

        # --- L_hard: student prediction vs ground truth ---
        loss_hard = F.mse_loss(student_pred, student_y)

        # --- L_soft: student prediction vs teacher prediction (soft labels) ---
        loss_soft = F.mse_loss(student_pred, teacher_pred)

        # --- L_kt: inter-attention bridge loss ---
        loss_kt = self.bridge(
            teacher_enc_feats=teacher_enc_feats,
            student_enc_feats=student_enc_feats,
            teacher_dec_feats=teacher_dec_feats,
            student_dec_feats=student_dec_feats,
        )

        # --- Joint loss (Eq. 10) ---
        loss = (
            self.alpha_hard * loss_hard
            + self.beta_soft * loss_soft
            + self.gamma_kt * loss_kt
        )

        # Log all components
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train_hard_loss", loss_hard, on_step=True, on_epoch=False)
        self.log("train_soft_loss", loss_soft, on_step=True, on_epoch=False)
        self.log("train_kt_loss", loss_kt, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validate student-only prediction against ground truth."""
        _, _, student_x, student_y, _ = batch
        student_x = student_x.permute(0, 1, 3, 4, 2)
        student_y = student_y.permute(0, 1, 3, 4, 2)

        pred = self.student(student_x)
        if self.trainer.precision == 16:
            pred = pred.float()
        self.valid_mse(pred, student_y)
        self.valid_mae(pred, student_y)

        mean = self.trainer.datamodule.mean
        std = self.trainer.datamodule.std

        pred_c = (pred * std) + mean
        target_c = (student_y * std) + mean
        self.valid_mape(pred_c + 273.15, target_c + 273.15)

    def on_validation_epoch_end(self):
        mse = self.valid_mse.compute()
        mae = self.valid_mae.compute()
        mape = self.valid_mape.compute()
        accuracy = (1.0 - mape) * 100
        self.log("valid_mse_epoch", self.valid_mse.compute(), prog_bar=True, on_epoch=True)
        self.log("valid_mae_epoch", self.valid_mae.compute(), prog_bar=True, on_epoch=True)
        self.log("valid_accuracy", accuracy, prog_bar=True, on_epoch=True)
        self.valid_mse.reset()
        self.valid_mae.reset()

    def configure_optimizers(self):
        """AdamW with warmup + cosine schedule."""
        oc = self.oc
        total_batch_size = oc.optim.total_batch_size
        max_epochs = oc.optim.max_epochs
        num_train = self.trainer.datamodule.train_dataset.length
        total_steps = max(1, (num_train // total_batch_size) * max_epochs)
        warmup_iter = int(np.round(oc.optim.warmup_percentage * total_steps))

        # Separate weight decay for LayerNorm params
        decay_parameters = get_parameter_names(self.student, [nn.LayerNorm])
        decay_parameters = [n for n in decay_parameters if "bias" not in n]

        # Also get bridge parameters
        bridge_params = list(self.bridge.named_parameters())
        bridge_decay = [n for n, _ in bridge_params if "bias" not in n]
        bridge_no_decay = [n for n, _ in bridge_params if "bias" in n]

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": [p for n, p in self.student.named_parameters()
                               if n in decay_parameters],
                    "weight_decay": oc.optim.wd,
                },
                {
                    "params": [p for n, p in self.student.named_parameters()
                               if n not in decay_parameters],
                    "weight_decay": 0.0,
                },
                {
                    "params": [p for n, p in self.bridge.named_parameters()
                               if n in bridge_decay],
                    "weight_decay": oc.optim.wd,
                },
                {
                    "params": [p for n, p in self.bridge.named_parameters()
                               if n not in bridge_decay],
                    "weight_decay": 0.0,
                },
            ],
            lr=oc.optim.lr,
            weight_decay=oc.optim.wd,
        )

        if warmup_iter > 0:
            warmup_scheduler = LambdaLR(
                optimizer,
                lr_lambda=warmup_lambda(
                    warmup_steps=warmup_iter,
                    min_lr_ratio=oc.optim.warmup_min_lr_ratio,
                ),
            )
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_steps - warmup_iter,
                eta_min=oc.optim.min_lr_ratio * oc.optim.lr,
            )
            lr_scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_iter],
            )
        else:
            lr_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=oc.optim.min_lr_ratio * oc.optim.lr,
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
