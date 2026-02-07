"""
Earthformer-to-Earthformer distillation for SST.

Teacher is fixed on one patch; student is trained on striding patches with
feature distillation (encoder + decoder features) and optional output loss.
Offline: teacher is loaded from checkpoint and frozen.
"""
import os
import warnings
from shutil import copyfile
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from omegaconf import OmegaConf
import argparse

from src.earthformer.datasets.sst.sst_patch_datamodule import SSTPatchDataModule
from src.earthformer.utils.optim import SequentialLR, warmup_lambda
from src.earthformer.utils.utils import get_parameter_names
from src.earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel

# Import teacher module to load checkpoint
from train_cuboid_sst import CuboidSSTPLModule

_curr_dir = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
exps_dir = os.path.join(_curr_dir, "experiments")


def build_cuboid_from_oc(oc):
    """Build CuboidTransformerModel from merged config (same as CuboidSSTPLModule)."""
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


class CuboidDistillPLModule(pl.LightningModule):
    """Lightning module for Earthformer teacher-student feature distillation."""

    def __init__(
        self,
        teacher_ckpt_path: str,
        oc_file: str,
        save_dir: str,
        enc_block_indices=None,
        loss_weight_enc: float = 1.0,
        loss_weight_dec: float = 1.0,
        loss_weight_output: float = 0.5,
    ):
        super().__init__()
        if enc_block_indices is None:
            enc_block_indices = [-2, -1]
        self.enc_block_indices = enc_block_indices
        self.loss_weight_enc = loss_weight_enc
        self.loss_weight_dec = loss_weight_dec
        self.loss_weight_output = loss_weight_output
        self.save_dir = save_dir
        teacher_pl = CuboidSSTPLModule.load_from_checkpoint(
            teacher_ckpt_path,
            oc_file=oc_file,
            save_dir=os.path.dirname(teacher_ckpt_path),
        )
        self.teacher = teacher_pl.torch_nn_module
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()
        oc = teacher_pl.oc
        self.student = build_cuboid_from_oc(oc)
        self.oc = oc

        if not os.path.isabs(self.save_dir):
            self.save_dir = os.path.join(exps_dir, self.save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        cfg_target = os.path.join(self.save_dir, "cfg.yaml")
        if oc_file and (not os.path.exists(cfg_target) or not os.path.samefile(oc_file, cfg_target)):
            copyfile(oc_file, cfg_target)

        self.valid_mse = torchmetrics.MeanSquaredError()
        self.valid_mae = torchmetrics.MeanAbsoluteError()

    def forward(self, x, model, with_features=True):
        if with_features:
            out, feats = model.forward_with_features(
                x,
                enc_block_indices=self.enc_block_indices,
                return_dec_pre_proj=True,
            )
            return out, feats
        return model(x), None

    def feature_loss(self, feats_teacher, feats_student):
        loss_enc = 0.0
        for te, st in zip(feats_teacher["enc_mem_l"], feats_student["enc_mem_l"]):
            if te.shape != st.shape:
                st = F.interpolate(
                    st.permute(0, 4, 1, 2, 3),
                    size=(te.shape[1], te.shape[2], te.shape[3]),
                    mode="trilinear",
                    align_corners=False,
                ).permute(0, 2, 3, 4, 1)
            loss_enc = loss_enc + F.mse_loss(st, te)
        loss_dec = 0.0
        if feats_teacher.get("dec_pre_proj") is not None and feats_student.get("dec_pre_proj") is not None:
            td = feats_teacher["dec_pre_proj"]
            sd = feats_student["dec_pre_proj"]
            if td.shape != sd.shape:
                sd = F.interpolate(
                    sd.permute(0, 4, 1, 2, 3),
                    size=(td.shape[1], td.shape[2], td.shape[3]),
                    mode="trilinear",
                    align_corners=False,
                ).permute(0, 2, 3, 4, 1)
            loss_dec = F.mse_loss(sd, td)
        return loss_enc, loss_dec

    def training_step(self, batch, batch_idx):
        teacher_x, teacher_y, student_x, student_y, _ = batch
        teacher_x = teacher_x.permute(0, 1, 3, 4, 2)
        student_x = student_x.permute(0, 1, 3, 4, 2)
        student_y = student_y.permute(0, 1, 3, 4, 2)

        with torch.no_grad():
            _, feats_teacher = self.forward(teacher_x, self.teacher, with_features=True)

        student_out, feats_student = self.forward(student_x, self.student, with_features=True)

        loss_enc, loss_dec = self.feature_loss(feats_teacher, feats_student)
        loss_out = F.mse_loss(student_out, student_y)
        loss = (
            self.loss_weight_enc * loss_enc
            + self.loss_weight_dec * loss_dec
            + self.loss_weight_output * loss_out
        )

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train_feat_enc", loss_enc, on_step=True, on_epoch=False)
        self.log("train_feat_dec", loss_dec, on_step=True, on_epoch=False)
        self.log("train_out_loss", loss_out, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        _, _, student_x, student_y, _ = batch
        student_x = student_x.permute(0, 1, 3, 4, 2)
        student_y = student_y.permute(0, 1, 3, 4, 2)
        pred = self.student(student_x)
        if self.trainer.precision == "16-mixed":
            pred = pred.float()
        self.valid_mse(pred, student_y)
        self.valid_mae(pred, student_y)

    def on_validation_epoch_end(self):
        self.log("valid_mse_epoch", self.valid_mse.compute(), prog_bar=True, on_epoch=True)
        self.log("valid_mae_epoch", self.valid_mae.compute(), prog_bar=True, on_epoch=True)
        self.valid_mse.reset()
        self.valid_mae.reset()

    def configure_optimizers(self):
        oc = self.oc
        total_batch_size = oc.optim.total_batch_size
        max_epochs = oc.optim.max_epochs
        num_train = self.trainer.datamodule.train_dataset.length
        total_steps = max(1, (num_train // total_batch_size) * max_epochs)
        warmup_iter = int(np.round(oc.optim.warmup_percentage * total_steps))
        decay_parameters = get_parameter_names(self.student, [nn.LayerNorm])
        decay_parameters = [n for n in decay_parameters if "bias" not in n]
        optimizer = torch.optim.AdamW(
            [
                {"params": [p for n, p in self.student.named_parameters() if n in decay_parameters], "weight_decay": oc.optim.wd},
                {"params": [p for n, p in self.student.named_parameters() if n not in decay_parameters], "weight_decay": 0.0},
            ],
            lr=oc.optim.lr,
            weight_decay=oc.optim.wd,
        )
        if warmup_iter > 0:
            warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda(warmup_steps=warmup_iter, min_lr_ratio=oc.optim.warmup_min_lr_ratio))
            cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_iter, eta_min=oc.optim.min_lr_ratio * oc.optim.lr)
            lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_iter])
        else:
            lr_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=oc.optim.min_lr_ratio * oc.optim.lr)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}}


def get_parser():
    p = argparse.ArgumentParser(description="Earthformer-Earthformer SST distillation")
    p.add_argument("--cfg", type=str, required=True, help="Path to distillation YAML config.")
    p.add_argument("--save", type=str, default="sst_distill_run", help="Experiment save directory.")
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--ckpt_name", type=str, default=None, help="Resume from this student checkpoint name.")
    return p


def main():
    args = get_parser().parse_args()
    oc = OmegaConf.load(open(args.cfg, "r"))
    seed = oc.get("seed", 2022)
    seed_everything(seed, workers=True)

    cfg_dir = os.path.dirname(os.path.abspath(args.cfg))
    teacher_cfg_path = oc.teacher_cfg_path
    if not os.path.isabs(teacher_cfg_path):
        teacher_cfg_path = os.path.join(cfg_dir, teacher_cfg_path)
    if not os.path.exists(teacher_cfg_path):
        raise FileNotFoundError(f"Teacher config not found: {teacher_cfg_path}")

    dataset_cfg = OmegaConf.to_object(oc.dataset)
    dataset_cfg.pop("_target_", None)
    dm = SSTPatchDataModule(**dataset_cfg)
    dm.setup("fit")

    teacher_ckpt = oc.teacher_ckpt_path
    if not os.path.isabs(teacher_ckpt):
        teacher_ckpt = os.path.join(exps_dir, teacher_ckpt)
    if not os.path.exists(teacher_ckpt):
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_ckpt}")

    distill_cfg = oc.get("distill", OmegaConf.create())
    pl_module = CuboidDistillPLModule(
        teacher_ckpt_path=teacher_ckpt,
        oc_file=teacher_cfg_path,
        save_dir=args.save,
        enc_block_indices=list(distill_cfg.get("enc_block_indices", [-2, -1])),
        loss_weight_enc=float(distill_cfg.get("loss_weight_enc", 1.0)),
        loss_weight_dec=float(distill_cfg.get("loss_weight_dec", 1.0)),
        loss_weight_output=float(distill_cfg.get("loss_weight_output", 0.5)),
    )

    optim_cfg = OmegaConf.to_object(oc.optim)
    micro_batch = optim_cfg.get("micro_batch_size", dataset_cfg["batch_size"])
    total_batch = optim_cfg["total_batch_size"]
    num_gpus = max(1, args.gpus)
    accumulate = max(1, total_batch // (micro_batch * num_gpus))

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_mse_epoch",
        dirpath=os.path.join(pl_module.save_dir, "checkpoints"),
        filename="student-{epoch:03d}",
        save_top_k=oc.optim.get("save_top_k", 1),
        save_last=True,
        mode="min",
    )
    callbacks = [checkpoint_callback]
    if oc.logging.get("monitor_lr", True):
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    trainer = Trainer(
        devices=args.gpus,
        accelerator="gpu",
        max_epochs=optim_cfg["max_epochs"],
        check_val_every_n_epoch=oc.trainer.get("check_val_every_n_epoch", 1),
        gradient_clip_val=optim_cfg.get("gradient_clip_val", 1.0),
        precision=oc.trainer.get("precision", "16-mixed"),
        accumulate_grad_batches=accumulate,
        default_root_dir=pl_module.save_dir,
        callbacks=callbacks,
        log_every_n_steps=max(1, int(0.01 * dm.train_dataset.length / micro_batch)),
    )

    ckpt_path = None
    if args.ckpt_name:
        ckpt_path = os.path.join(pl_module.save_dir, "checkpoints", args.ckpt_name)
        if not os.path.exists(ckpt_path):
            warnings.warn(f"Checkpoint {ckpt_path} not found; training from scratch.")
            ckpt_path = None

    trainer.fit(model=pl_module, datamodule=dm, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
