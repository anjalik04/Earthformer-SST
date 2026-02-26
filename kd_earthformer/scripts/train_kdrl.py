"""
Entry-point script for KDRL training (Earthformer-to-Earthformer with
inter-attention knowledge distillation).

Usage:
    cd c:\\Users\\tmoitra\\Work\\sst_rnd\\Earthformer
    python kd_earthformer/scripts/train_kdrl.py --cfg kd_earthformer/configs/sst_kdrl.yaml --save kdrl_run_1

Or:
    cd c:\\Users\\tmoitra\\Work\\sst_rnd\\Earthformer\\kd_earthformer
    python scripts/train_kdrl.py --cfg configs/sst_kdrl.yaml --save kdrl_run_1

NOTE: The Earthformer repo root is auto-added to sys.path.
"""
import os
import sys
import warnings
import argparse

import numpy as np
import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# --- Ensure Earthformer root and kd_earthformer root are on path ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_KD_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_EF_ROOT = os.path.abspath(os.path.join(_KD_ROOT, ".."))  # Earthformer/
for p in [_KD_ROOT, _EF_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.earthformer.datasets.sst.sst_patch_datamodule import SSTPatchDataModule
from src.kdrl_module import CuboidKDRLModule


def get_parser():
    p = argparse.ArgumentParser(
        description="KDRL: Earthformer-to-Earthformer Knowledge Distillation"
    )
    p.add_argument(
        "--cfg", type=str, required=True,
        help="Path to the KDRL YAML config (e.g., configs/sst_kdrl.yaml)."
    )
    p.add_argument(
        "--save", type=str, default="kdrl_run",
        help="Experiment save directory name."
    )
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument(
        "--ckpt_name", type=str, default=None,
        help="Resume from this student checkpoint filename."
    )
    return p


def main():
    args = get_parser().parse_args()
    oc = OmegaConf.load(open(args.cfg, "r"))
    seed = oc.get("seed", 2022)
    seed_everything(seed, workers=True)

    # --- Resolve teacher config path (relative to cfg file location) ---
    cfg_dir = os.path.dirname(os.path.abspath(args.cfg))
    teacher_cfg_path = oc.teacher_cfg_path
    if not os.path.isabs(teacher_cfg_path):
        teacher_cfg_path = os.path.join(cfg_dir, teacher_cfg_path)
    if not os.path.exists(teacher_cfg_path):
        # Also try relative to Earthformer scripts dir
        alt_path = os.path.join(
            _EF_ROOT, "scripts", "cuboid_transformer", "sst", teacher_cfg_path
        )
        if os.path.exists(alt_path):
            teacher_cfg_path = alt_path
        else:
            raise FileNotFoundError(
                f"Teacher config not found at {teacher_cfg_path} or {alt_path}"
            )

    # --- Resolve teacher checkpoint path ---
    teacher_ckpt = oc.teacher_ckpt_path
    if not os.path.isabs(teacher_ckpt):
        teacher_ckpt = os.path.join(cfg_dir, teacher_ckpt)
    if not os.path.exists(teacher_ckpt):
        # Try relative to Earthformer experiments dir
        alt_path = os.path.join(
            _EF_ROOT, "scripts", "cuboid_transformer", "sst", "experiments",
            teacher_ckpt
        )
        if os.path.exists(alt_path):
            teacher_ckpt = alt_path
        else:
            raise FileNotFoundError(
                f"Teacher checkpoint not found at {teacher_ckpt}"
            )

    # --- Setup DataModule ---
    dataset_cfg = OmegaConf.to_object(oc.dataset)
    dataset_cfg.pop("_target_", None)
    dm = SSTPatchDataModule(**dataset_cfg)
    dm.setup("fit")

    # --- Setup KDRL Module ---
    kdrl_cfg = oc.get("kdrl", OmegaConf.create())
    pl_module = CuboidKDRLModule(
        teacher_ckpt_path=teacher_ckpt,
        oc_file=teacher_cfg_path,
        save_dir=args.save,
        alpha_hard=float(kdrl_cfg.get("alpha_hard", 1.0)),
        beta_soft=float(kdrl_cfg.get("beta_soft", 0.5)),
        gamma_kt=float(kdrl_cfg.get("gamma_kt", 0.5)),
        enc_block_indices=list(kdrl_cfg.get("enc_block_indices", [-2, -1])),
        use_decoder_bridge=bool(kdrl_cfg.get("use_decoder_bridge", True)),
        bridge_num_heads=int(kdrl_cfg.get("bridge_num_heads", 4)),
        bridge_proj_dim=int(kdrl_cfg.get("bridge_proj_dim", 64)),
        bridge_attn_drop=float(kdrl_cfg.get("bridge_attn_drop", 0.0)),
        bridge_proj_drop=float(kdrl_cfg.get("bridge_proj_drop", 0.0)),
    )

    # --- Trainer setup ---
    optim_cfg = OmegaConf.to_object(oc.optim)
    micro_batch = optim_cfg.get("micro_batch_size", dataset_cfg.get("batch_size", 4))
    total_batch = optim_cfg["total_batch_size"]
    num_gpus = max(1, args.gpus)
    accumulate = max(1, total_batch // (micro_batch * num_gpus))

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_mse_epoch",
        dirpath=os.path.join(pl_module.save_dir, "checkpoints"),
        filename="student-kdrl-{epoch:03d}",
        save_top_k=optim_cfg.get("save_top_k", 1),
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
        precision=oc.trainer.get("precision", 16),
        accumulate_grad_batches=accumulate,
        default_root_dir=pl_module.save_dir,
        callbacks=callbacks,
        log_every_n_steps=max(1, int(0.01 * dm.train_dataset.length / micro_batch)),
    )

    # --- Resume from checkpoint if specified ---
    ckpt_path = None
    if args.ckpt_name:
        ckpt_path = os.path.join(pl_module.save_dir, "checkpoints", args.ckpt_name)
        if not os.path.exists(ckpt_path):
            warnings.warn(
                f"Checkpoint {ckpt_path} not found; training from scratch."
            )
            ckpt_path = None

    # --- Train ---
    trainer.fit(model=pl_module, datamodule=dm, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
