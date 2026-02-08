import torch
import numpy as np
from typing import Optional
from earthformer.datasets.sst.sst_patch_datamodule import SSTPatchDataset
from earthformer.datasets.sst.sst_patch_datamodule import SSTPatchDataModule

class SSTCustomDistillDataModule(SSTPatchDataModule):
    def __init__(self, cache_path: str, **kwargs):
        super().__init__(**kwargs)
        self.cache_path = cache_path

    def setup(self, stage: Optional[str] = None):
        # Load the preprocessed file
        # Resolve: Future pickle warning by setting weights_only=False for trusted data
        data = torch.load(self.cache_path, map_location="cpu", weights_only=False)
        
        self.mean, self.std = data['mean'], data['std']
        time_index = data['time_index']
        teacher_norm = torch.from_numpy(data['teacher_norm']).to(torch.float32)
        all_student_data = data['all_student_data'].to(torch.float32)

        # Strict Temporal Slicing: 2001-2015 and 2016
        train_idx = time_index.slice_indexer("2001", "2015")
        test_idx = time_index.slice_indexer("2016", "2016")

        def _build_ds(t_idx):
            # Extract time steps for all 10 patches
            t_teacher = teacher_norm[t_idx].numpy()
            t_students = [all_student_data[i][t_idx].numpy() for i in range(all_student_data.shape[0])]
            
            return SSTPatchDataset(
                teacher_data=t_teacher,
                student_patches=t_students,
                patch_ids=list(range(len(t_students))),
                in_len=self.hparams.in_len,
                out_len=self.hparams.out_len,
            )

        if stage == "fit" or stage is None:
            self.train_dataset = _build_ds(train_idx)
            self.val_dataset = _build_ds(test_idx)
        if stage == "test":
            self.test_dataset = _build_ds(test_idx)

        print(f">>> [CUSTOM LOADER] Train weeks: {train_idx.stop - train_idx.start}")
        print(f">>> [CUSTOM LOADER] Test weeks: {test_idx.stop - test_idx.start}")
