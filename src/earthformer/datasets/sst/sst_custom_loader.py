import torch
import numpy as np
from typing import Optional
from earthformer.datasets.sst.sst_patch_datamodule import SSTPatchDataset
from earthformer.datasets.sst.sst_patch_datamodule import SSTPatchDataModule

class SSTCustomDistillDataModule(SSTPatchDataModule):
    def __init__(self, cache_path: str, **kwargs):
        base_keys = ['data_dir', 'batch_size', 'num_workers', 'in_len', 'out_len', 'layout', 'img_size']
        
        # 2. Separate kwargs into 'base' and 'custom'
        base_kwargs = {k: v for k, v in kwargs.items() if k in base_keys}
        custom_kwargs = {k: v for k, v in kwargs.items() if k not in base_keys}
        
        # 3. Initialize the parent with only the keys it knows
        super().__init__(**base_kwargs)
        
        # 4. Store your custom keys manually
        self.cache_path = cache_path
        self.train_start_year = custom_kwargs.get('train_start_year', 2001)
        self.train_end_year = custom_kwargs.get('train_end_year', 2015)
        self.val_end_year = custom_kwargs.get('val_end_year', 2016)

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
