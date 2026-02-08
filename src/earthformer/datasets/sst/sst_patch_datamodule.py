"""
Patch-aware SST DataModule for teacher-student distillation.

Yields paired (teacher_x, student_x, student_y, patch_id) for each time window.
Teacher is fixed to one patch; student strides over a grid of patches.
Normalization is computed from the teacher patch training period.
"""
import os
from typing import List, Tuple, Optional
import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


def _parse_slice(s: slice) -> Tuple[float, float]:
    """Return (start, stop) from a slice for lat/lon."""
    return (s.start, s.stop)


class SSTPatchDataset(Dataset):
    """
    Dataset that returns paired teacher and student patch sequences for distillation.
    Each item: (teacher_x, teacher_y, student_x, student_y, patch_id).
    Shapes: (T_in, C, H, W), (T_out, C, H, W), (T_in, C, H, W), (T_out, C, H, W), int.
    """
    def __init__(
        self,
        teacher_data: np.ndarray,
        student_patches: List[np.ndarray],
        patch_ids: List[int],
        in_len: int,
        out_len: int,
    ):
        """
        Args:
            teacher_data: (T_total, C, H, W) normalized array for teacher patch.
            student_patches: list of (T_total, C, H, W) normalized arrays, one per patch.
            patch_ids: list of patch indices (same length as student_patches).
            in_len: input sequence length.
            out_len: target sequence length.
        """
        self.teacher_data = teacher_data
        self.student_patches = student_patches
        self.patch_ids = patch_ids
        self.in_len = in_len
        self.out_len = out_len
        self.seq_len = in_len + out_len
        self.n_patches = len(student_patches)
        n_times = teacher_data.shape[0] - self.seq_len + 1
        self.length = n_times * self.n_patches

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        patch_idx = idx % self.n_patches
        time_idx = idx // self.n_patches
        student_data = self.student_patches[patch_idx]
        pid = self.patch_ids[patch_idx]

        t_end = time_idx + self.seq_len
        teacher_seq = self.teacher_data[time_idx:t_end]
        student_seq = student_data[time_idx:t_end]

        teacher_x = torch.from_numpy(teacher_seq[:self.in_len])
        teacher_y = torch.from_numpy(teacher_seq[self.in_len:self.seq_len])
        student_x = torch.from_numpy(student_seq[:self.in_len])
        student_y = torch.from_numpy(student_seq[self.in_len:self.seq_len])

        return teacher_x, teacher_y, student_x, student_y, pid


class SSTPatchDataModule(pl.LightningDataModule):
    """
    Loads full SST data, computes normalization from teacher patch,
    and builds a grid of student patches with configurable stride.
    """

    def __init__(
        self,
        data_root: str,
        in_len: int = 52,
        out_len: int = 52,
        batch_size: int = 8,
        num_workers: int = 4,
        train_end_year: int = 2015,
        val_end_year: int = 2020,
        teacher_lat_slice: Optional[slice] = None,
        teacher_lon_slice: Optional[slice] = None,
        patch_lat_deg: float = 5.0,
        patch_lon_deg: float = 6.75,
        stride_lat_deg: Optional[float] = None,
        stride_lon_deg: Optional[float] = None,
        student_lat_range: Optional[Tuple[float, float]] = None,
        student_lon_range: Optional[Tuple[float, float]] = None,
        filename: str = "sst.week.mean.nc",
    ):
        super().__init__()
        if teacher_lat_slice is None:
            teacher_lat_slice = slice(15.625, 20.625)
        if teacher_lon_slice is None:
            teacher_lon_slice = slice(65.625, 72.375)
        if stride_lat_deg is None:
            stride_lat_deg = patch_lat_deg
        if stride_lon_deg is None:
            stride_lon_deg = patch_lon_deg
        self.save_hyperparameters(ignore=["teacher_lat_slice", "teacher_lon_slice"])
        self.teacher_lat_slice = teacher_lat_slice
        self.teacher_lon_slice = teacher_lon_slice

        self.mean = None
        self.std = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self._lat_values = None
        self._lon_values = None

    def prepare_data(self) -> None:
        pass

    def _get_patch_slices(
        self,
        lat_values: np.ndarray,
        lon_values: np.ndarray,
    ) -> List[Tuple[slice, slice]]:
        """Build list of (lat_slice, lon_slice) for student grid in index space."""
        lat_min, lat_max = float(lat_values.min()), float(lat_values.max())
        lon_min, lon_max = float(lon_values.min()), float(lon_values.max())
        p_lat = self.hparams.patch_lat_deg
        p_lon = self.hparams.patch_lon_deg
        s_lat = self.hparams.stride_lat_deg
        s_lon = self.hparams.stride_lon_deg

        if self.hparams.student_lat_range is not None:
            lat_min = max(lat_min, self.hparams.student_lat_range[0])
            lat_max = min(lat_max, self.hparams.student_lat_range[1])
        if self.hparams.student_lon_range is not None:
            lon_min = max(lon_min, self.hparams.student_lon_range[0])
            lon_max = min(lon_max, self.hparams.student_lon_range[1])

        slices = []
        lat_start = lat_min
        while lat_start + p_lat <= lat_max + 1e-6:
            lon_start = lon_min
            while lon_start + p_lon <= lon_max + 1e-6:
                lat_sel = (lat_values >= lat_start) & (lat_values < lat_start + p_lat)
                lon_sel = (lon_values >= lon_start) & (lon_values < lon_start + p_lon)
                if np.any(lat_sel) and np.any(lon_sel):
                    lat_idx = np.where(lat_sel)[0]
                    lon_idx = np.where(lon_sel)[0]
                    slices.append((slice(lat_idx[0], lat_idx[-1] + 1), slice(lon_idx[0], lon_idx[-1] + 1)))
                lon_start += s_lon
            lat_start += s_lat
        return slices

    def setup(self, stage: Optional[str] = None) -> None:
        import torch
        # 1. Load the preprocessed file from your uploaded dataset path
        # Replace 'your-dataset-name' with the actual name Kaggle gave your upload
        cache_path = "/kaggle/input/teacher-student-sst-dataset/thermodistill_cache.pt"
        
        print(f">>> [LOADER] Loading preprocessed data from {cache_path}")
        # weights_only=False is required because of the NumPy/Pandas objects
        data = torch.load(cache_path, map_location="cpu", weights_only=False)
        
        self.mean = data['mean']
        self.std = data['std']
        time_index = data['time_index']
        
        # 2. Convert half-precision to float32 for training
        # Teacher: (T, C, H, W) -> float32
        teacher_norm = torch.from_numpy(data['teacher_norm']).to(torch.float32)
        # Student: (N_patches, T, C, H, W) -> float32
        if isinstance(data['all_student_data'], np.ndarray):
            all_student_data = torch.from_numpy(data['all_student_data']).to(torch.float32)
        else:
            all_student_data = data['all_student_data'].to(torch.float32)

        # 3. Time Splitting Logic
        train_slice = slice(None, str(self.hparams.train_end_year))
        val_slice = slice(str(self.hparams.train_end_year + 1), str(self.hparams.val_end_year))
        test_slice = slice(str(self.hparams.val_end_year + 1), None)

        train_idx = time_index.slice_indexer(train_slice.start, train_slice.stop)
        val_idx = time_index.slice_indexer(val_slice.start, val_slice.stop)
        test_idx = time_index.slice_indexer(test_slice.start, test_slice.stop)

        # 4. Helper to create Dataset instances
        def make_dataset(t_idx):
            # Extract the correct time steps for all patches at once
            t_teacher = teacher_norm[t_idx]
            # Slicing the student tensor: [All Patches, Time Subset, C, H, W]
            t_students = [all_student_data[i][t_idx] for i in range(all_student_data.shape[0])]
            
            return SSTPatchDataset(
                teacher_data=t_teacher.numpy(), # Dataset expects numpy arrays
                student_patches=[p.numpy() for p in t_students],
                patch_ids=list(range(len(t_students))),
                in_len=self.hparams.in_len,
                out_len=self.hparams.out_len,
            )

        if stage == "fit" or stage is None:
            self.train_dataset = make_dataset(train_idx)
            self.val_dataset = make_dataset(val_idx)
        if stage == "test" or stage is None:
            self.test_dataset = make_dataset(test_idx)

        print(f">>> [LOADER] Setup complete. Train: {len(train_idx)} weeks, Val: {len(val_idx)} weeks")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )


def _resize_to_match(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize (T, H, W) to (T, target_h, target_w) via simple repeat/mean."""
    import numpy as np
    t, h, w = arr.shape
    if h == target_h and w == target_w:
        return arr
    out = np.zeros((t, target_h, target_w), dtype=arr.dtype)
    for i in range(t):
        out[i] = _resize_2d(arr[i], target_h, target_w)
    return out


def _resize_2d(x: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    # """Resize 2D array to target size using numpy (nearest-style block reduce/sum then scale)."""
    # h, w = x.shape
    # if h == target_h and w == target_w:
    #     return x
    # scale_h = h / target_h
    # scale_w = w / target_w
    # out = np.zeros((target_h, target_w), dtype=x.dtype)
    # for i in range(target_h):
    #     for j in range(target_w):
    #         i0 = int(i * scale_h)
    #         i1 = min(int((i + 1) * scale_h), h)
    #         j0 = int(j * scale_w)
    #         j1 = min(int((j + 1) * scale_w), w)
    #         out[i, j] = np.nanmean(x[i0:i1, j0:j1])
    # return out
    import torch
    import torch.nn.functional as F
    tensor_x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor_x, size=(target_h, target_w), mode='bilinear', align_corners=False)
    return resized.squeeze().numpy()











