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
import torch.nn.functional as F


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
        in_len: int = 12,
        out_len: int = 12,
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
        student_stride_south: bool = False,
        student_stride_lat_fraction: float = 0.30,
        student_num_stride_patches: int = 10,
        filename: str = "sst.week.mean.nc",
        use_cache: bool = True,
        cache_path: str = "/kaggle/input/teacher-student-sst-dataset/thermodistill_cache.pt"
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
        self.use_cache = use_cache
        self.cache_path = cache_path

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
        """Build list of (lat_slice, lon_slice) for student patches in index space.
        If student_stride_south is True: N patches striding south by (student_stride_lat_fraction * patch_lat_deg).
        Otherwise: 2D grid with stride_lat_deg / stride_lon_deg.
        """
        p_lat = self.hparams.patch_lat_deg
        p_lon = self.hparams.patch_lon_deg

        if getattr(self.hparams, "student_stride_south", False):
            stride_deg = self.hparams.student_stride_lat_fraction * p_lat
            n_patches = getattr(self.hparams, "student_num_stride_patches", 10)
            t_lat_start = getattr(self.teacher_lat_slice, "start", 15.625)
            t_lat_stop = getattr(self.teacher_lat_slice, "stop", 20.625)
            t_lon_start = getattr(self.teacher_lon_slice, "start", 65.625)
            t_lon_stop = getattr(self.teacher_lon_slice, "stop", 72.375)
            slices = []
            for i in range(n_patches):
                lat_start = t_lat_start - i * stride_deg
                lat_stop = t_lat_stop - i * stride_deg
                lat_sel = (lat_values >= lat_start) & (lat_values < lat_stop)
                lon_sel = (lon_values >= t_lon_start) & (lon_values < t_lon_stop)
                if np.any(lat_sel) and np.any(lon_sel):
                    lat_idx = np.where(lat_sel)[0]
                    lon_idx = np.where(lon_sel)[0]
                    slices.append((slice(int(lat_idx[0]), int(lat_idx[-1]) + 1), slice(int(lon_idx[0]), int(lon_idx[-1]) + 1)))
            return slices

        lat_min, lat_max = float(lat_values.min()), float(lat_values.max())
        lon_min, lon_max = float(lon_values.min()), float(lon_values.max())
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
        # 1. Attempt to use cached data for speed if requested and available
        if self.use_cache and os.path.exists(self.cache_path):
            print(f">>> [LOADER] Using Cached Data from {self.cache_path}")
            # weights_only=False is required because of the NumPy/Pandas objects
            data = torch.load(self.cache_path, map_location="cpu", weights_only=False)
            
            self.mean = data['mean']
            self.std = data['std']
            time_index = data['time_index']
            
            # Convert half-precision to float32 for training
            teacher_norm = torch.from_numpy(data['teacher_norm']).to(torch.float32)
            if isinstance(data['all_student_data'], np.ndarray):
                all_student_data = torch.from_numpy(data['all_student_data']).to(torch.float32)
            else:
                all_student_data = data['all_student_data'].to(torch.float32)
            
            n_patches = all_student_data.shape[0]
            student_patches = [all_student_data[i] for i in range(n_patches)]
            patch_ids = list(range(n_patches))
        else:
            # 2. Fallback to raw processing if cache is missing or disabled
            print(f">>> [LOADER] Processing Raw NetCDF from {self.hparams.data_root}")
            file_path = os.path.join(self.hparams.data_root, self.hparams.filename)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Dataset file not found at {file_path}")

            ds = xr.open_dataset(file_path)
            self._lat_values = ds.lat.values
            self._lon_values = ds.lon.values

            train_slice = slice(None, str(self.hparams.train_end_year))
            ds_teacher = ds.sel(
                lat=self.teacher_lat_slice,
                lon=self.teacher_lon_slice,
            )
            train_teacher = ds_teacher["sst"].sel(time=train_slice).values.astype(np.float32)
            self.mean = float(np.nanmean(train_teacher))
            self.std = float(np.nanstd(train_teacher))
            if self.std < 1e-8:
                self.std = 1.0

            teacher_full = ds_teacher["sst"].values.astype(np.float32)
            teacher_full = np.nan_to_num(teacher_full, nan=self.mean)
            teacher_norm = (teacher_full - self.mean) / self.std
            teacher_norm = torch.from_numpy(teacher_norm[:, np.newaxis, :, :])

            patch_slices = self._get_patch_slices(self._lat_values, self._lon_values)
            student_patches = []
            th, tw = teacher_norm.shape[2], teacher_norm.shape[3]
            for (lat_sli, lon_sli) in patch_slices:
                sub = ds["sst"].isel(lat=lat_sli, lon=lon_sli).values.astype(np.float32)
                sub = np.nan_to_num(sub, nan=self.mean)
                sub = (sub - self.mean) / self.std
                # Resize to teacher patch grid if needed
                if sub.shape[1] != th or sub.shape[2] != tw:
                    sub = _resize_to_match(sub, th, tw)
                sub = torch.from_numpy(sub[:, np.newaxis, :, :])
                student_patches.append(sub)

            patch_ids = list(range(len(student_patches)))
            time_index = ds.get_index("time")
            ds.close()

        # 3. Time Splitting Logic
        train_slice = slice(None, str(self.hparams.train_end_year))
        val_slice = slice(str(self.hparams.train_end_year + 1), str(self.hparams.val_end_year))
        test_slice = slice(str(self.hparams.val_end_year + 1), None)

        train_idx = time_index.slice_indexer(train_slice.start, train_slice.stop)
        val_idx = time_index.slice_indexer(val_slice.start, val_slice.stop)
        test_idx = time_index.slice_indexer(test_slice.start, test_slice.stop)

        # 4. Helper to create Dataset instances
        def make_dataset(t_idx):
            t_teacher = teacher_norm[t_idx]
            t_students = [p[t_idx] for p in student_patches]
            
            return SSTPatchDataset(
                teacher_data=t_teacher.numpy() if isinstance(t_teacher, torch.Tensor) else t_teacher,
                student_patches=[p.numpy() if isinstance(p, torch.Tensor) else p for p in t_students],
                patch_ids=patch_ids,
                in_len=self.hparams.in_len,
                out_len=self.hparams.out_len,
            )

        if stage == "fit" or stage is None:
            self.train_dataset = make_dataset(train_idx)
            self.val_dataset = make_dataset(val_idx)
        if stage == "test" or stage is None:
            self.test_dataset = make_dataset(test_idx)

        def get_idx_len(idx):
            if isinstance(idx, slice):
                return idx.stop - idx.start
            return len(idx)

        train_ds = self.train_dataset
        val_ds = self.val_dataset
        
        print("--- GEOSPATIAL PATCH DETAILS ---")
        print(f"Total Student Patches: {train_ds.n_patches}")
        print(f"Time Steps (Train): {len(train_ds) // train_ds.n_patches} weeks")
        print(f"Time Steps (Val):   {len(val_ds) // val_ds.n_patches} weeks")
        
        if hasattr(self, '_lat_values') and self._lat_values is not None:
            patch_slices = self._get_patch_slices(self._lat_values, self._lon_values)
            for i, (lat_slice, lon_slice) in enumerate(patch_slices):
                lats = self._lat_values[lat_slice]
                lons = self._lon_values[lon_slice]
                print(f"Patch {i:02d}: Lat [{lats.min():.3f} to {lats.max():.3f}], "
                      f"Lon [{lons.min():.3f} to {lons.max():.3f}]")
        
        tx, ty, sx, sy, pid = train_ds[0]
        print("\n--- TENSOR DIMENSIONS ---")
        print(f"Input Sequence (X):  {sx.shape}  # (Time, Channel, Height, Width)")
        print(f"Target Sequence (Y): {sy.shape}")
        print(f"Pixel Range (Mean):  {sx.mean():.4f}")

        print(f">>> [LOADER] Setup complete. Train: {get_idx_len(train_idx)} weeks, Val: {get_idx_len(val_idx)} weeks")

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
    t, h, w = arr.shape
    if h == target_h and w == target_w:
        return arr
    out = np.zeros((t, target_h, target_w), dtype=arr.dtype)
    for i in range(t):
        out[i] = _resize_2d(arr[i], target_h, target_w)
    return out


def _resize_2d(x: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize 2D array to target size using bilinear interpolation via torch."""
    tensor_x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor_x, size=(target_h, target_w), mode='bilinear', align_corners=False)
    return resized.squeeze().numpy()


