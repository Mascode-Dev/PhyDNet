"""GOES-16 satellite cloud imagery dataset for PhyDNet.
Downloads visible-channel (C02) CONUS scans from NOAA AWS,
computes optical-flow velocities via OpenCV Farneback."""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from datetime import datetime


def _download_goes(date_str, start_hour, hours, interval_min, raw_dir):
    """Download GOES-16 CONUS visible (C02) NetCDF files from NOAA S3."""
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    dt = datetime.strptime(date_str, '%Y-%m-%d')
    year = dt.year
    doy = dt.timetuple().tm_yday

    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED), region_name='us-east-1')
    bucket = 'noaa-goes16'

    downloaded = 0
    for h in range(start_hour, start_hour + hours):
        prefix = f'ABI-L1b-RadC/{year}/{doy:03d}/{h:02d}/'
        try:
            response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=200)
            if 'Contents' not in response:
                continue

            files = sorted([obj['Key'] for obj in response['Contents']
                           if 'M6C02' in obj['Key'] and obj['Key'].endswith('.nc')])

            for i, key in enumerate(files):
                if i % max(1, interval_min // 5) != 0:
                    continue
                local_path = os.path.join(raw_dir, os.path.basename(key))
                if not os.path.exists(local_path):
                    s3.download_file(bucket, key, local_path)
                    downloaded += 1
        except Exception as e:
            print(f"  S3 warning for {prefix}: {e}")

    if downloaded == 0:
        raise RuntimeError(
            f"No GOES-16 files downloaded for {date_str} hours {start_hour}-{start_hour+hours}. "
            f"Check internet connection and date validity."
        )
    print(f"  Downloaded {downloaded} GOES-16 NetCDF files")
    return downloaded


def _process_netcdf_files(raw_dir, crop_size):
    """Read NetCDF files, extract visible radiance, normalize, crop, resize."""
    import netCDF4 as nc
    import cv2

    nc_files = sorted([f for f in os.listdir(raw_dir) if f.endswith('.nc')])
    if len(nc_files) == 0:
        raise RuntimeError(f"No .nc files found in {raw_dir}. Run download first.")

    frames = []
    for f in nc_files:
        ds = nc.Dataset(os.path.join(raw_dir, f))
        h = ds.dimensions['y'].size
        w = ds.dimensions['x'].size

        crop_h = min(h, int(w * 0.55))
        h_start = max(0, (h - crop_h) // 2)
        w_start = max(0, (w - crop_h) // 2)
        rad = np.array(ds.variables['Rad'][h_start:h_start + crop_h, w_start:w_start + crop_h], dtype=np.float32)
        ds.close()
        rad = cv2.resize(rad, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
        frames.append(rad)

    frames = np.stack(frames, axis=0)  # (N, H, W)
    return frames


def _normalize_frames(frames):
    """Clip and normalize to [0, 1] using global percentiles across all frames."""
    p_low = np.percentile(frames, 0.5)
    p_high = np.percentile(frames, 99.5)
    frames = np.clip(frames, p_low, p_high)
    frames = (frames - p_low) / max(p_high - p_low, 1e-8)
    return frames


def _compute_optical_flow(frames):
    """Compute per-frame velocity via Farneback optical flow on normalized frames."""
    import cv2

    n = len(frames)
    velocities = np.zeros((n, 2), dtype=np.float32)
    for i in range(1, n):
        prev = (frames[i - 1] * 255).astype(np.uint8)
        curr = (frames[i] * 255).astype(np.uint8)
        flow = cv2.calcOpticalFlowFarneback(
            prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        velocities[i, 0] = flow[..., 0].mean()
        velocities[i, 1] = flow[..., 1].mean()
    if n > 1:
        velocities[0] = velocities[1]
    return velocities


class GOESCloud(Dataset):
    """GOES-16 visible satellite cloud dataset for PhyDNet.

    Downloads CONUS C02 (0.64 um visible) scans from NOAA AWS,
    crops a central region, resizes to crop_size x crop_size,
    computes per-frame optical flow velocities.

    Args:
        root: data cache directory (stores raw NetCDF files)
        date_str: 'YYYY-MM-DD' date to download
        start_hour: UTC hour to start download
        hours: number of hours to download
        interval_min: sampling interval in minutes (GOES-16 CONUS is ~5 min native)
        crop_size: output image size (must be 64 for PhyDNet)
        n_frames_input: input sequence length
        n_frames_output: output sequence length
        is_train: train/test split
        train_ratio: fraction of sequences for training
    """

    def __init__(self, root='data/cloud', date_str='2024-07-15',
                 start_hour=18, hours=6, interval_min=10,
                 crop_size=64, n_frames_input=10, n_frames_output=10,
                 is_train=True, train_ratio=0.8):

        self.crop_size = crop_size
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.total_frames = n_frames_input + n_frames_output
        self.is_train = is_train

        os.makedirs(root, exist_ok=True)
        raw_dir = os.path.join(root, f'raw_{date_str.replace("-", "")}')
        os.makedirs(raw_dir, exist_ok=True)

        if len(os.listdir(raw_dir)) == 0:
            _download_goes(date_str, start_hour, hours, interval_min, raw_dir)

        frames = _process_netcdf_files(raw_dir, crop_size)
        frames = _normalize_frames(frames)
        velocities = _compute_optical_flow(frames)

        self.frames = frames
        self.velocities = velocities

        n_sequences = max(1, frames.shape[0] - self.total_frames + 1)
        n_train = int(n_sequences * train_ratio)
        self.indices = list(range(0, n_train if is_train else n_sequences))
        if not is_train:
            self.indices = list(range(n_train, n_sequences))

        if len(self.indices) == 0:
            self.indices = list(range(n_sequences))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        end = start + self.total_frames

        images = self.frames[start:end]
        vel = self.velocities[start:end]

        images = images[:, np.newaxis, :, :]  # (T, 1, H, W)

        inp_img = torch.from_numpy(images[:self.n_frames_input].copy()).float()
        out_img = torch.from_numpy(images[self.n_frames_input:].copy()).float()
        inp_vel = torch.from_numpy(vel[:self.n_frames_input].copy()).float()
        out_vel = torch.from_numpy(vel[self.n_frames_input:].copy()).float()

        return [idx, inp_img, out_img, inp_vel, out_vel]
