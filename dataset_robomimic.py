from __future__ import annotations

import random
from typing import Any
from collections.abc import Sequence

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, ConcatDataset, DataLoader

class StatsNormalizer:
    def __init__(self, stats: dict[str, Any], action_norm_type: str = "minmax"):
        self.stats = stats
        self.action_norm_type = str(action_norm_type).lower()

    def _as_like(self, value, ref):
        if isinstance(ref, torch.Tensor):
            return torch.as_tensor(value, dtype=ref.dtype, device=ref.device)
        return value

    def normalize(self, x, key: str):
        if key not in self.stats:
            return x
        s = self.stats[key]
        eps = 1e-6

        if key == "actions" and self.action_norm_type == "minmax":
            a_min = self._as_like(s["min"], x)
            a_max = self._as_like(s["max"], x)
            span = a_max - a_min
            if isinstance(span, torch.Tensor):
                span = span.clamp(min=eps)
            else:
                span = np.maximum(span, eps)
            return 2.0 * (x - a_min) / span - 1.0

        mean = self._as_like(s["mean"], x)
        std = self._as_like(s["std"], x)
        return (x - mean) / (std + eps)

    def unnormalize(self, x, key: str):
        if key not in self.stats:
            return x
        s = self.stats[key]
        eps = 1e-6

        if key == "actions" and self.action_norm_type == "minmax":
            a_min = self._as_like(s["min"], x)
            a_max = self._as_like(s["max"], x)
            span = a_max - a_min
            if isinstance(span, torch.Tensor):
                span = span.clamp(min=eps)
            else:
                span = np.maximum(span, eps)
            return ((x + 1.0) * 0.5) * span + a_min

        mean = self._as_like(s["mean"], x)
        std = self._as_like(s["std"], x)
        return x * (std + eps) + mean

def compute_global_stats(dataset_paths: list[str], action_norm_type: str = "minmax") -> dict[str, Any]:
    all_actions = []
    all_proprio = []

    for path in dataset_paths:
        with h5py.File(path, "r") as f:
            demos = list(f["data"].keys())
            for demo in demos:
                grp = f[f"data/{demo}"]
                actions = grp["actions"][:]
                pos = grp["obs/robot0_eef_pos"][:]
                quat = grp["obs/robot0_eef_quat"][:]
                proprio = np.concatenate([pos, quat], axis=-1)
                all_actions.append(actions)
                all_proprio.append(proprio)

    if not all_actions:
        raise ValueError("No valid actions found to compute dataset stats.")

    actions = np.concatenate(all_actions, axis=0)
    proprio = np.concatenate(all_proprio, axis=0)

    action_std = actions.std(0)
    action_std[action_std < 1e-6] = 1.0
    proprio_std = proprio.std(0)
    proprio_std[proprio_std < 1e-6] = 1.0

    return {
        "actions": {
            "mean": actions.mean(0),
            "std": action_std,
            "min": actions.min(0),
            "max": actions.max(0),
            "norm_type": action_norm_type,
        },
        "proprio": {
            "mean": proprio.mean(0),
            "std": proprio_std,
        },
    }

class RobomimicDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        obs_horizon: int,
        pred_horizon: int,
        stats: dict,
        indices: list[tuple[str, int]],
        camera_names: list[str] = ["agentview_image"],
        image_size: int = 84,
        action_norm_type: str = "minmax"
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.image_size = image_size
        self.camera_names = camera_names
        self.indices = indices

        self.f = h5py.File(dataset_path, "r")
        self.normalizer = StatsNormalizer(
            stats,
            action_norm_type=action_norm_type,
        )

        # ImageNet normalization standard for DINO
        self.img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 1, 3, 1, 1)
        self.img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 1, 3, 1, 1)

    def __len__(self) -> int:
        return len(self.indices)

    def _pad_window(self, arr: np.ndarray, start: int, end: int) -> np.ndarray:
        n = arr.shape[0]
        if start < 0:
            pad = np.repeat(arr[0:1], -start, axis=0)
            valid = arr[0:end]
            out = np.concatenate([pad, valid], axis=0)
        elif end > n:
            valid = arr[start:n]
            pad = np.repeat(arr[n - 1 : n], end - n, axis=0)
            out = np.concatenate([valid, pad], axis=0)
        else:
            out = arr[start:end]
        return out

    def _get_images(self, demo_grp, start_obs: int, end_obs: int) -> torch.Tensor:
        cams = []
        for cam in self.camera_names:
            key = f"obs/{cam}"
            if key in demo_grp:
                raw = demo_grp[key][:]
                seq = self._pad_window(raw, start_obs, end_obs)
            else:
                seq = np.zeros((self.obs_horizon, self.image_size, self.image_size, 3), dtype=np.uint8)
            cams.append(seq)

        # (V, T, H, W, C) -> (T, V, C, H, W)
        arr = np.stack(cams, axis=0)
        t_v_h_w_c = torch.from_numpy(arr).permute(1, 0, 4, 2, 3).float() / 255.0

        t, v, c, h, w = t_v_h_w_c.shape
        if h != self.image_size or w != self.image_size:
            flat = t_v_h_w_c.reshape(t * v, c, h, w)
            flat = F.interpolate(flat, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
            t_v_h_w_c = flat.view(t, v, c, self.image_size, self.image_size)

        t_v_h_w_c = (t_v_h_w_c - self.img_mean) / self.img_std

        if t_v_h_w_c.shape[1] == 1:
            return t_v_h_w_c[:, 0]
        return t_v_h_w_c

    def __getitem__(self, idx: int):
        demo_key, t = self.indices[idx]
        demo_grp = self.f[f"data/{demo_key}"]

        start_obs = t - self.obs_horizon + 1
        end_obs = t + 1

        imgs = self._get_images(demo_grp, start_obs, end_obs)

        pos = demo_grp["obs/robot0_eef_pos"][:]
        quat = demo_grp["obs/robot0_eef_quat"][:]
        prop = np.concatenate([pos, quat], axis=-1)
        prop_seq = self._pad_window(prop, start_obs, end_obs)
        prop_seq = self.normalizer.normalize(prop_seq, "proprio")
        prop_t = torch.from_numpy(prop_seq).float()

        actions = demo_grp["actions"][:]
        start_act = t - self.obs_horizon + 1
        end_act = t + self.pred_horizon
        act_seq = self._pad_window(actions, start_act, end_act)
        act_seq = self.normalizer.normalize(act_seq, "actions")
        act_t = torch.from_numpy(act_seq).float()

        return imgs, prop_t, act_t

    def __del__(self):
        f = getattr(self, "f", None)
        if f is not None:
            try:
                f.close()
            except Exception:
                pass


def build_indices(dataset_path: str, obs_horizon: int) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    with h5py.File(dataset_path, "r") as f:
        demos = list(f["data"].keys())
        for demo in demos:
            n_steps = int(f[f"data/{demo}/actions"].shape[0])
            for t in range(obs_horizon - 1, n_steps):
                out.append((demo, t))
    return out

def get_robomimic_dataloaders(
    dataset_paths: list[str],
    batch_size: int,
    obs_horizon: int,
    pred_horizon: int,
    camera_names: list[str] = ["agentview_image"],
    image_size: int = 84,
    val_split: float = 0.2,
    num_workers: int = 8
):
    stats = compute_global_stats(dataset_paths)

    train_sets = []
    val_sets = []
    random.seed(42)

    for path in dataset_paths:
        idx = build_indices(path, obs_horizon)
        random.shuffle(idx)
        split = int(len(idx) * (1.0 - val_split))
        train_idx = idx[:split]
        val_idx = idx[split:]

        train_sets.append(RobomimicDataset(path, obs_horizon, pred_horizon, stats, train_idx, camera_names, image_size))
        val_sets.append(RobomimicDataset(path, obs_horizon, pred_horizon, stats, val_idx, camera_names, image_size))

    train_ds = ConcatDataset(train_sets)
    val_ds = ConcatDataset(val_sets)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_dl, val_dl, stats
