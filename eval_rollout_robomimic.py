import argparse
import csv
import datetime
import json
import os
import pickle
import time
from collections import deque
from typing import Any

import imageio
import numpy as np
import torch
import torch.nn.functional as F

from dataset_robomimic import StatsNormalizer
from model import CJEPA, CJEPAConfig
from vision import VisionBackboneWrapper


def _boolify_success(val: Any):
    if val is None:
        return None
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (int, float, np.integer, np.floating)):
        return bool(val)
    if isinstance(val, np.ndarray):
        if val.size == 1:
            return bool(val.reshape(-1)[0])
        return bool(np.any(val))
    if isinstance(val, dict):
        for k in ("success", "task_success", "is_success", "done_success", "task", "results"):
            if k in val:
                out = _boolify_success(val.get(k))
                if out is not None:
                    return out
        outs = [_boolify_success(v) for v in val.values()]
        outs = [x for x in outs if x is not None]
        return bool(any(outs)) if outs else None
    if isinstance(val, (list, tuple)):
        outs = [_boolify_success(v) for v in val]
        outs = [x for x in outs if x is not None]
        return bool(any(outs)) if outs else None
    return None


def _get_success(env, info: dict | None) -> bool:
    if isinstance(info, dict):
        for key in ("success", "task_success", "is_success", "done_success", "task", "results"):
            if key in info:
                out = _boolify_success(info.get(key))
                if out is not None:
                    return out
        out = _boolify_success(info)
        if out is not None:
            return out

    for attr in ("is_success", "_check_success", "check_success"):
        fn = getattr(env, attr, None)
        if callable(fn):
            out = _boolify_success(fn())
            if out is not None:
                return out

    return False


def _safe_torch_load(path: str, map_location: str | torch.device = "cpu"):
    try:
        return torch.load(path, map_location=map_location)
    except pickle.UnpicklingError as exc:
        if "Weights only load failed" not in str(exc):
            raise
        print("[eval] Checkpoint has non-tensor metadata. Retrying with weights_only=False...")
        return torch.load(path, map_location=map_location, weights_only=False)


def _to_hwc_uint8(img: np.ndarray, image_size: int) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim != 3:
        raise ValueError(f"Expected image with ndim=3, got shape={arr.shape}")

    # CHW -> HWC if needed.
    if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]

    if arr.dtype.kind == "f":
        maxv = float(np.max(arr)) if arr.size else 1.0
        if maxv <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    h, w = arr.shape[:2]
    if h != image_size or w != image_size:
        t = torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0)
        t = F.interpolate(t, size=(image_size, image_size), mode="bilinear", align_corners=False)
        arr = t[0].permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()

    return arr


def _obs_to_inputs(obs: dict, camera_keys: list[str], image_size: int):
    cams_hwc = []
    for key in camera_keys:
        if key in obs:
            raw = obs[key]
        elif key.endswith("_image") and key[:-6] in obs:
            raw = obs[key[:-6]]
        else:
            raw = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        cams_hwc.append(_to_hwc_uint8(raw, image_size))

    if len(cams_hwc) == 1:
        img_for_model = cams_hwc[0]  # (H,W,C)
    else:
        img_for_model = np.stack(cams_hwc, axis=0)  # (V,H,W,C)

    if "robot0_eef_pos" not in obs or "robot0_eef_quat" not in obs:
        raise KeyError("Expected robot0_eef_pos and robot0_eef_quat in env observation")
    proprio = np.concatenate([obs["robot0_eef_pos"], obs["robot0_eef_quat"]], axis=-1)
    video_frame = np.concatenate(cams_hwc, axis=1) if len(cams_hwc) > 1 else cams_hwc[0]
    return img_for_model, proprio, video_frame


def _prepare_images_tensor(imgs_hist_hwc: np.ndarray, device: torch.device) -> torch.Tensor:
    # imgs_hist_hwc: (T, H, W, C) or (T, V, H, W, C), uint8
    img_t = torch.from_numpy(imgs_hist_hwc).float() / 255.0
    if img_t.ndim == 4:
        img_t = img_t.permute(0, 3, 1, 2)  # (T,C,H,W)
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=img_t.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=img_t.dtype).view(1, 3, 1, 1)
        img_t = (img_t - mean) / std
        return img_t.unsqueeze(0).to(device)  # (1,T,C,H,W)

    if img_t.ndim == 5:
        img_t = img_t.permute(0, 1, 4, 2, 3)  # (T,V,C,H,W)
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=img_t.dtype).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=img_t.dtype).view(1, 1, 3, 1, 1)
        img_t = (img_t - mean) / std
        return img_t.unsqueeze(0).to(device)  # (1,T,V,C,H,W)

    raise ValueError(f"Unsupported image history shape: {tuple(imgs_hist_hwc.shape)}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Rollout success-rate evaluation for nano_cjepa Robomimic policy")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--progress_every", type=int, default=10)
    p.add_argument("--max_seconds_per_episode", type=float, default=20.0)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--save_video", action="store_true")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--camera_names", nargs="+", default=["agentview_image", "robot0_eye_in_hand_image"])
    p.add_argument("--image_size", type=int, default=84)
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.output_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join("outputs", f"eval_{timestamp}")
    if args.save_video or args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    ckpt = _safe_torch_load(args.checkpoint, map_location="cpu")
    cfg = CJEPAConfig(**ckpt["cfg"])
    stats = ckpt["stats"]
    normalizer = StatsNormalizer(stats, action_norm_type="minmax")

    policy = CJEPA(cfg).to(device)
    vision_encoder = VisionBackboneWrapper(pretrained=False).to(device)
    policy.load_state_dict(ckpt["policy_state"])
    vision_encoder.load_state_dict(ckpt["vision_state"])
    policy.eval()
    vision_encoder.eval()

    # Build robomimic env from dataset metadata.
    from robomimic.utils.file_utils import get_env_metadata_from_dataset
    from robomimic.utils.env_utils import create_env_from_metadata
    import robomimic.utils.obs_utils as ObsUtils

    rgb_keys = [c if c.endswith("_image") else f"{c}_image" for c in args.camera_names]
    env_camera_names = [c[:-6] if c.endswith("_image") else c for c in args.camera_names]

    obs_modality_specs = {
        "obs": {
            "low_dim": ["robot0_eef_pos", "robot0_eef_quat"],
            "rgb": rgb_keys,
        }
    }
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs)

    env_meta = get_env_metadata_from_dataset(args.dataset)
    env_kwargs = dict(env_meta.get("env_kwargs") or {})
    env_kwargs["camera_names"] = env_camera_names
    env_kwargs["camera_heights"] = int(args.image_size)
    env_kwargs["camera_widths"] = int(args.image_size)
    env_kwargs.pop("camera_height", None)
    env_kwargs.pop("camera_width", None)
    env_meta["env_kwargs"] = env_kwargs

    try:
        env = create_env_from_metadata(env_meta, render=False, render_offscreen=True, use_image_obs=True)
    except TypeError:
        env = create_env_from_metadata(env_meta, render=False, render_offscreen=True)

    horizon = int(args.horizon if args.horizon is not None else getattr(env, "horizon", 400))

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    successes = 0
    episodes_run = 0
    timed_out = 0
    returns = []
    lengths = []
    per_action_ms_all = []
    all_stats = []

    print(f"[eval] Running {args.episodes} rollouts (horizon={horizon})...")
    print(f"[eval] Using camera(s): {args.camera_names}")
    if len(args.camera_names) > 1:
        print("[eval] Multiple cameras requested; policy fuses per-view features by mean for control.")
    if args.output_dir:
        print(f"[eval] Saving outputs to {args.output_dir}")

    for ep in range(int(args.episodes)):
        if hasattr(env, "seed"):
            try:
                env.seed(int(args.seed) + ep)
            except Exception:
                pass

        start_time = time.time()
        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

        img0, prop0, frame0 = _obs_to_inputs(obs, args.camera_names, int(args.image_size))
        img_buf = deque([img0] * cfg.obs_horizon, maxlen=cfg.obs_horizon)
        prop_buf = deque([prop0] * cfg.obs_horizon, maxlen=cfg.obs_horizon)

        video_frames = [frame0] if args.save_video else None
        ep_success = False
        ep_return = 0.0
        ep_len = 0
        action_times_ms = []

        for _t in range(horizon):
            imgs_hist = np.stack(list(img_buf), axis=0)  # (T,H,W,C)
            props_hist = np.stack(list(prop_buf), axis=0)  # (T,7)
            props_hist = normalizer.normalize(props_hist, "proprio")

            img_t = _prepare_images_tensor(imgs_hist, device=device)
            prop_t = torch.from_numpy(props_hist).float().unsqueeze(0).to(device)

            infer_start = time.time()
            with torch.no_grad():
                bsz, t_obs = img_t.shape[:2]
                if img_t.ndim == 5:
                    # (B,T,C,H,W): single-view.
                    img_flat = img_t.reshape(bsz * t_obs, *img_t.shape[2:])
                    vision_feats_flat = vision_encoder(img_flat, return_spatial=False)
                    vision_feats = vision_feats_flat.view(bsz, t_obs, cfg.vision_dim)
                elif img_t.ndim == 6:
                    # (B,T,V,C,H,W): multi-view.
                    n_views = img_t.shape[2]
                    img_flat = img_t.reshape(bsz * t_obs * n_views, *img_t.shape[3:])
                    vision_feats_flat = vision_encoder(img_flat, return_spatial=False)
                    vision_feats = vision_feats_flat.view(bsz, t_obs, n_views, cfg.vision_dim).mean(dim=2)
                else:
                    raise ValueError(f"Unexpected image tensor rank: {img_t.ndim}")
                ctx = torch.cat([vision_feats, prop_t], dim=-1)
                ctx_latents = policy.context_norm(ctx)
                pred_actions_norm = policy(ctx_latents)  # (1, pred_horizon, action_dim)

                action_t = normalizer.unnormalize(pred_actions_norm[:, 0, :], "actions")[0]
                action = action_t.detach().cpu().numpy()
            action_times_ms.append((time.time() - infer_start) * 1000.0)

            if hasattr(env, "action_space"):
                try:
                    action = np.clip(action, env.action_space.low, env.action_space.high)
                except Exception:
                    pass

            step_out = env.step(action)
            if len(step_out) == 5:
                obs, reward, terminated, truncated, info = step_out
                done = bool(terminated or truncated)
            else:
                obs, reward, done, info = step_out

            ep_return += float(reward) if reward is not None else 0.0
            ep_len += 1

            if _get_success(env, info):
                ep_success = True

            img, prop, frame = _obs_to_inputs(obs, args.camera_names, int(args.image_size))
            img_buf.append(img)
            prop_buf.append(prop)
            if args.save_video:
                video_frames.append(frame)

            if ep_success or done:
                break

            if args.max_seconds_per_episode is not None:
                if (time.time() - start_time) > float(args.max_seconds_per_episode):
                    timed_out += 1
                    break

        ep_duration = time.time() - start_time
        successes += int(ep_success)
        episodes_run += 1
        returns.append(ep_return)
        lengths.append(ep_len)
        per_action_ms_all.extend(action_times_ms)

        if args.save_video and args.output_dir and video_frames:
            video_path = os.path.join(args.output_dir, f"video_{ep}.mp4")
            try:
                imageio.mimsave(video_path, video_frames, fps=20)
            except Exception as exc:
                print(f"[eval] Failed to save video for episode {ep}: {exc}")

        ep_stats = {
            "episode": ep,
            "success": ep_success,
            "return": ep_return,
            "length": ep_len,
            "duration": ep_duration,
            "policy_ms_mean": float(np.mean(action_times_ms) if action_times_ms else 0.0),
            "policy_ms_max": float(np.max(action_times_ms) if action_times_ms else 0.0),
        }
        all_stats.append(ep_stats)

        if args.progress_every > 0 and (ep + 1) % int(args.progress_every) == 0:
            print(f"[eval] {ep + 1}/{args.episodes} episodes done, success_rate={successes/(ep+1):.2f}")

    success_rate = float(successes) / float(max(episodes_run, 1))
    print("eval/success_rate:", success_rate)
    print("eval/success_rate_pct:", 100.0 * success_rate)
    print("eval/stats/episodes_run:", float(episodes_run))
    print("eval/stats/episodes_timed_out:", float(timed_out))
    print("eval/stats/return_mean:", float(np.mean(returns) if returns else 0.0))
    print("eval/stats/episode_len_mean:", float(np.mean(lengths) if lengths else 0.0))
    print("eval/policy/ms_mean:", float(np.mean(per_action_ms_all) if per_action_ms_all else 0.0))

    if args.output_dir:
        csv_path = os.path.join(args.output_dir, "stats.csv")
        with open(csv_path, "w", newline="") as f:
            fieldnames = ["episode", "success", "return", "length", "duration", "policy_ms_mean", "policy_ms_max"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for st in all_stats:
                writer.writerow(st)

        summary_path = os.path.join(args.output_dir, "summary.json")
        summary = {
            "success_rate": success_rate,
            "episodes_run": episodes_run,
            "episodes_timed_out": timed_out,
            "return_mean": float(np.mean(returns) if returns else 0.0),
            "episode_len_mean": float(np.mean(lengths) if lengths else 0.0),
            "policy_ms_mean": float(np.mean(per_action_ms_all) if per_action_ms_all else 0.0),
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[eval] Saved stats to {csv_path}")
        print(f"[eval] Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
