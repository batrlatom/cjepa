import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dataset_robomimic import get_robomimic_dataloaders
from model import CJEPA, CJEPAConfig
from vision import VisionBackboneWrapper


@dataclass
class _AsyncEvalJob:
    epoch: int
    checkpoint_path: str
    output_dir: str
    process: subprocess.Popen
    launched_at: float


class AsyncRolloutEvaluator:
    def __init__(
        self,
        *,
        run_dir: Path,
        dataset_path: str,
        camera_names: list[str],
        every_n_epochs: int = 0,
        episodes: int = 10,
        horizon: int | None = None,
        seed: int = 0,
        device: str = "auto",
        progress_every: int = 0,
        max_seconds_per_episode: float = 20.0,
        max_concurrent_jobs: int = 1,
        image_size: int = 84,
        save_video: bool = False,
    ):
        self.run_dir = run_dir.resolve()
        self.dataset_path = str(Path(dataset_path).resolve())
        self.camera_names = list(camera_names)
        self.every_n_epochs = max(0, int(every_n_epochs))
        self.episodes = max(1, int(episodes))
        self.horizon = int(horizon) if horizon is not None else None
        self.seed = int(seed)
        self.device = str(device)
        self.progress_every = max(0, int(progress_every))
        self.max_seconds_per_episode = float(max_seconds_per_episode)
        self.max_concurrent_jobs = max(1, int(max_concurrent_jobs))
        self.image_size = int(image_size)
        self.save_video = bool(save_video)

        self.output_dir = self.run_dir / "async_eval"
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.eval_script = Path(__file__).resolve().parent / "eval_rollout_robomimic.py"
        self.jobs: list[_AsyncEvalJob] = []

        if self.enabled:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self.every_n_epochs > 0

    def _collect_finished_jobs(self) -> None:
        active: list[_AsyncEvalJob] = []
        for job in self.jobs:
            if job.process.poll() is None:
                active.append(job)
                continue

            elapsed = max(time.time() - job.launched_at, 0.0)
            summary_path = Path(job.output_dir) / "summary.json"
            success_rate = float("nan")
            return_mean = float("nan")
            episode_len_mean = float("nan")
            if summary_path.exists():
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    success_rate = float(summary.get("success_rate", float("nan")))
                    return_mean = float(summary.get("return_mean", float("nan")))
                    episode_len_mean = float(summary.get("episode_len_mean", float("nan")))
                except Exception:
                    pass

            print(
                f"[async-eval] epoch={job.epoch} finished "
                f"exit_code={job.process.returncode} "
                f"success_rate={success_rate:.3f} "
                f"return_mean={return_mean:.3f} "
                f"episode_len_mean={episode_len_mean:.2f} "
                f"runtime_sec={elapsed:.1f}"
            )

        self.jobs = active

    def _launch_job(self, epoch: int, checkpoint_payload: dict) -> None:
        run_dir = self.output_dir / f"rollout_epoch_{epoch:04d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        ckpt_path = self.ckpt_dir / f"epoch_{epoch:04d}.pt"
        torch.save(checkpoint_payload, ckpt_path)

        ckpt_path_abs = ckpt_path.resolve()
        run_dir_abs = run_dir.resolve()
        cmd = [
            sys.executable,
            str(self.eval_script),
            "--checkpoint",
            str(ckpt_path_abs),
            "--dataset",
            self.dataset_path,
            "--episodes",
            str(self.episodes),
            "--seed",
            str(self.seed + epoch),
            "--progress_every",
            str(self.progress_every),
            "--max_seconds_per_episode",
            str(self.max_seconds_per_episode),
            "--device",
            self.device,
            "--output_dir",
            str(run_dir_abs),
            "--image_size",
            str(self.image_size),
            "--camera_names",
            *self.camera_names,
        ]
        if self.horizon is not None:
            cmd.extend(["--horizon", str(self.horizon)])
        if self.save_video:
            cmd.append("--save_video")

        process = subprocess.Popen(cmd, cwd=str(self.eval_script.parent))
        self.jobs.append(
            _AsyncEvalJob(
                epoch=epoch,
                checkpoint_path=str(ckpt_path_abs),
                output_dir=str(run_dir_abs),
                process=process,
                launched_at=time.time(),
            )
        )
        print(f"[async-eval] launched epoch={epoch} pid={process.pid} output={run_dir}")

    def on_epoch_end(self, epoch_1b: int, checkpoint_payload: dict) -> None:
        self._collect_finished_jobs()
        if epoch_1b <= 0:
            return
        if not self.enabled:
            return
        if epoch_1b % self.every_n_epochs != 0:
            return
        if len(self.jobs) >= self.max_concurrent_jobs:
            print(
                f"[async-eval] skipped launch at epoch={epoch_1b}: "
                f"{len(self.jobs)} active job(s) >= max_concurrent_jobs={self.max_concurrent_jobs}"
            )
            return
        self._launch_job(epoch_1b, checkpoint_payload)

    def finalize(self) -> None:
        self._collect_finished_jobs()
        if self.jobs:
            print(
                f"[async-eval] training ended with {len(self.jobs)} active rollout job(s). "
                "They will continue in background."
            )


def _make_checkpoint_payload(
    *,
    epoch: int,
    policy: CJEPA,
    vision_encoder: VisionBackboneWrapper,
    stats: dict,
    cfg: CJEPAConfig,
    val_loss: float,
    cime: dict | None = None,
) -> dict:
    payload = {
        "epoch": epoch,
        "policy_state": policy.state_dict(),
        "vision_state": vision_encoder.state_dict(),
        "stats": stats,
        "cfg": {
            "action_dim": cfg.action_dim,
            "proprio_dim": cfg.proprio_dim,
            "vision_dim": cfg.vision_dim,
            "D": cfg.D,
            "n_heads": cfg.n_heads,
            "n_layers": cfg.n_layers,
            "obs_horizon": cfg.obs_horizon,
            "pred_horizon": cfg.pred_horizon,
        },
        "val_loss": float(val_loss),
    }
    if cime is not None:
        payload["cime"] = dict(cime)
    return payload


def _make_counterfactual_context(ctx_latents: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    """
    Create counterfactual context by masking non-anchor time tokens.
    The first observation token acts as an identity anchor.
    """
    if mask_ratio <= 0.0:
        return ctx_latents

    B, T_obs, D = ctx_latents.shape
    if T_obs <= 1:
        return ctx_latents

    cf_ctx = ctx_latents.clone()
    n_eligible = T_obs - 1
    k = int(round(mask_ratio * n_eligible))
    k = max(1, min(k, n_eligible))

    for b in range(B):
        idx = torch.randperm(n_eligible, device=ctx_latents.device)[:k] + 1
        anchor = ctx_latents[b, 0].detach().clone().unsqueeze(0).expand(idx.numel(), D).clone()
        cf_ctx[b, idx] = anchor

    return cf_ctx


def _make_negative_actions(target_actions: torch.Tensor, noise_std: float) -> torch.Tensor:
    """
    Generate negative action trajectories via batch permutation + noise.
    """
    B = target_actions.shape[0]
    if B > 1:
        perm = torch.randperm(B, device=target_actions.device)
        neg = target_actions[perm].clone()
        same = perm == torch.arange(B, device=target_actions.device)
        if same.any():
            neg[same] = neg[same] + noise_std * torch.randn_like(neg[same])
    else:
        neg = target_actions + noise_std * torch.randn_like(target_actions)

    # Small universal perturbation to reduce accidental positives.
    neg = neg + 0.25 * noise_std * torch.randn_like(neg)
    return neg.clamp(-1.0, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", required=True, help="Path(s) to robomimic HDF5 dataset(s)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--run_dir", type=str, default="runs/robomimic/run_1")
    parser.add_argument(
        "--camera_names",
        nargs="+",
        default=["agentview_image", "robot0_eye_in_hand_image"],
        help="Camera observation keys to use (multi-view supported)",
    )
    parser.add_argument(
        "--rollout_eval_dataset",
        type=str,
        default=None,
        help="Must match the training dataset path (kept for backward compatibility)",
    )
    parser.add_argument("--rollout_eval_every_epochs", type=int, default=0, help="Run async rollout eval every N epochs (0 disables)")
    parser.add_argument("--rollout_eval_episodes", type=int, default=10, help="Episodes per async rollout eval")
    parser.add_argument("--rollout_eval_horizon", type=int, default=None, help="Optional fixed rollout horizon")
    parser.add_argument("--rollout_eval_seed", type=int, default=0)
    parser.add_argument("--rollout_eval_device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--rollout_eval_progress_every", type=int, default=0)
    parser.add_argument("--rollout_eval_max_seconds_per_episode", type=float, default=20.0)
    parser.add_argument("--rollout_eval_max_concurrent_jobs", type=int, default=1)
    parser.add_argument("--rollout_eval_save_video", action="store_true")
    parser.add_argument("--use_cime", action="store_true", help="Enable Counterfactual Intervention Margin Energy loss")
    parser.add_argument("--cime_lambda_cf", type=float, default=0.1, help="Weight for counterfactual margin term")
    parser.add_argument("--cime_lambda_neg", type=float, default=0.1, help="Weight for negative-action margin term")
    parser.add_argument("--cime_margin", type=float, default=0.1, help="Margin value for hinge losses")
    parser.add_argument("--cime_cf_mask_ratio", type=float, default=1.0, help="Fraction of non-anchor history tokens masked")
    parser.add_argument("--cime_neg_noise_std", type=float, default=0.2, help="Stddev for negative action perturbations")
    args = parser.parse_args()
    train_dataset_paths = [str(Path(p).resolve()) for p in args.dataset]

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Set up Config for Lift/Robomimic standard
    cfg = CJEPAConfig(
        action_dim=7,
        proprio_dim=7,
        vision_dim=512,
        D=256,
        n_heads=8,
        n_layers=4,
        obs_horizon=2,
        pred_horizon=16
    )

    # 1. Dataset setup
    train_dl, val_dl, stats = get_robomimic_dataloaders(
        dataset_paths=args.dataset,
        batch_size=args.batch_size,
        obs_horizon=cfg.obs_horizon,
        pred_horizon=cfg.pred_horizon,
        camera_names=args.camera_names,
        image_size=84,
        val_split=0.1
    )
    print(f"Using camera(s): {args.camera_names}")
    
    # 2. Model setup
    vision_encoder = VisionBackboneWrapper(pretrained=True).to(device)
    policy = CJEPA(cfg).to(device)
    
    # Optimizer (Train both policy and fine-tune vision encoder)
    optimizer = optim.AdamW([
        {'params': policy.parameters()},
        {'params': vision_encoder.parameters(), 'lr': args.lr * 0.1} # Fine-tune CNN slightly slower
    ], lr=args.lr, weight_decay=1e-4)
    
    # Loss for Behavior Cloning
    criterion = nn.MSELoss()
    cime_cfg = {
        "enabled": bool(args.use_cime),
        "lambda_cf": float(args.cime_lambda_cf),
        "lambda_neg": float(args.cime_lambda_neg),
        "margin": float(args.cime_margin),
        "cf_mask_ratio": float(args.cime_cf_mask_ratio),
        "neg_noise_std": float(args.cime_neg_noise_std),
    }

    best_val_loss = float('inf')
    rollout_eval_dataset = train_dataset_paths[0]
    if args.rollout_eval_dataset is not None:
        requested = str(Path(args.rollout_eval_dataset).resolve())
        if requested != rollout_eval_dataset:
            raise ValueError(
                f"--rollout_eval_dataset must match training dataset ({rollout_eval_dataset}), got {requested}"
            )

    if int(args.rollout_eval_every_epochs) > 0 and len(train_dataset_paths) != 1:
        raise ValueError(
            "Async rollout eval requires exactly one training dataset path so eval dataset matches train dataset."
        )

    rollout_eval = AsyncRolloutEvaluator(
        run_dir=run_dir,
        dataset_path=rollout_eval_dataset,
        camera_names=args.camera_names,
        every_n_epochs=args.rollout_eval_every_epochs,
        episodes=args.rollout_eval_episodes,
        horizon=args.rollout_eval_horizon,
        seed=args.rollout_eval_seed,
        device=args.rollout_eval_device,
        progress_every=args.rollout_eval_progress_every,
        max_seconds_per_episode=args.rollout_eval_max_seconds_per_episode,
        max_concurrent_jobs=args.rollout_eval_max_concurrent_jobs,
        image_size=84,
        save_video=args.rollout_eval_save_video,
    )
    if rollout_eval.enabled:
        print(
            "[train] Async rollout eval enabled: "
            f"every={args.rollout_eval_every_epochs} episodes={args.rollout_eval_episodes} "
            f"dataset={rollout_eval_dataset} device={args.rollout_eval_device}"
        )
    if args.use_cime:
        print(
            "[train] CIME enabled: "
            f"lambda_cf={args.cime_lambda_cf} lambda_neg={args.cime_lambda_neg} "
            f"margin={args.cime_margin} cf_mask_ratio={args.cime_cf_mask_ratio} "
            f"neg_noise_std={args.cime_neg_noise_std}"
        )

    print(f"Starting training for {args.epochs} epochs...")
    try:
        for epoch in range(args.epochs):
            policy.train()
            vision_encoder.train()
            train_loss = 0.0
            train_bc_loss = 0.0
            train_cf_loss = 0.0
            train_neg_loss = 0.0

            start_t = time.time()
            for imgs, prop_t, act_t in tqdm(train_dl, desc=f"Epoch {epoch+1}"):
                # imgs: (B, T_obs, C, H, W) or (B, T_obs, V, C, H, W)
                # prop_t: (B, T_obs, 7)
                # act_t: (B, T_obs + T_pred - 1, 7) -> we only want the future preds

                imgs = imgs.to(device)
                prop_t = prop_t.to(device)
                act_t = act_t.to(device)

                B, T_obs = imgs.shape[:2]

                # Extract target actions (the pred_horizon window after the observation window)
                # act_t includes the obs window, so we slice off the end
                target_actions = act_t[:, -cfg.pred_horizon:, :]

                optimizer.zero_grad()

                # 1. Encode Vision. For multi-view, encode each view and fuse by mean to
                # keep context dimension compatible with existing checkpoints/config.
                if imgs.ndim == 5:
                    imgs_flat = imgs.reshape(B * T_obs, *imgs.shape[2:])
                    vision_feats_flat = vision_encoder(imgs_flat, return_spatial=False)
                    vision_feats = vision_feats_flat.view(B, T_obs, cfg.vision_dim)
                elif imgs.ndim == 6:
                    n_views = imgs.shape[2]
                    imgs_flat = imgs.reshape(B * T_obs * n_views, *imgs.shape[3:])
                    vision_feats_flat = vision_encoder(imgs_flat, return_spatial=False)
                    vision_feats = vision_feats_flat.view(B, T_obs, n_views, cfg.vision_dim).mean(dim=2)
                else:
                    raise ValueError(f"Unexpected image tensor rank: {imgs.ndim}")

                # 2. Prepare context latents
                ctx = torch.cat([vision_feats, prop_t], dim=-1)
                ctx_latents = policy.context_norm(ctx)

                # 3. Predict Actions
                summary = policy.encode_summary(ctx_latents)
                pred_actions = policy.predict_actions_from_summary(summary)  # (B, pred_horizon, 7)

                # 4. Compute Loss
                loss_bc = criterion(pred_actions, target_actions)
                loss_cf = torch.zeros((), device=device)
                loss_neg = torch.zeros((), device=device)
                if args.use_cime:
                    cf_ctx_latents = _make_counterfactual_context(ctx_latents, mask_ratio=args.cime_cf_mask_ratio)
                    cf_summary = policy.encode_summary(cf_ctx_latents)
                    neg_actions = _make_negative_actions(target_actions, noise_std=args.cime_neg_noise_std)

                    e_pos = policy.energy_from_summary(summary, target_actions)
                    e_cf = policy.energy_from_summary(cf_summary, target_actions)
                    e_neg = policy.energy_from_summary(summary, neg_actions)

                    # Counterfactual margin: E(pos) + m <= E(counterfactual, pos_action)
                    loss_cf = F.relu(args.cime_margin + e_pos - e_cf).mean()
                    # Negative-action margin: E(pos) + m <= E(context, neg_action)
                    loss_neg = F.relu(args.cime_margin + e_pos - e_neg).mean()

                loss = loss_bc + args.cime_lambda_cf * loss_cf + args.cime_lambda_neg * loss_neg

                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()
                train_bc_loss += loss_bc.item()
                train_cf_loss += float(loss_cf.detach().cpu())
                train_neg_loss += float(loss_neg.detach().cpu())

            train_loss /= len(train_dl)
            train_bc_loss /= len(train_dl)
            train_cf_loss /= len(train_dl)
            train_neg_loss /= len(train_dl)

            # Validation
            policy.eval()
            vision_encoder.eval()
            val_loss = 0.0
            val_bc_loss = 0.0
            val_cf_loss = 0.0
            val_neg_loss = 0.0

            with torch.no_grad():
                for imgs, prop_t, act_t in val_dl:
                    imgs = imgs.to(device)
                    prop_t = prop_t.to(device)
                    act_t = act_t.to(device)

                    B, T_obs = imgs.shape[:2]
                    target_actions = act_t[:, -cfg.pred_horizon:, :]

                    if imgs.ndim == 5:
                        imgs_flat = imgs.reshape(B * T_obs, *imgs.shape[2:])
                        vision_feats_flat = vision_encoder(imgs_flat, return_spatial=False)
                        vision_feats = vision_feats_flat.view(B, T_obs, cfg.vision_dim)
                    elif imgs.ndim == 6:
                        n_views = imgs.shape[2]
                        imgs_flat = imgs.reshape(B * T_obs * n_views, *imgs.shape[3:])
                        vision_feats_flat = vision_encoder(imgs_flat, return_spatial=False)
                        vision_feats = vision_feats_flat.view(B, T_obs, n_views, cfg.vision_dim).mean(dim=2)
                    else:
                        raise ValueError(f"Unexpected image tensor rank: {imgs.ndim}")

                    ctx = torch.cat([vision_feats, prop_t], dim=-1)
                    ctx_latents = policy.context_norm(ctx)

                    summary = policy.encode_summary(ctx_latents)
                    pred_actions = policy.predict_actions_from_summary(summary)
                    loss_bc = criterion(pred_actions, target_actions)
                    loss_cf = torch.zeros((), device=device)
                    loss_neg = torch.zeros((), device=device)
                    if args.use_cime:
                        cf_ctx_latents = _make_counterfactual_context(ctx_latents, mask_ratio=args.cime_cf_mask_ratio)
                        cf_summary = policy.encode_summary(cf_ctx_latents)
                        neg_actions = _make_negative_actions(target_actions, noise_std=args.cime_neg_noise_std)

                        e_pos = policy.energy_from_summary(summary, target_actions)
                        e_cf = policy.energy_from_summary(cf_summary, target_actions)
                        e_neg = policy.energy_from_summary(summary, neg_actions)
                        loss_cf = F.relu(args.cime_margin + e_pos - e_cf).mean()
                        loss_neg = F.relu(args.cime_margin + e_pos - e_neg).mean()

                    loss = loss_bc + args.cime_lambda_cf * loss_cf + args.cime_lambda_neg * loss_neg
                    val_loss += loss.item()
                    val_bc_loss += loss_bc.item()
                    val_cf_loss += float(loss_cf.detach().cpu())
                    val_neg_loss += float(loss_neg.detach().cpu())

            val_loss /= len(val_dl)
            val_bc_loss /= len(val_dl)
            val_cf_loss /= len(val_dl)
            val_neg_loss /= len(val_dl)

            msg = (
                f"Epoch [{epoch+1}/{args.epochs}] Time: {time.time()-start_t:.1f}s | "
                f"Train Loss: {train_loss:.4f} (BC {train_bc_loss:.4f}"
            )
            if args.use_cime:
                msg += f", CF {train_cf_loss:.4f}, NEG {train_neg_loss:.4f}"
            msg += f") | Val Loss: {val_loss:.4f} (BC {val_bc_loss:.4f}"
            if args.use_cime:
                msg += f", CF {val_cf_loss:.4f}, NEG {val_neg_loss:.4f}"
            msg += ")"
            print(msg)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = run_dir / "best_model.pt"
                torch.save(
                    _make_checkpoint_payload(
                        epoch=epoch,
                        policy=policy,
                        vision_encoder=vision_encoder,
                        stats=stats,
                        cfg=cfg,
                        val_loss=best_val_loss,
                        cime=cime_cfg,
                    ),
                    ckpt_path,
                )
                print(f"  --> Saved new best model to {ckpt_path}")

            rollout_eval.on_epoch_end(
                epoch_1b=epoch + 1,
                checkpoint_payload=_make_checkpoint_payload(
                    epoch=epoch,
                    policy=policy,
                    vision_encoder=vision_encoder,
                    stats=stats,
                    cfg=cfg,
                    val_loss=val_loss,
                    cime=cime_cfg,
                ),
            )
    finally:
        rollout_eval.finalize()

if __name__ == "__main__":
    main()
