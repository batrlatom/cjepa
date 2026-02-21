import argparse
import pickle
from pathlib import Path

import torch
import numpy as np

from dataset_robomimic import get_robomimic_dataloaders
from model import CJEPA, CJEPAConfig
from vision import VisionBackboneWrapper

def run_inference(dataset_path: str, checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint {checkpoint_path}...")

    try:
        ckpt = torch.load(checkpoint_path, map_location=device)
    except pickle.UnpicklingError as exc:
        if "Weights only load failed" not in str(exc):
            raise
        # PyTorch 2.6+ defaults weights_only=True; legacy checkpoints with
        # non-tensor metadata need an explicit trusted load.
        print("Checkpoint includes non-tensor metadata. Retrying with weights_only=False...")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    cfg = CJEPAConfig(**ckpt['cfg'])
    
    policy = CJEPA(cfg).to(device)
    vision_encoder = VisionBackboneWrapper(pretrained=False).to(device)
    
    policy.load_state_dict(ckpt['policy_state'])
    vision_encoder.load_state_dict(ckpt['vision_state'])
    stats = ckpt['stats']
    
    policy.eval()
    vision_encoder.eval()

    # Load 1 batch of validation data just for inspection
    _, val_dl, stats_out = get_robomimic_dataloaders(
        dataset_paths=[dataset_path],
        batch_size=1,
        obs_horizon=cfg.obs_horizon,
        pred_horizon=cfg.pred_horizon,
        val_split=0.1
    )
    
    print("\n--- Running single inference step ---")
    with torch.no_grad():
        imgs, prop_t, act_t = next(iter(val_dl))
        imgs = imgs.to(device)
        prop_t = prop_t.to(device)
        
        # Target Actions (we strip off the observation history just like train)
        target_actions = act_t[:, -cfg.pred_horizon:, :].to(device)
        
        B, T_obs = imgs.shape[:2]
        if imgs.ndim == 5:
            imgs = imgs.squeeze(2)
        
        # Encode Context Latents
        imgs_flat = imgs.reshape(B * T_obs, *imgs.shape[2:])
        vision_feats_flat = vision_encoder(imgs_flat, return_spatial=False)
        vision_feats = vision_feats_flat.view(B, T_obs, cfg.vision_dim)
        
        ctx = torch.cat([vision_feats, prop_t], dim=-1)
        ctx_latents = policy.context_norm(ctx)
        
        # Predict 
        pred_actions = policy(ctx_latents) # Shape: (1, 16, 7)
        
        # Unnormalize actions back to physical gripper space using the Dataset StatsNormalizer
        from dataset_robomimic import StatsNormalizer
        normalizer = StatsNormalizer(stats, action_norm_type="minmax")
        
        pred_actions_numpy = pred_actions.cpu().numpy()[0]
        actual_actions_numpy = target_actions.cpu().numpy()[0]
        
        physical_pred = normalizer.unnormalize(pred_actions_numpy, "actions")
        physical_actual = normalizer.unnormalize(actual_actions_numpy, "actions")
        
        mse = np.mean((physical_pred - physical_actual)**2)
        
        print(f"Proprioception Shape : {prop_t.shape}")
        print(f"Visual Shape         : {imgs.shape}")
        print(f"Prediction Horizon   : {cfg.pred_horizon} steps @ dim={cfg.action_dim}")
        print(f"Output Action MSE    : {mse:.4f} (Physical Coordinates)")
        
        print("\n[Trajectory Step 0] - Actual vs Predicted:")
        print(f"  Target : {physical_actual[0].round(3)}")
        print(f"  Pred   : {physical_pred[0].round(3)}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str, help="Path to Robomimic HDF5 dataset")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to best_model.pt")
    args = parser.parse_args()
    
    run_inference(args.dataset, args.checkpoint)
