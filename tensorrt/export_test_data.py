import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Export synthetic Robomimic TensorRT test inputs")
    parser.add_argument("--output_file", type=str, default="robomimic_test_inputs.npz")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--obs_horizon", type=int, default=2)
    parser.add_argument("--num_cameras", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=84)
    parser.add_argument("--proprio_dim", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(int(args.seed))

    images = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(
            int(args.batch_size),
            int(args.obs_horizon),
            int(args.num_cameras),
            3,
            int(args.image_size),
            int(args.image_size),
        ),
    ).astype(np.float32)

    proprio = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(int(args.batch_size), int(args.obs_horizon), int(args.proprio_dim)),
    ).astype(np.float32)

    out_path = Path(args.output_file)
    np.savez(out_path, images=images, proprio=proprio)

    print(f"Saved synthetic inputs to {out_path}")
    print(f"images shape : {images.shape}")
    print(f"proprio shape: {proprio.shape}")


if __name__ == "__main__":
    main()
