"""
Parse DFL tracking XML files and save player trajectories as numpy arrays.
Each trajectory is a (T, 2) array of (x, y) positions at 0.04s intervals.
"""
import os
import glob
import yaml
import numpy as np
import xml.etree.ElementTree as ET

MIN_FRAMES = 100  # discard very short trajectories


def parse_position_file(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    trajectories = []
    for fs in root.findall(".//FrameSet"):
        frames = fs.findall("Frame")
        if len(frames) < MIN_FRAMES:
            continue
        xy = np.array([[float(f.get("X")), float(f.get("Y"))] for f in frames], dtype=np.float32)
        trajectories.append(xy)
    return trajectories


def load_all(raw_dir):
    pattern = os.path.join(raw_dir, "DFL_04_03_positions_raw_observed_*.xml")
    files = sorted(glob.glob(pattern))
    print(f"Found {len(files)} position files")
    all_traj = []
    for f in files:
        traj = parse_position_file(f)
        all_traj.extend(traj)
        print(f"  {os.path.basename(f)}: {len(traj)} trajectories")
    return all_traj


def build_samples(trajectories, window):
    """
    For each trajectory, create (input, target) pairs.
    - Retained frames: even indices (0, 2, 4, ...)
    - Missing frames : odd  indices (1, 3, 5, ...) → targets

    For each missing frame at original index 2k+1:
      input  : window retained frames centered on the gap
               [r_{k - W//2 + 1}, ..., r_k, r_{k+1}, ..., r_{k + W//2}]
      target : p_{2k+1}  (ground truth)
    """
    W = window
    half = W // 2
    inputs, targets = [], []

    for xy in trajectories:
        retained = xy[0::2]   # shape (M, 2)  where M = ceil(T/2)
        missing  = xy[1::2]   # shape (K, 2)  K = floor(T/2)

        for k in range(len(missing)):
            lo = k - half + 1
            hi = k + half + 1       # exclusive; retained[lo:hi] covers W frames
            if lo < 0 or hi > len(retained):
                continue
            ctx = retained[lo:hi]   # (W, 2)
            inputs.append(ctx)
            targets.append(missing[k])

    return np.array(inputs, dtype=np.float32), np.array(targets, dtype=np.float32)


if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(os.path.dirname(cfg["data"]["processed_file"]), exist_ok=True)

    traj = load_all(cfg["data"]["raw_dir"])
    print(f"\nTotal trajectories: {len(traj)}")

    X, y = build_samples(traj, cfg["model"]["window"])
    print(f"Samples  — inputs: {X.shape}, targets: {y.shape}")

    np.save(cfg["data"]["processed_file"], {"X": X, "y": y})
    print(f"Saved → {cfg['data']['processed_file']}")
