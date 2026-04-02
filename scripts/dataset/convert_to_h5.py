import os
import pickle
import h5py
import numpy as np
import argparse
from tqdm import tqdm

def convert_pkl_to_h5(env_id, log_dir="datasets"):
    """
    Convert collected .pkl trajectories into a flat HDF5 file compatible with stable-worldmodel HDF5Dataset.
    It expects datasets at the root with names like 'observation', 'action', etc.
    and an 'offset' dataset indicating the start of each episode.
    """
    input_dir = os.path.join(log_dir, env_id, "datasets")
    output_file = os.path.join(log_dir, env_id, f"{env_id}_stable_wm.h5")

    if not os.path.exists(input_dir):
        print(f"Directory not found: {input_dir}")
        return

    pkl_files = [f for f in os.listdir(input_dir) if f.endswith(".pkl")]
    pkl_files.sort()

    all_trajs = []
    for f in tqdm(pkl_files, desc="Loading pkl files"):
        with open(os.path.join(input_dir, f), "rb") as df:
            data = pickle.load(df)
            all_trajs.extend(data.get("trajectories", []))

    if not all_trajs:
        print("No trajectories found.")
        return

    # stable-worldmodel expects flat datasets
    print("Concatenating data...")
    all_obs = np.concatenate([t["observations"] for t in all_trajs], axis=0).astype(np.float32)
    all_acts = np.concatenate([t["actions"] for t in all_trajs], axis=0).astype(np.float32)
    all_rewards = np.concatenate([t["rewards"] for t in all_trajs], axis=0).astype(np.float32)
    all_terminals = np.concatenate([t["terminals"] for t in all_trajs], axis=0).astype(np.bool_)
    all_next_obs = np.concatenate([t["next_observations"] for t in all_trajs], axis=0).astype(np.float32)
    
    # Success is often expected, even if all false
    all_success = np.zeros(len(all_terminals), dtype=np.bool_)
    
    # Calculate offsets
    lengths = [len(t["observations"]) for t in all_trajs]
    offsets = np.cumsum([0] + lengths[:-1]).astype(np.int64)

    print(f"Writing to {output_file}...")
    with h5py.File(output_file, "w") as h5f:
        h5f.create_dataset("observation", data=all_obs)
        h5f.create_dataset("action", data=all_acts)
        h5f.create_dataset("reward", data=all_rewards)
        h5f.create_dataset("terminal", data=all_terminals)
        h5f.create_dataset("next_observation", data=all_next_obs)
        h5f.create_dataset("success", data=all_success)
        h5f.create_dataset("offset", data=offsets)

    print(f"Converted {len(all_trajs)} trajectories to {output_file}")
    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", type=str, default="mo-halfcheetah-v5")
    args = parser.parse_args()
    convert_pkl_to_h5(args.env)
