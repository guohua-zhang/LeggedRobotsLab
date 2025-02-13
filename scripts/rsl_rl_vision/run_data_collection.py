import os
import gzip
import json
import argparse
import subprocess


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--r2r_data_path", type=str, default="/home/zhaojing/h1-training-isaaclab/assets/vln-ce/R2R_VLNCE_v1-3_preprocessed/train/train_filtered.json.gz")
    parser.add_argument("--task", type=str, default="go2_matterport_dataset")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    
    # Define the base arguments for data collection
    collection_args = [
        f"--task={args.task}",
        "--headless",
        "--enable_cameras"
    ]

    with gzip.open(args.r2r_data_path, 'rt') as file:
        data = json.load(file)
    episodes = data['episodes']

    start_idx = 0
    if args.resume:
        # Get already collected episodes from the data directory
        data_dir = "collected_data"
        if os.path.exists(data_dir):
            data_files = os.listdir(data_dir)
            data_files = [f for f in data_files if f.endswith(".npy")]
            if data_files:
                episode_nums = [int(f.split("_")[1].split(".")[0]) for f in data_files]
                start_idx = max(episode_nums) + 1
                print(f"Resuming from episode {start_idx}.....")

    for i in range(start_idx, len(episodes)):
        episode_id = episodes[i]['episode_id']

        # check if the episode has been collected
        if os.path.exists(f"collected_data/episode_{episode_id}.npy"):
            print(f"Episode {episode_id} already collected. Skipping....")
            continue
        
        # msg = f"\n======================= Collecting Data for Episode {episode_id} ======================="
        # # msg += f"\nScene: {episode['scene_id']}"
        # # msg += f"\nStart Position: {episode['start_position']}"
        # # msg += f"\nInstruction: {episode['instruction']['instruction_text']}\n"
        # print(msg)
        
        # Create episode-specific arguments
        episode_args = collection_args + [f"--episode_index={i}"]
        
        # Run the data collection script
        subprocess.run(['python', 'collect_data_matterport.py'] + episode_args)
        
        # Optional: Add a small delay between episodes
        # time.sleep(1) 