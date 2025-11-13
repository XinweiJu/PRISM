import os

def generate_sequence_txt(base_path):
    # Define dataset splits
    splits = {
        "train": [
            ("cecum", 1, "a", 276), ("cecum", 1, "b", 765), ("cecum", 2, "a", 370), 
            ("cecum", 2, "b", 1142), ("cecum", 2, "c", 595), ("cecum", 3, "a", 730),
            ("sigmoid", 1, "a", 700), ("sigmoid", 2, "a", 514),
            ("trans", 1, "a", 61), ("trans", 1, "b", 700), ("trans", 2, "a", 194), 
            ("trans", 2, "b", 103), ("trans", 2, "c", 235), ("trans", 3, "a", 250), ("trans", 3, "b", 214)
        ],
        "val": [
            ("cecum", 4, "a", 465), ("sigmoid", 3, "a", 613), ("trans", 4, "a", 382)
        ],
        "test": [
            ("cecum", 4, "b", 425), ("desc", 4, "a", 148), ("sigmoid", 3, "b", 536), ("trans", 4, "b", 597)
        ]
    }
    
    for split, sequences in splits.items():
        output_path = os.path.join(base_path, f"{split}_files.txt")
        with open(output_path, "w") as f:
            for model, texture, video, frames in sequences:
                sequence_name = f"{model}_t{texture}_{video}"
                for idx in range(30, frames-30):
                    f.write(f"/Datasets/C3VD_Undistorted/Dataset/{sequence_name}/{str(idx).zfill(4)}_color.png\n")

# Define base output path
base_output_path = os.path.dirname(os.path.realpath(__file__))
os.makedirs(base_output_path, exist_ok=True)
generate_sequence_txt(base_output_path)

print(f"Files have been generated in '{base_output_path}'.")
