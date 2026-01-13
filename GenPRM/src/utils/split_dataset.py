"""
Exports ALL splits of a Hugging Face dataset into individual example directories.
This allows multiple nodes to process in parallel

Example:
python utils/split_dataset.py \
    --dataset Qwen/ProcessBench \
    --split_dir _data/split_input/ProcessBench
"""

import argparse
import json
import os
from datasets import load_dataset, Dataset


def export_all_splits(dataset_name, root_output_dir):
    """Process all available splits in the dataset"""
    # Load full dataset dictionary
    dataset_dict = load_dataset(dataset_name)

    # Create root output directory
    os.makedirs(root_output_dir, exist_ok=True)

    # Process each split
    for split_name in dataset_dict.keys():
        process_split(
            dataset=dataset_dict[split_name],
            split_name=split_name,
            root_output_dir=root_output_dir
        )


def process_split(dataset, split_name, root_output_dir):
    """Handle individual split processing"""
    # Create split-specific parent directory
    split_dir = root_output_dir
    os.makedirs(split_dir, exist_ok=True)

    # Save split-level dataset info
    dataset.info.write_to_directory(split_dir)

    # Process individual examples
    for idx, example in enumerate(dataset):
        example_dir = os.path.join(
            split_dir,
            f"{split_name}_{idx:03d}"
        )
        os.makedirs(example_dir, exist_ok=True)
        with open(os.path.join(example_dir, "sample.json"), "w") as f:
            json.dump(example, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset name")
    parser.add_argument("--split_dir", required=True, help="output directory")
    args = parser.parse_args()

    export_all_splits(
        dataset_name=args.dataset,
        root_output_dir=args.split_dir
    )
