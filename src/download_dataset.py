from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


DATASET_SLUG = "grassknoted/asl-alphabet"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Kaggle ASL Alphabet dataset.")
    parser.add_argument("--output", default="data", help="Directory where the dataset should be downloaded.")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        DATASET_SLUG,
        "-p",
        str(output_dir),
        "--unzip",
    ]
    subprocess.run(command, check=True)
    print(f"Dataset downloaded into: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
