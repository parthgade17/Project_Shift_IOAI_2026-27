"""
Project SHIFT: Author Benchmark Generation & Packaging Script.
IOAI 2027 Scientific Committee.
"""

import sys
import shutil
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from scripts.generate_datasets import main as run_generation


def package_author_private_data():
    datasets_dir = parent_dir / "datasets"
    private_dir = script_dir / "private_data"
    private_dir.mkdir(parents=True, exist_ok=True)

    for fname in ["hidden_test_labels.json", "hidden_test_full_reference.jsonl"]:
        src = datasets_dir / fname
        if src.exists():
            shutil.copyfile(src, private_dir / fname)
            print(f"  [OK] Synchronized {fname} to author_package/private_data/")


if __name__ == "__main__":
    print("Generating and certifying Project SHIFT benchmark datasets...")
    run_generation()
    package_author_private_data()
    print("Author benchmark packaging complete.")
