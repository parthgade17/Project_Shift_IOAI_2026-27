"""
Project SHIFT: Author Shortcut Audit Suite Runner.
IOAI 2027 Scientific Committee.
"""

import sys
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from shift.serialize import load_episodes_jsonl
from shift.shortcut_audits import ShortcutAuditor


def run_author_audits():
    test_path = parent_dir / "datasets" / "hidden_test_full_reference.jsonl"
    cf_path = parent_dir / "datasets" / "test_counterfactual.jsonl"

    if not test_path.exists():
        print(f"Error: dataset file not found at {test_path}")
        return

    episodes = load_episodes_jsonl(test_path)
    if cf_path.exists():
        episodes.extend(load_episodes_jsonl(cf_path))

    print(f"Running full shortcut audit suite on N={len(episodes)} episodes...")
    results = ShortcutAuditor.run_full_shortcut_suite(episodes)

    print("\nShortcut Audit Results:")
    for k, v in results.items():
        print(f"  {k}: {v}")

    if results.get("all_shortcut_audits_passed"):
        print("\n>>> ALL ADVERSARIAL AUDITS PASSED <<<")
    else:
        print("\n>>> VULNERABILITY DETECTED IN BENCHMARK <<<")


if __name__ == "__main__":
    run_author_audits()
