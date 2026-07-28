import argparse
from pathlib import Path
import os

def list_nested_folders(root_path):
    root = Path(root_path)

    if not root.exists() or not root.is_dir():
        print(f"[Error] '{root}' is not a valid directory.")
        return []

    folders = set()
    for p in root.rglob('*'):
        if p.is_dir():
            rel_depth = len(p.relative_to(root).parts)
            if rel_depth == 3:
                folders.add(p)

    return sorted(folders)

def main():
    parser = argparse.ArgumentParser(description="List folders up to depth 3 and parse arguments.")

    parser.add_argument("path", type=str, help="Root directory to start scanning")
    parser.add_argument("--index", type=int, default=0, help="Index value (default: 0)")
    parser.add_argument("--attr", nargs='*', default=["best", "worst"],
                        help='List of attr (default: ["best", "worst"])')

    args = parser.parse_args()
    folders = list_nested_folders(args.path)
    for folder in folders:
        cmd = f"./chiplet.sh {folder} {args.index} {' '.join(args.attr)}"
        print(cmd)
        os.system(cmd)

if __name__ == "__main__":
    main()