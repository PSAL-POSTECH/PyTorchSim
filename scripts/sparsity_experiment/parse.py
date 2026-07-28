import argparse
import os
import subprocess

def get_stored_paths(log_file):
    """Extracts stored file paths from the given log file."""
    stored_paths = []
    try:
        result = subprocess.run(["grep", "stored", log_file], capture_output=True, text=True)
        for line in result.stdout.splitlines():
            parts = line.split(" ")
            if "stored" in parts:
                index = parts.index("stored")
                if index + 1 < len(parts):
                    stored_paths.append(parts[index + 2].strip('"'))
    except Exception as e:
        print(f"Error reading stored paths: {e}")
    return stored_paths

def get_last_total_cycle(file_path):
    """Extracts the last Total cycle value from the given file."""
    total_cycle = None
    try:
        result = subprocess.run(["grep", "Total cycle", file_path], capture_output=True, text=True)
        lines = result.stdout.splitlines()
        if lines:
            last_line = lines[-1]
            total_cycle = last_line.split()[-1]  # Extract the last value
    except Exception as e:
        print(f"Error reading total cycle from {file_path}: {e}")
    return total_cycle

def main(log_file):
    stored_paths = get_stored_paths(log_file)
    k = []
    for path in stored_paths:
        print(path)
        if os.path.exists(path):
            total_cycle = get_last_total_cycle(path)
            if total_cycle:
                k.append(total_cycle)
            else:
                print(f"{path}: No Total cycle found")
        else:
            print(f"{path}: File does not exist")
    return k

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Total Cycle from stored paths.")
    parser.add_argument("log_file", type=str, help="Path to the log file containing stored paths")
    args = parser.parse_args()
    a_l = []
    b_l = []
    if os.path.exists(args.log_file):
        a, b = main(args.log_file + "/0.0")
        a_l.append(a)
        b_l.append(b)
        a, b = main(args.log_file + "/0.2")
        a_l.append(a)
        b_l.append(b)
        a, b = main(args.log_file + "/0.4")
        a_l.append(a)
        b_l.append(b)
        a, b = main(args.log_file + "/0.6")
        a_l.append(a)
        b_l.append(b)
        a, b = main(args.log_file + "/0.8")
        a_l.append(a)
        b_l.append(b)
        print(" ".join(a_l))
        print(" ".join(b_l))
 
    else:
        print(f"Log file {args.log_file} not found.")
