import re
import sys

def parse_log_file(file_path, interval):
    with open(file_path, "r") as file:
        index = 0
        for line in file:
            if index % interval != 0:
                index += 1
                continue
            index += 1
            print(line.strip())

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Wrong input")
        sys.exit(1)
    
    log_file = sys.argv[1]
    interval = int(sys.argv[2])
    parse_log_file(log_file, interval)

