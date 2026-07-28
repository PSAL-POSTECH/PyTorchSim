import re
import sys

def time_to_milliseconds(timestamp):
    match = re.match(r"\[(\d{4}-\d{2}-\d{2}) (\d{2}):(\d{2}):(\d{2})\.(\d{3})\]", timestamp)
    if not match:
        return None

    _, hh, mm, ss, ms = match.groups()

    total_ms = (int(hh) * 3600 + int(mm) * 60 + int(ss)) * 1000 + int(ms)
    return total_ms

def parse_log_file(file_path):
    with open(file_path, "r") as file:
        for line in file:
            time_match = re.search(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\]", line)
            # Cycle 값 추출 (예: Total cycle 43858000)
            cycle_match = re.search(r"\[0\] : Total cycle (\d+)", line)
            
            if time_match and cycle_match:
                timestamp = time_match.group(0)  # "[YYYY-MM-DD HH:MM:SS.sss]" 형식
                cycle = cycle_match.group(1)  # Cycle 값
                
                time_ms = time_to_milliseconds(timestamp)
                print(time_ms, cycle)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Wrong input")
        sys.exit(1)
    
    log_file = sys.argv[1]
    parse_log_file(log_file)

