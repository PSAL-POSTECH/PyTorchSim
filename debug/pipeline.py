import argparse
import os
args = argparse.ArgumentParser()
args.add_argument('--input', type=str, default='input.txt')
args.add_argument('--min', type=int, default=0)
args.add_argument('--max', type=int, default=0)

parsed_args = args.parse_args()

filename = parsed_args.input
min_value = parsed_args.min
max_value = parsed_args.max - 1
#max_addr = '0x10378' # address of the last M5Op

def filter_exec_in_file(filename, match_string):
    result = []
    prev_line = "prev line"
    
    with open(filename, 'r') as file:
        lines = file.readlines()

        cur_stream = 0
        index_dict = dict()
        
        for line in lines:
            parts = line.strip().split(' ')

            if len(parts) >= 3:
                try:
                    num = int(int(parts[0][:-1])/1000)
                    first_string = parts[1][:-1]

                    addr = "temp addr"
                    
                    if min_value <= num <= max_value and first_string == match_string:
                        idx = 0

#                        if parts[2] == "Pushing": # Pushing mem inst: %s pc: addr (inst)
#                            result.append(f"{num}|execute I: {parts[-2]} {parts[-1]} this is a memory ref. instruction.")
                        if parts[2] == "Issuing":
                            if parts[3] == "inst:": # Issuing inst: %s pc: addr (inst) into FU %d
                                stream = int(parts[-7].split('.')[0].split('/')[-1])
                                index = int(parts[-7].split('.')[-1])
                                addr = parts[-5]
                                inst = parts[-4]

#                                if addr > max_addr:
#                                    continue

                                if (cur_stream != stream):
                                    index_dict.clear()
                            
                                if addr in index_dict:
                                    idx = index - index_dict[addr]
                                else:
                                    index_dict[addr] = index

                                result.append(f"{num}|2_execute I: {addr} {inst} {idx}")

                                cur_stream = stream
                            elif parts[3] == "mem": # Issuing mem ref early inst: %s pc: addr (inst) instToWaitFor: %d
                                stream = int(parts[-6].split('.')[0].split('/')[-1])
                                index = int(parts[-6].split('.')[-1])
                                addr = parts[-4]
                                inst = parts[-3]

#                                if addr > max_addr:
#                                    continue

                                if (cur_stream != stream):
                                    index_dict.clear()

                                if addr in index_dict:
                                    idx = index - index_dict[addr]
                                else:
                                    index_dict[addr] = index

                                result.append(f"{num}|1_execute M: {addr} {inst} {idx}") 

                                cur_stream = stream
                            else : # Issuing %s to %d  # Prev : Trying to issue inst: %s pc: addr (inst) to FU %d
                                prev_parts = prev_line.strip().split(' ')

                                stream = int(prev_parts[-7].split('.')[0].split('/')[-1])
                                index = int(prev_parts[-7].split('.')[-1])
                                addr = prev_parts[-5]
                                inst = prev_parts[-4]

#                                if addr > max_addr:
#                                    continue

                                if (cur_stream != stream):
                                    index_dict.clear()

                                if addr in index_dict:
                                    idx = index - index_dict[addr]
                                else:
                                    index_dict[addr] = index

                                result.append(f"{num}|2_execute I: {addr} {inst} {idx}")

                                cur_stream = stream                                
#                        elif parts[2] == "Discarding": # Discarding inst: %s pc: addr (inst) as its stream state was unexpected, expected: %d
#                            result.append(f"")
                        elif parts[2] == "Completed": # Completed inst: %s pc: addr (inst)
                            stream = int(parts[-4].split('.')[0].split('/')[-1])
                            index = int(parts[-4].split('.')[-1])
                            addr = parts[-2]
                            inst = parts[-1]

#                            if addr > max_addr:
#                                continue

                            if (cur_stream != stream):
                                continue

                            if addr in index_dict:
                                idx = index - index_dict[addr]
                            else:
                                index_dict[addr] = index

                            result.append(f"{num}|0_execute C: {addr} {inst} {idx}")

                            cur_stream = stream
                        else:
                            prev_line = line
                            continue
                except ValueError:
                    continue
            prev_line = line
                    
    return result

def filter_decode_in_file(filename, match_string):
    result = []
    
    with open(filename, 'r') as file:
        lines = file.readlines()
        
        for line in lines:
            parts = line.strip().split(' ')
            
            if len(parts) >= 3:
                try:
                    num = int(int(parts[0][:-1])/1000)
                    first_string = parts[1][:-1]
                    
                    if min_value <= num <= max_value and first_string == match_string:
                        if parts[2] == "Microop":
                            inst = parts[-1]
                            addr = parts[-2]

                            if inst == '(vnop)':
                                continue

#                            if addr > max_addr:
#                                continue

                            result.append(f"{num}|3_decode: {parts[-2]} {parts[-1]} {parts[-6][-5]}")                            
                        elif parts[2] == "Passing":
                            addr = parts[7]

#                            if addr > max_addr:
#                                continue
                            
                            result.append(f"{num}|3_decode: {parts[7]} {parts[8]}")
                        else:
                            continue
                except ValueError:
                    continue
                    
    return result

def filter_fetch2_in_file(filename, match_string):
    result = []
    
    with open(filename, 'r') as file:
        lines = file.readlines()
        
        for line in lines:
            parts = line.strip().split(' ')
            
            if len(parts) >= 3:
                try:
                    num = int(int(parts[0][:-1])/1000)
                    first_string = parts[1][:-1]
                    
                    if min_value <= num <= max_value and first_string == match_string:
                        if parts[2] == "Instruction": # Instruction extracted from line ~
                            addr = parts[-2]

#                            if addr > max_addr:
#                                continue                            

                            result.append(f"{num}|4_fetch2: {parts[-2]} {parts[-1]}")
                        else:
                            continue
                except ValueError:
                    continue
                    
    return result

def filter_fetch1_in_file(filename, match_string):
    result = []
    
    with open(filename, 'r') as file:
        lines = file.readlines()
        
        temp = "start_addr"
        
        for line in lines:
            parts = line.strip().split(' ')
            
            if len(parts) >= 3:
                try:
                    num = int(int(parts[0][:-1])/1000)
                    first_string = parts[1][:-1]
                    
                    if min_value <= num <= max_value and first_string == match_string:
                        if parts[2] == "Inserting":
                            addr = parts[-7]
                            temp = addr

#                            if addr > max_addr:
#                                continue                            

                            result.append(f"{num}|5_fetch1: {addr} ~ ")
                        elif parts[2] == "Processing":
#                            if temp > max_addr:
#                                continue

                            result.append(f"{num}|5_fetch1: {temp} ~ ")
                        else:
                            continue
                except ValueError:
                    continue
                    
    return result



filtered_fetch1 = filter_fetch1_in_file(filename, 'system.cpu.fetch1')
filtered_fetch2 = filter_fetch2_in_file(filename, 'system.cpu.fetch2')
filtered_decode = filter_decode_in_file(filename, 'system.cpu.decode')
filtered_exec = filter_exec_in_file(filename, 'system.cpu.execute')

for line in filtered_exec:
    print(line)

for line in filtered_decode:
    print(line)

for line in filtered_fetch2:
    print(line)

for line in filtered_fetch1:
    print(line)
