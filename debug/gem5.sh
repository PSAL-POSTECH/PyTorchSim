#!/bin/bash

path="$1"

# if $TORCHSIM_DIR/debug does not exist, create it
if [ ! -d "$TORCHSIM_DIR/debug" ]; then
  mkdir $TORCHSIM_DIR/debug
fi
if [ ! -d "$TORCHSIM_DIR/debug/out" ]; then
  mkdir $TORCHSIM_DIR/debug/out
fi

/workspace/gem5/build/RISCV/gem5.debug \
--debug-flags=Fetch,Decode,MinorExecute \
-d m5out $TORCHSIM_DIR/gem5_script/script_systolic.py \
-c $path/cycle_bin --vlane 128 > $TORCHSIM_DIR/debug/out/gem5_log.txt

# grep ticks of M5Op
ticks=($(grep "Changing stream on" $TORCHSIM_DIR/debug/out/gem5_log.txt | grep "M5Op" | awk '{print $1}'))

# trim only cycle number
for i in "${!ticks[@]}"; do
  ticks[$i]=${ticks[$i]::-4}
done

# extract instruction
python $TORCHSIM_DIR/debug/pipeline.py --input $TORCHSIM_DIR/debug/out/gem5_log.txt --min ${ticks[-2]} --max ${ticks[-1]} > $TORCHSIM_DIR/debug/out/gem5_inst.txt