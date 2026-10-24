"""What runs a kernel object, in two paths each a package, one file per tool it drives:

    functional/    the values: tensors -> runtime/*.raw -> Spike (spike.py) -> tensors
    timing/        the cycles: gem5 over sampled tiles (gem5.py) -> tables -> TOGSim (togsim.py)
    wallclock.py   where a run's own wall clock went, across both
"""
