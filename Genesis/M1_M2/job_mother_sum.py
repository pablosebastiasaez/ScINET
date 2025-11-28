#!/usr/bin/env python
#%%
import os
import numpy as np

os.system(f'mkdir sum_scatter')
os.system(f'cp M2_sum.py sum_scatter/M2_sum.py')
os.system(f'cp job_compiled_sum.sh sum_scatter/job_compiled_sum.sh')
os.system(f'cp setup_scatter.py sum_scatter/setup_scatter.py')
os.system(f'cp scatter_cython.pyx sum_scatter/scatter_cython.pyx')
# Change to the created folder
os.chdir(f'sum_scatter')
os.system('sbatch ./job_compiled_sum.sh')
# change back to the parent directory
os.chdir('..')
#%%
