#!/usr/bin/env python
#%%
import os
import numpy as np

os.system(f'mkdir difference_scatter')
os.system(f'cp M2_difference.py difference_scatter/M2_difference.py')
os.system(f'cp job_compiled_difference.sh difference_scatter/job_compiled_difference.sh')
os.system(f'cp setup_scatter.py difference_scatter/setup_scatter.py')
os.system(f'cp scatter_cython.pyx difference_scatter/scatter_cython.pyx')
# Change to the created folder
os.chdir(f'difference_scatter')
os.system('sbatch ./job_compiled_difference.sh')
# change back to the parent directory
os.chdir('..')
#%%
