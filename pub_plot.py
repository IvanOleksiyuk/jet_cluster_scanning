import os
import subprocess

directory = 'plots/pub_plot/'

os.makedirs(directory, exist_ok=True)

#data plots
subprocess.run("python preproc/data_plotting.py", check=True)