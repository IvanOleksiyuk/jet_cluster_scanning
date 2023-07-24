import os
import subprocess

directory = 'plots/pub_plot/'

os.makedirs(directory, exist_ok=True)

steps = ['data', 'agorithm1', 'agorithm2', 'main']

#data plots
subprocess.run("python preproc/data_plotting.py config/pub_plot_path.yaml", check=True)

#Run one CS experiment with 5K signal events and plot all the results sving them properly
subprocess.run("python preproc/data_plotting.py config/pub_plot_path.yaml", check=True)

#Create the distribution plots with vertical lines for the cuts

#Create the ROC, SF and SI plots

#Create all the main plots 
