import os.path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yaml

from mytools import *


result_folder = 'full_results'
with open('configuration.yaml', 'r') as f:
    configs = yaml.safe_load(f)
for file_name in os.listdir(result_folder):
    file_path = os.path.join(result_folder, file_name)
    nframes = 4000
    dt = 0.05
    result_data = pd.read_csv(file_path)
    ransac_error = result_data["ransac_error"]
    pca_error = result_data["pca_error"]
    pred_error = result_data["prediction_error"]
    xq_error_rad = result_data["estimate_diff"]
    xq_error = np.rad2deg(xq_error_rad)
    ransac_pca_diff = result_data["ransac_pca_diff"]

    tspan = np.arange(0, dt*nframes, dt)
    fig = plt.figure()
    plt.plot(tspan, ransac_error, label='RANSAC error', linewidth=1)
    plt.plot(tspan, pca_error, label='PCA error', linewidth=1)
    plt.plot(tspan, pred_error, label='Prediction error', linewidth=1)
    plt.plot(tspan, xq_error, label='Estimate error', linewidth=1)
    plt.ylim(0, 120)
    plt.ylabel("Angle (deg)")
    plt.legend()

    fig = plt.figure()
    plt.scatter(tspan[ransac_pca_diff < configs["ransac_pca_threshold"]], ransac_error[ransac_pca_diff < configs["ransac_pca_threshold"]],label="Used Measurement Error", marker=".")
    plt.scatter(tspan[ransac_pca_diff >= configs["ransac_pca_threshold"]], pred_error[ransac_pca_diff >= configs["ransac_pca_threshold"]],label="Used Prediction Error", marker=".")
    plt.ylim(0, 120)
    plt.ylabel("Angle (deg)")
    plt.legend()
    
plt.show()