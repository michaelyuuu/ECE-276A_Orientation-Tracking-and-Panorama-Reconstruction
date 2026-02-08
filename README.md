# ECE276A PROJECT 1: ORIENTATION TRACKING AND PANORAMA STITCHING

## OVERVIEW
This project implements a sensor fusion algorithm to estimate the 3D orientation (Roll, Pitch, Yaw) of a sensing platform using IMU data. It subsequently uses these orientation estimates to stitch a sequence of RGB images into a seamless panorama.

The core algorithm formulates orientation estimation as an optimization problem on the quaternion manifold (Lie Group). It fuses:
1. **Gyroscope Integration:** For the motion model.
2. **Accelerometer Gravity Vector:** For the observation model (correction).

The optimization is solved via gradient descent using PyTorch for automatic differentiation.

## DEPENDENCIES
Ensure you have the following Python libraries installed:
* `numpy`
* `matplotlib`
* `transforms3d`
* `torch` (PyTorch)

You can install them via pip:
pip install numpy matplotlib transforms3d torch

## FILE STRUCTURE & DATA SETUP
The code expects the data directory to be located **one level above** the script directory (due to `../data` paths in the code).

Please ensure your project directory is organized as follows:

Project_Root/
├── data/                   <-- Data Folder
│   └── trainset/
│       ├── cam/            <-- Contains cam1.p, cam2.p...
│       ├── imu/            <-- Contains imuRaw1.p, imuRaw2.p...
│       └── vicon/          <-- Contains viconRot1.p...
└── code/                   <-- Source Code Folder (Current Directory)
    ├── load_data.py
    ├── main.py
    └── README.md

* **`load_data.py`**: Handles loading `.p` (pickle) files. It looks for data in `../data/trainset/`.
* **`main.py`**: The entry point for the project. It performs calibration, optimization, visualization, and stitching.

## HOW TO RUN

1. **Select the Dataset:**
   Open `load_data.py` and modify the `dataset` variable to choose the dataset number you wish to process (e.g., "1", "2", "10", etc.):

   # In load_data.py
   dataset = "1" 

2. **Execute the Pipeline:**
   Run the main script from the terminal:

   python main.py

## OUTPUTS
The script generates the following visualizations:
1. **Orientation Plots:** Comparison of Estimated Orientation vs. Ground Truth (Roll/Pitch/Yaw).
2. **Panorama Image:** A 2D Equirectangular projection stitched from the camera feed.

> **Note:** Check the bottom of `main.py` to toggle between saving files (`plt.savefig`) or showing them interactively (`plt.show`).

## CONTACT
For any questions or issues, please contact: chy084@ucsd.edu