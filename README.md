# TEAM04 | Miro BioRescue

## IMPORTANT INFORMATION

### *This project requires two diamond laptops and two Miro-E robots to run. Before attempting to run this project, ensure you have two diamond laptops.*

## Overview

This project aims to achieve a biomimetic interaction between two Miro robots. It involves the use of a lost Miro and helper Miro, in which the lost Miro is blind, and the helper Miro must navigate an environment to find the lost miro, and then guide it back to the starting location. To achieve this, many methods have been used such as: Sonar for mapping and navigation; ONNX for object (Miro) recognition; instructions via a frequency tone, involving a bandpass on the listening robot to prevent noise and ascertain the correct instruction; and the A-star search algorithm for pathfinding.  

## Installation

### Running in Docker

Docker is a great way to run an application with all dependencies and libraries bundles together. 
Make sure to [install Docker](https://docs.docker.com/get-docker/) first. 

## Usage

1. Get two diamond laptops, and two Miro-E robots, and run the supplied docker container image on both machines (`miroprojects/com3528-2025-26:team04`).

2. Navigate to `root/mdk/catkin_ws/src/team04`.

3. Connect to diamond-lab wifi on both laptops.

4. On one laptop connect to one of the miro bots with these instructions.

        > miro mode robot
        > miro ip update

5. Enter the IP of one of the miro bots you have.

6. Now, use the other laptop to connect to the other miro, following the same instructions (4-5).

7. Make sure to run `source ~/.bashrc` on both laptops to resource their environment.

8. Now, on one miro, run `python3 /scripts/Helper.py`

9. Finally, on the other miro, run `python3 scripts/Lost.py`

10. They will now begin the project demo.

## `miro-docker` Changes

### Required Libraries

- **Numpy** - Basic functions such as np.arrays 

- **Scipy** - Used for bandpass, functions such as butter and fourier transforms

- **OpenCV** - Used for formatting images and grids for the neural network 

- **ONNXruntime** - More compact framework than YOLO for training the object detection model

- **matplotlib** - Used for graphing various outputs from functions, such as the bandpass and miro video output

### Packages

- `team04` package - includes Lost.py and Helper.py, aswell as other scripts for testing purposes 

## Notes

### *If the uploaded docker image has failed to work, use the original miro-docker container and instead clone the `https://github.com/jbrice2-source/team04` repository into `root/mdk/catkin_ws/src/` and pip install the **Required Libraries**. Then follow the **Usage** instructions to test the project.*