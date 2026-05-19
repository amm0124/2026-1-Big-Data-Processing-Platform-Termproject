# 2026-1-Big-Data-Processing-Platform-Termproject

## Getting data

Download the raw data [here](https://springernature.figshare.com/articles/dataset/An_integrated_dataset_of_spatiotemporal_and_event_data_in_elite_soccer/28196177)

## Project Structure

data_processing.py: Functions for loading and processing metadata, event data, and position data.
visualization.py: Functions for visualizing the processed data.
data_summary.ipynb: Jupyter notebook to replicate the descriptive statistics and visualizations presented in the paper.
Data Source and Characteristics

Soccer matches from the German Bundesliga (1st and 2nd divisions)
Size: 7 full matches
Official metadata (match information)
Official event data.
Official position data captured by TRACAB

## Usage

Data Processing and visualization

Download the raw data here
Open the data_summary.ipynb notebook.
Define the path to your dataset directory in the path variable.
Run the cells to load and process the data.
The processed data summary will be displayed.

# Match Tracking Data Processing for Deep Learning

This module handles the extraction, transformation, and storage of DFL (Deutsche Fußball Liga) XML tracking data into a machine-learning-ready format.

## Overview
Raw tracking data is typically provided in XML format, which is excellent for standardized transfer but highly inefficient for training neural networks (like LSTMs or Transformers). This pipeline uses the [Floodlight](https://floodlight.readthedocs.io/) library to parse the XML files into `XY` spatial objects, extracts the underlying NumPy arrays, and saves them into a highly optimized compressed binary file (`.npz`).

## Why `.npz` instead of `.csv`?
Spatial tracking data inherently possesses a 3D spatio-temporal structure: **(Time $\times$ Entities $\times$ Coordinates)**. 
* Saving to a CSV requires flattening this structure into a 2D grid, which is slow to read/write and requires complex reshaping when building data loaders.
* Saving to a `.npz` (NumPy Zip) file preserves the exact 3D tensor shapes, compresses the massive file size, and allows deep learning frameworks (PyTorch/TensorFlow) to load the arrays instantly into memory.

## Data Pipeline

### 1. Inputs
The pipeline expects the following DFL XML files:
* `..._positions_raw_observed_...xml`
* `..._matchinformation_...xml`
* `..._events_raw_...xml`

### 2. Output Format
The processed data is saved as **`match_tracking_data.npz`**. 

This single file acts as a dictionary containing six distinct 2D NumPy arrays representing both halves of the match:
* `fh_home` : First Half - Home Team
* `fh_away` : First Half - Away Team
* `fh_ball` : First Half - Ball
* `sh_home` : Second Half - Home Team
* `sh_away` : Second Half - Away Team
* `sh_ball` : Second Half - Ball

### 3. Tensor Shapes
Each array maintains the shape `(T, N * 2)`, where:
* `T`: Number of timesteps (frames).
* `N`: Number of entities (players or ball).
* `2`: The x and y spatial coordinates.

*Example:* If a half has 65,000 frames and 11 players per team, the shape of the `fh_home` array will be `(65000, 22)`.

## Usage: Loading the Data for Training

Loading the `.npz` file in your training script is as simple as accessing a standard Python dictionary.

```python
import numpy as np

# Load the compressed dataset
dataset = np.load('match_tracking_data.npz')

# Extract arrays
X_train_fh_home = dataset['fh_home']
X_train_fh_away = dataset['fh_away']
X_train_fh_ball = dataset['fh_ball']

print(f"Loaded Home Team (First Half) Shape: {X_train_fh_home.shape}")