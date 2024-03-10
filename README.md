# Navier-Stokes Equations Solution Using Deep Learning

This repository contains a Python implementation of a Deep Learning-based approach to solving the Navier-Stokes equations, which are fundamental in fluid dynamics. The solution utilizes a Physics-Informed Neural Network (PINN) to model fluid flow around a cylindrical obstacle, capturing complex flow patterns such as vortices.

## Features
- Utilizes the PINN architecture to enforce the Navier-Stokes equations directly within the loss function, ensuring the learned model adheres to physical laws.
- Demonstrates how to preprocess fluid dynamics data for use in deep learning models.
- Offers insights into leveraging automatic differentiation for solving partial differential equations (PDEs) relevant to fluid mechanics.

## Requirements
Before running the script, ensure you have the following packages installed:

- `torch`: For building and training the neural network model.
- `numpy`: For handling numerical operations.
- `scipy`: Used for loading the dataset.
- `matplotlib`: For visualizing the results.

You can install these packages using pip:
```bash
pip install torch numpy scipy matplotlib
```

# Data Source

The data used in this project for training and testing the Physics-Informed Neural Network model was obtained from the following source:

## [High Precision Machine (HPM) Data](https://github.com/maziarraissi/HPM/tree/master/Data) by Maziar Raissi

This dataset is part of a collection of data used for various physics-informed machine learning projects. Specifically, the `cylinder_wake.mat` file contains the data used for solving the Navier-Stokes equation in our project.

### How to Access the Data

To use the same dataset for your experiments, please follow these steps:

1. Visit the [HPM Data Repository](https://github.com/maziarraissi/HPM/tree/master/Data).
2. Navigate to the folder containing the dataset you are interested in. For this project, the relevant file is `cylinder_wake.mat`.
3. Download the dataset directly to your local machine or clone the entire repository using Git:

```bash
git clone https://github.com/maziarraissi/HPM.git
```
