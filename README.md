# Addressing Data Quality Decompensation in Federated Learning via Dynamic Client Selection

# Overview

This project implements Shapley-Bid Reputation Optimized Federated Learning (SBRO-FL), a method designed to address challenges in Federated Learning (FL), including data quality heterogeneity, compensation demands, and budget constraints. The SBRO-FL approach dynamically selects clients based on their historical performance and submitted bids while ensuring fairness and maintaining system stability.

# Platform

This experiment is built on top of the [FLEXible](https://github.com/FLEXible-FL/FLEXible) platform, an open-source framework for Federated Learning (FL). FLEXible provides a highly configurable and modular environment to simulate FL scenarios, making it an ideal choice for research and experimentation in FL.

# Installation

This project is built using Python and the Conda package management system. To set up the environment on your local machine, ensure that you have Conda installed.

## Project Structure

- **[experiment.py](experiments/experiment.py)**: This is the entry point for running the experiments. It contains the experiment parameters and path settings needed to configure and run the experiments.
- **[result](./result)**: This folder contains the results generated from the experiments. Each subfolder corresponds to specific experiment outcomes and configurations.

### Running the Experiment

To start an experiment, navigate to the **[experiment.py](experiments/experiment.py)** file and adjust the parameters as needed for your desired setup. Results will be saved automatically in the **result** folder.

### License

This code is released under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.  
You may use, modify, and distribute it under the terms of the license.

If you use this code in your project, especially in a web-based service, you must make your modified source code available as well.
