# Image Classification on CIFAR 10 with PyTorch lightning

This code is based on the [PyTorch Lightning CIFAR10 ~94% Baseline Tutorial](https://pytorchlightning.github.io/lightning-tutorials/notebooks/lightning_examples/cifar10-baseline.html). This is part of a Code Sprint organized by [Artificial Inteligence and Cybernetics Lab](https://agencylab.github.io/) at [Independent University, Bangladesh](http://www.iub.edu.bd/). 

# Setup
 - Create a virtual environment using `python -m venv venv`
 - Run 'pip install -r requirements.txt' to install the required packages.
 - Edit the `utils/setup_configs.py` file to change the hyperparameters.
 - Run 'setup_configs.py' to create the `configs/configs.yaml` file.
 - Run 'tests/test_datamodule.py' to test the data module.
 - Run 'tests/test_model.py' to test the model.
 - Run 'train.py' to train the model.
 - Run 'test.py' to test the model.