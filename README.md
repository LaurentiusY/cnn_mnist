# MNIST CNN CLI

Train a simple CNN on the MNIST dataset from a command-line interface. The app lets you choose CPU/GPU (if available), review default hyperparameters, optionally update them, and then trains the model while reporting accuracy.

## Requirements
- Python 3.x
- PyTorch and torchvision

Install dependencies:
```
python -m pip install torch torchvision
```

## How to Run
From the project root:
```
python pytorch_project_mnist
```


## CLI Flow
1) Choose device (GPU/CPU; GPU only if CUDA is available)
2) View default hyperparameters and optionally change them
3) See "model is training, please wait"
4) On completion, see "training completed" and final accuracy

## Project Structure
```
mnist_app/
  __init__.py
  cli.py        # user prompts
  config.py     # default hyperparameters and ranges
  data.py       # datasets and dataloaders
  main.py       # entrypoint
  model.py      # CNN model
  train.py      # training/evaluation loops
pytorch_project_mnist  # thin wrapper entrypoint
run_mnist.cmd          # Windows launcher
```
