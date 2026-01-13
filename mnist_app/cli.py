import torch

from .config import HYPERPARAM_RANGES


def prompt_device():
    has_cuda = torch.cuda.is_available()
    if not has_cuda:
        print("CUDA not available. Using CPU.")
        return torch.device("cpu")

    while True:
        choice = input("Choose device (1=GPU, 2=CPU): ").strip()
        if choice == "1":
            return torch.device("cuda")
        if choice == "2":
            return torch.device("cpu")
        print("Invalid choice. Please enter 1 or 2.")


def prompt_hyperparameters(defaults):
    params = dict(defaults)
    while True:
        print("\nDefault hyperparameters:")
        print("1) batch_size =", params["batch_size"], "(range: 1-1024)")
        print("2) learning_rate =", params["learning_rate"], "(range: 1e-6 to 1.0)")
        print("3) num_epochs =", params["num_epochs"], "(range: 1-100)")

        change = input("Change hyperparameter? (y/n): ").strip().lower()
        if change == "n":
            return params
        if change != "y":
            print("Please enter 'y' or 'n'.")
            continue

        choice = input("Which hyperparameter to change? (1-3): ").strip()
        if choice == "1":
            params["batch_size"] = _prompt_int(
                "New batch_size (1-1024): ",
                HYPERPARAM_RANGES["batch_size"],
            )
        elif choice == "2":
            params["learning_rate"] = _prompt_float(
                "New learning_rate (1e-6 to 1.0): ",
                HYPERPARAM_RANGES["learning_rate"],
            )
        elif choice == "3":
            params["num_epochs"] = _prompt_int(
                "New num_epochs (1-100): ",
                HYPERPARAM_RANGES["num_epochs"],
            )
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
            continue

        print("change complete")


def _prompt_int(prompt, value_range):
    min_val, max_val = value_range
    while True:
        raw = input(prompt).strip()
        try:
            value = int(raw)
        except ValueError:
            print("Invalid integer.")
            continue
        if not (min_val <= value <= max_val):
            print("Out of range.")
            continue
        return value


def _prompt_float(prompt, value_range):
    min_val, max_val = value_range
    while True:
        raw = input(prompt).strip()
        try:
            value = float(raw)
        except ValueError:
            print("Invalid float.")
            continue
        if not (min_val <= value <= max_val):
            print("Out of range.")
            continue
        return value
