DEFAULT_HYPERPARAMS = {
    "batch_size": 64,
    "learning_rate": 1e-3,
    "num_epochs": 5,
}

HYPERPARAM_RANGES = {
    "batch_size": (1, 1024),
    "learning_rate": (1e-6, 1.0),
    "num_epochs": (1, 100),
}
