import argparse
import equinox as eqx
import matplotlib.pyplot as plt
from jax import random
import jax.numpy as jnp
import numpy as np
from config import datasets
from models.neural_ode import NeuralODE, Func
from config import WIDTH, DEPTH, SEED, SOLVER_CONFIG

NUM_PREDICTIONS = 100

def evaluate(dataset: str, model_path: str) -> None:
    if dataset not in datasets:
        raise ValueError(f"Unsupported dataset '{dataset}'. Available: {list(datasets.keys())}")

    # --- Load data + transform ---
    ts, ys, transform = datasets[dataset](return_transform=True)

    # --- Build model skeleton for loading ---
    dummy_key = random.PRNGKey(SEED)
    func = Func(
        in_dim=ys.shape[-1] + 1,
        out_dim=ys.shape[-1],
        width=WIDTH,
        depth=DEPTH,
        key=dummy_key,
    )
    model_like = NeuralODE(func, ys[0])  # y₀ value doesn't matter for loading
    model = eqx.tree_deserialise_leaves(model_path, model_like)

    # --- Evaluate model on data points ---
    y_pred = model(ts, **SOLVER_CONFIG["eval"])

    # --- Generate a finer time grid for smooth prediction ---
    num_fine = NUM_PREDICTIONS
    t_fine = jnp.linspace(ts[0], ts[-1], num_fine)
    y_pred_fine = model(t_fine, **SOLVER_CONFIG["eval"])

    # --- Invert normalization to real-world units ---
    y_pred_real = transform.inverse(np.array(y_pred_fine))
    ys_real = transform.inverse(np.array(ys))

    # --- Save predictions ---
    output_path = f"pred_{dataset}.npy"
    np.save(output_path, y_pred_real)
    print(f"\n💾 Saved predictions to {output_path}")
    print(f"Array shape: {y_pred_real.shape}\n")

    # --- Plot results ---
    plt.figure(figsize=(10, 5))
    if ys_real.shape[-1] == 1:
        plt.plot(ts, ys_real[:, 0], "o", label="True data", markersize=5, alpha=0.7)
        plt.plot(t_fine, y_pred_real[:, 0], "-", label="Predicted (fine)", linewidth=2)
    else:
        for i in range(ys_real.shape[-1]):
            plt.plot(ts, ys_real[:, i], "o", label=f"True dim {i}", markersize=5, alpha=0.7)
            plt.plot(t_fine, y_pred_real[:, i], "-", label=f"Pred dim {i} (fine)", linewidth=2)

    plt.title(f"Neural ODE evaluation: {dataset}")
    plt.xlabel("t")
    plt.ylabel("y (original units)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Neural ODE model.")
    parser.add_argument(
        "--dataset",
        choices=list(datasets.keys()),
        required=True,
        help="Dataset key from config.datasets.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to the trained model file. Defaults to 'trained_<dataset>.eqx'.",
    )
    args = parser.parse_args()

    model_path = args.model or f"trained_{args.dataset}.eqx"
    evaluate(args.dataset, model_path)
