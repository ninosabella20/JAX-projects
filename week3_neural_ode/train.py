import argparse
import jax.numpy as jnp, jax.random as jr
import equinox as eqx, optax
from models.neural_ode import Func, NeuralODE
from config import *


@eqx.filter_value_and_grad
def grad_loss(model, ts, ys):
    y_pred = model(ts, **SOLVER_CONFIG["train"])  # y₀ is now learned inside the model
    return jnp.mean((y_pred - ys) ** 2)


@eqx.filter_jit
def train_step(model, opt_state, ts, ys, optimizer):
    loss, grads = grad_loss(model, ts, ys)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


def main(dataset: str):
    if dataset not in datasets:
        raise ValueError(f"Unsupported dataset '{dataset}'")

    ts, ys = datasets[dataset]()

    key = jr.PRNGKey(SEED)
    func = Func(
        in_dim=ys.shape[-1] + 1,
        out_dim=ys.shape[-1],
        width=WIDTH,
        depth=DEPTH,
        key=key,
    )
    y0_init = ys[0]
    model = NeuralODE(func, y0_init)

    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    model_path = f"trained_{dataset}.eqx"

    print(f"🚀 Starting training on '{dataset}'...\n")

    try:
        for step in range(NUM_STEPS):
            loss, model, opt_state = train_step(model, opt_state, ts, ys, optimizer)
            if step % PRINT_EVERY == 0:
                print(f"Step {step:5d} | loss = {loss:.6f}")

    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user.")
        print(f"💾 Saving current model to {model_path} before exiting...")
        eqx.tree_serialise_leaves(model_path, model)
        print("✅ Model saved successfully.")
        return

    print("\n✅ Training complete!")
    eqx.tree_serialise_leaves(model_path, model)
    print(f"💾 Saved final model to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=datasets.keys(), required=True)
    args = parser.parse_args()
    main(args.dataset)
