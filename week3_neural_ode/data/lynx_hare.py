import numpy as np
import jax.numpy as jnp
from jax import Array

LYNX_HARE_PATH = "/Users/gabriel.brown/Documents/nde/week3_neural_ode/data/raw/LH_data.npy"


class DataTransform:
    """Reversible normalization for multi-dimensional data."""
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean
        self.std = std

    def forward(self, y: np.ndarray) -> np.ndarray:
        """Standardize (zero mean, unit variance)."""
        return (y - self.mean) / self.std

    def inverse(self, y: np.ndarray) -> np.ndarray:
        """Undo standardization."""
        return y * self.std + self.mean


def load_lynx_hare_data(
    path: str = LYNX_HARE_PATH,
    normalize: bool = True,
    return_transform: bool = False
) -> tuple[Array, Array, DataTransform | None]:
    raw = np.load(path)
    years = raw[:, 0]
    hares = raw[:, 1]
    lynx = raw[:, 2]

    # Stack into (N, 2)
    ys = np.stack([hares, lynx], axis=1)

    # Compute transform parameters
    mean, std = ys.mean(axis=0), ys.std(axis=0)
    transform = DataTransform(mean, std)

    if normalize:
        ys = transform.forward(ys)

    # Normalize years (center and scale)
    ts = (years - years.mean()) / years.std()

    ts = jnp.array(ts, dtype=jnp.float32)
    ys = jnp.array(ys, dtype=jnp.float32)

    if return_transform:
        return ts, ys, transform
    return ts, ys


# --- Run as standalone EDA script ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ts, ys, transform = load_lynx_hare_data(return_transform=True)

    print("First 5 time points (normalized):")
    print(np.array(ts[:5]))
    print("\nFirst 5 rows of population data (normalized):")
    print(np.array(ys[:5]))
    print("\nTransform parameters:")
    print(f"mean = {transform.mean}, std = {transform.std}")

    # Example inverse transform check
    example = np.array(ys[:3])
    print("\nInverse transform example:")
    print(transform.inverse(example))

    plt.figure(figsize=(10, 5))
    plt.plot(ts, ys[:, 0], label="Hares (normalized)", color="dodgerblue")
    plt.plot(ts, ys[:, 1], label="Lynx (normalized)", color="crimson")
    plt.title("Preprocessed Lynx–Hare Data")
    plt.xlabel("Normalized Year")
    plt.ylabel("Normalized Population")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
