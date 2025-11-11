import numpy as np
import jax.numpy as jnp
from jax import Array

COVID_PATH = "/Users/gabriel.brown/Documents/nde/week3_neural_ode/data/raw/covid_data.npy"


class DataTransform:
    """Simple reversible transformation for scaling COVID data."""
    def __init__(self, mean: float, std: float, log_scale: bool = True):
        self.mean = mean
        self.std = std
        self.log_scale = log_scale

    def forward(self, y: np.ndarray) -> np.ndarray:
        """Apply log1p (if enabled) and standardization."""
        if self.log_scale:
            y = np.log1p(y)
        return (y - self.mean) / self.std

    def inverse(self, y: np.ndarray) -> np.ndarray:
        """Undo standardization and log1p transform."""
        y = y * self.std + self.mean
        if self.log_scale:
            y = np.expm1(y)
        return y


def load_covid_data(path: str = COVID_PATH,
                    normalize: bool = True,
                    return_transform: bool = True) -> tuple[Array, Array, DataTransform | None]:
    """Load and preprocess daily new COVID-19 infections.

    The file must have two columns: [day, new_infections].
    Applies log(1 + x) transform and standardization to the infection counts.
    Time is centered and scaled as well.
    """
    raw = np.load(path)
    days = raw[:, 0]
    cases = raw[:, 1]

    # Compute log1p(cases) for stats
    log_cases = np.log1p(cases)
    mean, std = log_cases.mean(), log_cases.std()

    transform = DataTransform(mean, std, log_scale=True)

    # Transform infections
    ys = transform.forward(cases)[:, None]

    # Center and scale time
    ts = (days - days.mean()) / days.std()

    ts = jnp.array(ts, dtype=jnp.float32)
    ys = jnp.array(ys, dtype=jnp.float32)

    if return_transform:
        return ts, ys, transform
    return ts, ys


# --- Run as standalone EDA script ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ts, ys, transform = load_covid_data(return_transform=True)

    print("First 10 time points (normalized):")
    print(np.array(ts[:10]))
    print("\nFirst 10 rows of infection data (log-scaled & normalized):")
    print(np.array(ys[:10]))
    print("\nTransform parameters:")
    print(f"mean = {transform.mean:.4f}, std = {transform.std:.4f}")

    # Example: invert normalized values back to raw units
    raw_pred_example = transform.inverse(np.array(ys[:5]))
    print("\nInverse-transformed example (approx original scale):")
    print(raw_pred_example)

    plt.figure(figsize=(10, 5))
    plt.plot(ts, ys[:, 0], color="crimson", linewidth=2)
    plt.title("Preprocessed COVID-19 Daily New Infections (Germany, First Wave)")
    plt.xlabel("Normalized Day")
    plt.ylabel("Normalized log(1 + cases)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
