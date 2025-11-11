import jax.numpy as jnp
from jax import Array


def get_sine_data(num_points: int = 200, t_max: float = 10.0) -> tuple[Array, Array]:
    ts = jnp.linspace(0, t_max, num_points)
    ys = jnp.sin(ts).reshape(-1, 1)
    return ts, ys
