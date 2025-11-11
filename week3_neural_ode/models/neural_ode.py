import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
import diffrax


class Func(eqx.Module):
    mlp: eqx.nn.MLP
    out_scale: jnp.ndarray

    def __init__(self, in_dim, out_dim, width, depth, *, key):
        self.mlp = eqx.nn.MLP(
            in_dim, out_dim, width, depth,
            activation=jnp.tanh,
            final_activation=jnp.tanh,
            key=key,
        )
        self.out_scale = jnp.array(1.0)

    def __call__(self, t, y, args):
        inp = jnp.concatenate([y, jnp.atleast_1d(t)])
        return self.out_scale * self.mlp(inp)


class NeuralODE(eqx.Module):
    func: Func
    y0: jnp.ndarray

    def __init__(self, func, y0_init):
        self.func = func
        self.y0 = jnp.array(y0_init)

    def __call__(self, ts, *, solver=None, substeps: int = 10, controller=None):
        term = diffrax.ODETerm(self.func)
        solver = solver or diffrax.Euler()
        controller = controller or diffrax.ConstantStepSize()

        dt_data = ts[1] - ts[0]
        dt_solver = dt_data / substeps
        saveat = diffrax.SaveAt(ts=ts)

        sol = diffrax.diffeqsolve(
            term, solver,
            t0=ts[0], t1=ts[-1],
            dt0=dt_solver,
            y0=self.y0,
            saveat=saveat,
            stepsize_controller=controller,
        )
        return sol.ys
