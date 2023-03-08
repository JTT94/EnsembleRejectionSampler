import jax
import jax.numpy as jnp
import models.utils as utils
import numpy as np
import einops


class StochasticVolatility:
    def __init__(self, dimension, alpha, beta, sigma_v, N, ytrue) -> None:
        self.dimension = dimension
        self.N = N
        self.alpha = alpha
        self.beta = beta
        self.sv = sigma_v
        self.ss = self.sv / np.sqrt(1 - alpha**2)

        # self.xtrue = self.generate_x()
        # self.ytrue = self.generate_y(self.xtrue)
        self.ytrue = einops.rearrange(ytrue, "t -> t 1")
        self.T = self.ytrue.shape[0]

        #
        self.w0_bound = 1.0 / self.ss
        self.wt_bound = 1 / self.sv
        self.w_prev_bound_fn = lambda x_next: 1 / self.sv * jnp.ones(x_next.shape[0])
        self.w_next_bound_fn = lambda x_prev: 1 / self.sv * jnp.ones(x_prev.shape[0])
        self.w_init_fn = (
            lambda x: jnp.exp(-jnp.sum(x**2, axis=-1) / (2 * self.ss**2)) / self.ss
        )

    def single_transition_fn(self, x_next, x_prev):
        x_pred = self.push_forward(x_prev)
        return (
            jnp.exp(-jnp.sum((x_next - x_pred) ** 2, axis=-1) / (2 * self.sv**2))
            / self.sv
        )

    def push_forward(self, x_prev):
        return self.alpha * x_prev

    def sample_xs_fn(self, rng):
        T, N, d = self.T, self.N, self.dimension
        rng, rng_x = jax.random.split(rng)

        y_batch = einops.repeat(self.ytrue, "t d -> t n d", n=N)

        x = (
            jnp.log(y_batch**2)
            - jnp.log(self.beta**2)
            - jnp.log(jax.random.normal(rng_x, (T, N, d)) ** 2)
        )
        return x

    def weight_matrix_fn(self, x_prev, x_t):
        x_pred = self.push_forward(x_prev)
        dists = utils.compute_squared_distances(x_pred, x_t)
        logw = dists / (2.0 * self.sv**2)
        logwmin = jnp.min(logw)
        w = jnp.exp(-logwmin) * jnp.exp(-logw + logwmin) / self.sv
        return w


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pandas as pd
    import ers
    import time
    import functools

    fp = "/Users/jamesthornton/ers/EnsembleRejectionSampler/data/sp500returns.csv"
    ytrue = pd.read_csv(fp).returns[-200:].values.copy()
    model = StochasticVolatility(
        N=2_000, dimension=1, alpha=0.95, beta=0.7, sigma_v=0.3, ytrue=ytrue
    )

    transition_prob_fn = jax.vmap(model.single_transition_fn, in_axes=(None, 0))
    ers_step = ers.get_ers_step(
        sample_xs_fn=model.sample_xs_fn,
        transition_prob_fn=transition_prob_fn,
        w_init_fn=model.w_init_fn,
        weight_matrix_fn=model.weight_matrix_fn,
        w0_bound=model.w0_bound,
        wt_bound=model.wt_bound,
        w_prev_bound_fn=model.w_prev_bound_fn,
        w_next_bound_fn=model.w_next_bound_fn,
    )
    rng = jax.random.PRNGKey(0)

    # ---------------
    jit_ers_step = jax.jit(ers_step)
    accept, x_traj, ts, x_ind = jit_ers_step(rng)

    @jax.jit
    def body_fn(rng):
        accept, x_traj, ts, x_ind = jit_ers_step(rng)
        return accept

    @functools.partial(jax.jit, static_argnums=(1,))
    def sample_n(rng, n):
        rngs = jax.random.split(rng, n)
        rngs = jnp.array(rngs)
        return jax.vmap(body_fn)(rngs)

    _ = sample_n(rng, 1)

    tic = time.time()
    accept = sample_n(rng, 100)
    toc = time.time()

    print(toc - tic)
    print(jnp.mean(accept))
    print(accept.shape)

# export PYTHONPATH=/Users/jamesthornton/ers/EnsembleRejectionSampler/ers
# XLA_FLAGS="--xla_force_host_platform_device_count=8"

# export XLA_FLAGS=--xla_dump_to=/tmp/xla_dump
