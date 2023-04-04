import jax
import jax.numpy as jnp
import models.utils as utils
import numpy as np
import einops


class HardObstacle:
    def __init__(self, rng, dimension, sigma_v, T, N) -> None:
        self.dimension = dimension

        self.N = N
        self.T = T
        self.sv = sigma_v
        self.xtrue = self.generate_x(rng)

        #
        self.w0_bound = 1
        self.wt_bound = 1 / self.sv
        self.w_prev_bound_fn = lambda x_next: 1 / self.sv * jnp.ones(x_next.shape[0])
        self.w_next_bound_fn = lambda x_prev: 1 / self.sv * jnp.ones(x_prev.shape[0])
        self.w_init_fn = lambda x: jnp.ones(x.shape[0])

    def single_transition_fn(self, x_next, x_prev):
        x_pred = self.push_forward(x_prev)
        return (
            jnp.exp(-jnp.sum((x_next - x_pred) ** 2, axis=-1) / (2 * self.sv**2))
            / self.sv
        )

    def push_forward(self, x_prev):
        return x_prev

    def sample_xs_fn(self, rng):
        return jax.random.uniform(rng, (self.T, self.N, self.dimension))

    def weight_matrix_fn(self, x_prev, x_t):
        x_pred = jax.vmap(self.push_forward)(x_prev)
        dists = utils.compute_squared_distances(x_pred, x_t)
        logw = dists / (2.0 * self.sv**2)
        logwmin = jnp.min(logw)
        w = jnp.exp(-logwmin) * jnp.exp(-logw + logwmin) / self.sv
        return w

    def generate_x(self, rng):
        # true hidden state
        T = self.T
        d = self.dimension
        xtrue = jnp.zeros((T, d))
        rng, step_rng = jax.random.split(rng)
        xtrue = xtrue.at[0].set(jax.random.randn(step_rng, (d,)))
        for t in range(1, T):
            x2 = self.push_forward(xtrue[t - 1])
            rng, step_rng = jax.random.split(rng)
            next_x = x2 + self.sv * jax.random.randn(step_rng, (d,))
            xtrue = xtrue.at[t].set(next_x)
        return xtrue


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time
    import functools
    import ers
    import pandas as pd
    
    num_trials = 5_000
    batch_n = 10
    seed = 0
    results = []
    
    data_rng = jax.random.PRNGKey(0)
    
    for T in [100, 250, 500, 1_000]:
        for N in [T, 2*T, 5*T, 10*T]:
            for d in [1,2,3]:
                model = HardObstacle(rng = data_rng, N=N, T=T, dimension=d, sigma_v=0.2)

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
                
                
                n_trials = 0 
                n_accepts = 0
                tic = time.time()
                for _ in range(num_trials // batch_n):
                    rng, rng_step = jax.random.split(rng)
                    accept = sample_n(rng_step, batch_n)
                    n_accepts += jnp.sum(accept)
                    n_trials += len(accept)
                toc = time.time()

                p_acc = n_accepts / n_trials
                duration = toc - tic

                cols = ['n_trials', 'n_accepts', 'p_acc', 'duration', 'N', 'T', 'd']
                item = [n_trials, n_accepts.item(), p_acc.item(), duration, N, T, d]
                results.append(item)
                df = pd.DataFrame(results, columns = cols)
                df.to_csv('./hardobstacle_results.csv')
                print(item)
