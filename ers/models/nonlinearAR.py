import jax
import jax.numpy as jnp
import models.utils as utils
import numpy as np
import einops


class NonlinearAR:
    def __init__(self, rng, dimension, alpha, sigma_v, sigma_w, T, N) -> None:
        self.dimension = dimension
        if np.isscalar(alpha):
            alpha = np.eye(dimension) * alpha

        self.N = N
        self.alpha = alpha
        self.T = T
        self.sv = sigma_v
        self.sw = sigma_w
        
        x_rng, y_rng = jax.random.split(rng)
        self.xtrue = self.generate_x(x_rng)
        self.ytrue = self.generate_y(self.xtrue, y_rng)

        #
        self.w0_bound = 1
        self.wt_bound = 1 / self.sv
        self.w_prev_bound_fn = lambda x_next: 1 / self.sv * jnp.ones(x_next.shape[0])
        self.w_next_bound_fn = lambda x_prev: 1 / self.sv * jnp.ones(x_prev.shape[0])
        self.w_init_fn = lambda x: jnp.exp(-jnp.sum(x**2, axis=-1) / 2)

    def single_transition_fn(self, x_next, x_prev):
        x_pred = self.push_forward(x_prev)
        return (
            jnp.exp(-jnp.sum((x_next - x_pred) ** 2, axis=-1) / (2 * self.sv**2))
            / self.sv
        )

    def push_forward(self, x_prev):
        return self.alpha @ jnp.tanh(x_prev)

    def sample_xs_fn(self, rng):
        T, N, d = self.T, self.N, self.dimension
        rng, rng_x = jax.random.split(rng)
        y_batch = einops.repeat(self.ytrue, "t d -> t n d", n=N)
        x = y_batch + self.sw * jax.random.normal(rng_x, (T, N, d))
        return x

    def weight_matrix_fn(self, x_prev, x_t):
        x_pred = jax.vmap(self.push_forward)(x_prev)
        dists = utils.compute_squared_distances(x_prev, x_t)
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
        xtrue = xtrue.at[0].set(jax.random.normal(step_rng, (d,)))
        for t in range(1, T):
            rng, step_rng = jax.random.split(rng)
            x2 = self.push_forward(xtrue[t - 1])
            next_x = x2 + self.sv * jax.random.normal(step_rng, (d,))
            xtrue = xtrue.at[t].set(next_x)
        return xtrue

    def generate_y(self, x, rng):
        T = x.shape[0]
        d = self.dimension
        y = jnp.zeros((T, d))
        rng, step_rng = jax.random.split(rng)
        y_0 = x[0] + self.sw * jax.random.normal(step_rng, (d,))
        y = y.at[0].set(y_0) 

        for t in range(1, T):
            rng, step_rng = jax.random.split(rng)
            y_next = x[t] + self.sw * jax.random.normal(step_rng, (d,))
            y = y.at[t].set(y_next) 
        return y


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import ers
    import time
    import functools
    import pandas as pd
    
    num_trials = 1_000
    batch_n = 10
    seed = 42
    tag = "_d_trials"
    
    if tag == "_d_exp":
        t_range = [25, 50, 100, 200]
        n_range = [100, 250, 500, 1_000, 5_000, 10_000]
        d_range = [1,2, 3, 4, 5, 6, 7, 8]
        
    if tag == "_t_exp"
        t_range = [100, 250, 500, 1_000]
#         n_range = [T, 2*T, 5*T, 10*T]
        d_range = [1,2,3]
        
    results = []

    for T in t_range:
        
        if tag == "_t_exp":
            n_range = [T, 2*T, 5*T, 10*T]
            
        for N in n_range:
            for d in d_range:
                
                model = NonlinearAR(rng=jax.random.PRNGKey(0), N=N, T=T, dimension=d, alpha=0.9, sigma_v=0.3, sigma_w=0.1)
                
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

                _ = sample_n(rng, 2)

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
                df.to_csv(f'./nonlinear_ar_results{tag}.csv')
                print(item)
                
                if n_accepts < 1:
                    break