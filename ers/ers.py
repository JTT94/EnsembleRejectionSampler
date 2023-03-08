import jax.numpy as jnp
import jax


"""

joint_filter_prob =  p(x_1,y_1)
joint_filter_prob =  p(x_t,y_t | y_1:t)
state_filter_prob =  p(x_t| y_1:t)
joint_predict_prob =  p(x_t+1,y_t+1 | y_1:t)
incremental__likelihood = p(y_t+1 | y_1:t)

# joint predictive state
p(x^i_t,y_t | y_1:t-1) = sum_j p(x^j_t-1| y_1:t-1) w(x^j_t-1, x^i_t)
sum [N] x [N x N] -> [N]
joint_predict_prob =  \sum_j state_filter_prob_j * weight_matrix_ij

# incremental likelihood
incremental__likelihood =  sum joint_predict_prob
p(y_t |y_1:t-1) = sum_i p(x^i_t,y_t | y_1:t-1)

# state filter
state_filter_prob = joint_predict_prob / incremental__likelihood
p(x^i_t| y_1:t) = p(x^i_t,y_t | y_1:t-1) / p(y_t |y_1:t-1)

"""


def get_forward_filtering_fn(
    w_init_fn,
    weight_matrix_fn,
    w0_bound=None,
    wt_bound=None,
    w_prev_bound_fn=None,
    w_next_bound_fn=None,
):
    def forward_filtering(
        xs,
        bound_indices=None,
    ):
        t_init = 0
        x_0 = xs[t_init]
        w_init = w_init_fn(x_0)

        if bound_indices is not None:
            bound_i = bound_indices[t_init]
            w_init = w_init.at[bound_i].set(w0_bound)

        joint_predict_prob_0 = w_init
        incremental_likelihood_0 = jnp.sum(joint_predict_prob_0)
        state_filter_prob_0 = joint_predict_prob_0 / incremental_likelihood_0
        log_likelihood = jnp.log(incremental_likelihood_0)

        def body_fn(carry_state, x_t):
            log_likelihood, state_filter_prob_prev, t_prev, x_prev = carry_state
            t = t_prev + 1
            # p(x^i_t,y_t | y_1:t-1) = sum_j p(x^j_t-1| y_1:t-1) w(x^j_t-1, x^i_t)
            weight_matrix = weight_matrix_fn(x_prev, x_t)

            if bound_indices is not None:
                bound_i = bound_indices[t]
                bound_prev_i = bound_i - 1
                weight_matrix = weight_matrix.at[bound_prev_i, :].set(
                    w_prev_bound_fn(x_t)
                )
                weight_matrix = weight_matrix.at[bound_prev_i, bound_i].set(wt_bound)
                weight_matrix = weight_matrix.at[:, bound_i].set(
                    w_next_bound_fn(x_prev)
                )

            joint_predict_prob_t = state_filter_prob_prev @ weight_matrix

            # sum_i p(x^i_t,y_t | y_1:t-1)
            incremental_likelihood_t = jnp.sum(joint_predict_prob_t)

            # p(x^i_t| y_1:t) = p(x^i_t,y_t | y_1:t-1) / p(y_t |y_1:t-1)
            state_filter_prob_t = joint_predict_prob_t / incremental_likelihood_t

            log_likelihood += jnp.log(incremental_likelihood_t)

            carry_state = (log_likelihood, state_filter_prob_t, t, x_t)
            store_state = (
                state_filter_prob_t,
                joint_predict_prob_t,
                t,
                incremental_likelihood_t,
            )
            return carry_state, store_state

        carry_state, store_state = jax.lax.scan(
            body_fn, (log_likelihood, state_filter_prob_0, t_init, x_0), xs[1:]
        )
        log_likelihood = carry_state[0]
        state_filter_probs, _, ts, _ = store_state
        state_filter_probs = jnp.concatenate(
            [state_filter_prob_0[None, :], state_filter_probs], axis=0
        )
        ts = jnp.concatenate([jnp.expand_dims(t_init, 0), ts], axis=0)
        return log_likelihood, state_filter_probs, ts

    return forward_filtering


def get_backward_sampling_fn(transition_prob_fn):
    # vmap transition_prob_fn, on second axis
    # proposal_xs # [T, N, *d]
    # sampled_xs # [T, *d]

    def backward_sampling(rng, proposal_xs, state_filter_probs):
        T, N = proposal_xs.shape[:2]
        t_final = T - 1  # 0-indexed
        rng, rng_x_T = jax.random.split(rng)
        x_index = jax.random.choice(rng_x_T, N, (), p=state_filter_probs[-1])
        x_T = proposal_xs[-1, x_index]
        rngs = jax.random.split(rng, proposal_xs.shape[0] - 1)

        def step_fn(carry_state, rng):
            x_t, t = carry_state
            state_filter_probs_t = state_filter_probs[t - 1]
            x_t_minus_1s = proposal_xs[t - 1]

            transition_prob = transition_prob_fn(x_t, x_t_minus_1s)
            state_prob = transition_prob * state_filter_probs_t
            state_prob = state_prob / jnp.sum(state_prob)
            x_index = jax.random.choice(rng, N, (), p=state_prob)
            x_t_minus_1 = x_t_minus_1s[x_index]
            carry_state = (x_t_minus_1, t - 1)
            store_state = (x_t_minus_1, t - 1, x_index)

            return carry_state, store_state

        _, store_state = jax.lax.scan(step_fn, (x_T, t_final), rngs)

        x_traj, ts, x_ind = store_state
        # concat initial state and reverse order
        x_traj = jnp.concatenate([jnp.expand_dims(x_T, 0), x_traj])[::-1]
        ts = jnp.concatenate([jnp.array([t_final]), ts])[::-1]
        x_ind = jnp.concatenate([jnp.array([x_index]), x_ind])[::-1]
        return x_traj, ts, x_ind

    return backward_sampling


def get_ers_step(
    sample_xs_fn,
    transition_prob_fn,
    w_init_fn,
    weight_matrix_fn,
    w0_bound,
    wt_bound,
    w_prev_bound_fn,
    w_next_bound_fn,
):
    """_summary_

    Args:
        sample_xs_fn: rng -> [T, N, *d]
        transition_prob_fn: x_next [d], x_prev [N, d] -> [N]
        w_init_fn (_type_): _description_
        weight_matrix_fn (_type_): _description_
        w0_bound (_type_): _description_
        wt_bound (_type_): _description_
        w_prev_bound_fn (_type_): _description_
        w_next_bound_fn (_type_): _description_

    Returns:
        _type_: _description_
    """
    # ers step
    forward_filter = get_forward_filtering_fn(w_init_fn, weight_matrix_fn)
    backward_sampling = get_backward_sampling_fn(transition_prob_fn)
    forward_bound_filter = get_forward_filtering_fn(
        w_init_fn,
        weight_matrix_fn,
        w0_bound=w0_bound,
        wt_bound=wt_bound,
        w_prev_bound_fn=w_prev_bound_fn,
        w_next_bound_fn=w_next_bound_fn,
    )

    def ers_step(rng):
        rng_proposal_x, rng_backward_sample, rng_accept_reject = jax.random.split(
            rng, 3
        )
        proposal_xs = sample_xs_fn(rng_proposal_x)

        log_likelihood, state_filter_probs, ts = forward_filter(proposal_xs)

        x_traj, ts, x_ind = backward_sampling(
            rng_backward_sample, proposal_xs, state_filter_probs
        )

        bound_log_likelihood, _, _ = forward_bound_filter(proposal_xs, x_ind)

        # accept reject step
        accept = (
            jnp.log(jax.random.uniform(rng_accept_reject))
            < log_likelihood - bound_log_likelihood
        )
        return accept, x_traj, ts, x_ind

    return ers_step


if __name__ == "__main__":
    T = 50
    N = 128
    d = 1
    rng = jax.random.PRNGKey(0)

    from models.utils import exp_squared_distances

    weight_matrix_fn = lambda x_t, x_prev: exp_squared_distances(x_t, x_prev, 1.0)
    w_init_fn = lambda x: jnp.ones(x.shape[0]) / x.shape[0]

    w0_bound = 0.1
    wt_bound = 0.1
    w_prev_bound_fn = lambda x_next: 0.1 * jnp.ones(x_next.shape[0])
    w_next_bound_fn = lambda x_prev: 0.1 * jnp.ones(x_prev.shape[0])
    transition_prob_fn = lambda x_next, x_prev: 0.1
    transition_prob_fn = jax.vmap(transition_prob_fn, in_axes=(None, 0))

    def sample_xs_fn(rng):
        rng, rng_x = jax.random.split(rng)
        x = jax.random.normal(rng_x, (T, N, d))
        return x

    ers_step = get_ers_step(
        sample_xs_fn=sample_xs_fn,
        transition_prob_fn=transition_prob_fn,
        w_init_fn=w_init_fn,
        weight_matrix_fn=weight_matrix_fn,
        w0_bound=w0_bound,
        wt_bound=wt_bound,
        w_prev_bound_fn=w_prev_bound_fn,
        w_next_bound_fn=w_next_bound_fn,
    )

    jit_ers_step = jax.jit(ers_step)
    accept, x_traj, ts, x_ind = ers_step(rng)
