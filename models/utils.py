import jax.numpy as jnp
import jax


# compute squared distance matrix between two sets of points
def compute_squared_distances(x1, x2):
    x1sq = jnp.sum(x1**2, axis=-1)
    x2sq = jnp.sum(x2**2, axis=-1)
    dists = x1sq[:, None] + x2sq[None, :] - 2 * jnp.dot(x1, x2.T)
    return dists


def _log_matvec(log_A, log_b):
    Amax = jnp.max(log_A)
    bmax = jnp.max(log_b)
    A = jnp.exp(log_A - Amax)
    b = jnp.exp(log_b - bmax)
    return Amax + bmax + jnp.log(A @ b)


def batch_matmul(x, y, in_axes=(0, 0)):
    return jax.vmap(lambda x, y: x @ y, in_axes=in_axes)


def exp_squared_distances(x1: jnp.ndarray, x2: jnp.ndarray, sigma: float):
    """_summary_

    Args:
        x1: [N, *d]
        x2: [M, *d]
        sigma: std of the gaussian

    Returns:
        weight_matrix: [N, M]
    """
    dists = compute_squared_distances(x1, x2)
    log_w = dists / (2 * sigma**2)
    log_w_min = jnp.min(log_w)
    weight_matrix = jnp.exp(-log_w_min) * jnp.exp(-log_w + log_w_min) / sigma
    return weight_matrix
