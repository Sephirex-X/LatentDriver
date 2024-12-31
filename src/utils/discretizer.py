import numpy as np
class Discretizer:
  """Discretizes continuous values into a 1-dimensional array.

  The total number of discrete actions is equal to the product of (all
  bins + 1). We add +1 to be inclusive of boundaries of the min and max values.
  If the continuous value has a shape of (..., 3), and 3 bins are used
  with bin sizes [2, 3, 4], then there will be a total of 60 discrete actions
  (3 * 4 * 5).
  """

  def __init__(
      self, min_value:  np.array, max_value: np.array, bins: np.array
  ) -> None:
    """Initializes internal discretizer state.

    Args:
      min_value: Minimal values for the different vector elements to discretize
        of shape (num_vector_elements,).
      max_value: Maximum values for the different vector elements to discretize
        of shape (num_vector_elements,).
      bins: Number of bins for the different vector elements to discretize of
        shape (num_vector_elements,).
    """
    if min_value.shape != max_value.shape and min_value.shape != bins.shape:
      raise ValueError('Shapes do not match.')
    self._mins = min_value
    self._maxs = max_value
    self._bins = bins
    self._shift = min_value
    self._scale = bins / (max_value - min_value)
    self._max_discrete_idx = np.prod(self._bins+1) -1
    # self._max_discrete_idx = np.prod(self._bins)

  def discretize(self, values: np.array) -> np.array:
    """Discretizes a continuous batched n-d vector of values to 1d indices.

    Args:
      values: Vector of continuous values of shape (..., num_vector_elements) to
        discretize.

    Returns:
      Discretized values in a tensor of shape (..., 1) with maximum
        value self._max_discrete_idx.
    """
    if values.shape[-1] != self._mins.shape[-1]:
      raise ValueError('Input value shape does not match bin shape.')
    normalized_indices = (values - self._shift) * self._scale
    indices_nd = np.rint(
        np.maximum(np.minimum(normalized_indices, self._bins), 0)
    ).astype(np.int32)
    indices_1d = np.ravel_multi_index(  # pytype: disable=wrong-arg-types  # jnp-type
        np.split(indices_nd, self._bins.shape[0], -1),
        self._bins + 1,
        mode='clip',
    )
    return indices_1d

  def make_continuous(self, indices_1d: np.array) -> np.array:
    """Takes a discretized matrix and converts it back to continuous values.

    Args:
      indices_1d: Discrete matrix of shape (..., 1) to convert back to
        continuous matrices of shape (..., num_vector_elements).

    Returns:
      Continuous values of shape (..., num_vector_elements) corresponding to the
        value discretized by `indices_1d`.
    """
    indices_nd = np.stack(
        np.unravel_index(np.reshape(indices_1d, [-1]), self._bins + 1),
        axis=-1,
    )
    # Shape: (..., num_vector_elements)
    indices_nd = np.reshape(
        indices_nd, list(indices_1d.shape[:-1]) + [self._bins.shape[-1]]
    )
    values = indices_nd.astype(np.float32)
    return values / self._scale + self._shift