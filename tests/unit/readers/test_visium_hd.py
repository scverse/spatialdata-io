import numpy as np

from spatialdata_io.readers.visium_hd import (
    _decompose_projective_matrix,
    _projective_matrix_is_affine,
)

# --- UNIT TESTS FOR HELPER FUNCTIONS ---


def test_projective_matrix_is_affine() -> None:
    """Test the affine matrix check function."""
    # An affine matrix should have [0, 0, 1] as its last row
    affine_matrix = np.array([[2, 0.5, 10], [0.5, 2, 20], [0, 0, 1]])
    assert _projective_matrix_is_affine(affine_matrix)

    # A projective matrix is not affine if the last row is different
    projective_matrix = np.array([[2, 0.5, 10], [0.5, 2, 20], [0.01, 0.02, 1]])
    assert not _projective_matrix_is_affine(projective_matrix)


def test_decompose_projective_matrix() -> None:
    """Test the decomposition of a projective matrix into affine and shift components."""
    projective_matrix = np.array([[1, 2, 3], [4, 5, 6], [0.1, 0.2, 1]])
    affine, shift = _decompose_projective_matrix(projective_matrix)

    expected_affine = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])

    # The affine component should be correctly extracted
    assert np.allclose(affine, expected_affine)
    # Recomposing the affine and shift matrices should yield the original projective matrix
    assert np.allclose(affine @ shift, projective_matrix)
