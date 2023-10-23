import numpy as np
import pytest
from numpy.typing import ArrayLike

from sfem import Line, SimplexElement, Tetrahedron, Triangle


@pytest.mark.parametrize(
    "element, nodes, measure",
    [
        pytest.param(Line, [-1.0, 1.0], 2.0, id="line"),
        pytest.param(
            Triangle, [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]], 2.0, id="triangle"
        ),
        pytest.param(
            Tetrahedron,
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            1 / 6,
            id="tetra",
        ),
    ],
)
def test_creation(element: SimplexElement, nodes: ArrayLike, measure: float):
    """Tests the creation of simplex elements."""
    e = element(nodes)
    assert e.measure == measure
    assert np.allclose(e.nodes, nodes)


@pytest.mark.parametrize(
    "element, nodes, point_inside, point_outside",
    [
        pytest.param(Line, [-1.0, 1.0], 0.33, 1.1, id="line"),
        pytest.param(
            Triangle,
            [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]],
            [-0.1, -0.1],
            [5.2, 10.1],
            id="triangle",
        ),
        pytest.param(
            Tetrahedron,
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [0.1, 0.3, 0.4],
            [5.2, 10.1, 0.1],
            id="tetra",
        ),
    ],
)
def test_contains(
    element: SimplexElement, nodes, point_inside: ArrayLike, point_outside: ArrayLike
):
    """Tests the contains method of simplex elements."""
    e = element(nodes)
    assert e.contains(point_inside)
    assert not e.contains(point_outside)


@pytest.mark.parametrize(
    "element, nodes, point, N_value, grad_N_value",
    [
        pytest.param(
            Line,
            [-1.0, 1.0],
            0.33,
            (0.335, 0.665),
            ([-0.5], [0.5]),
            id="line",
        ),
        pytest.param(
            Triangle,
            [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]],
            [-0.1, -0.1],
            (0.1, 0.45, 0.45),
            ([-0.5, -0.5], [0.5, 0.0], [0.0, 0.5]),
            id="triangle",
        ),
        pytest.param(
            Tetrahedron,
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [0.1, 0.3, 0.4],
            (0.2, 0.1, 0.3, 0.4),
            ([-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
            id="tetra",
        ),
    ],
)
def test_shape_functions(
    element: SimplexElement,
    nodes: ArrayLike,
    point: ArrayLike,
    N_value: float,
    grad_N_value: np.ndarray,
):
    """Tests the shape functions and their gradients."""
    e = element(nodes)
    for i, (N, grad_N) in enumerate(zip(N_value, grad_N_value)):
        assert np.isclose(e.N(i, point), N)
        assert np.allclose(e.grad_N(i, point), grad_N)


@pytest.mark.parametrize(
    "element, nodes, matrices",
    [
        pytest.param(
            Line,
            [-1.0, 1.0],
            {
                "stiffness": [[0.5, -0.5], [-0.5, 0.5]],
                "mass": [[0.66666667, 0.33333333], [0.33333333, 0.66666667]],
            },
            id="line",
        ),
        pytest.param(
            Triangle,
            [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]],
            {
                "stiffness": [[1.0, -0.5, -0.5], [-0.5, 0.5, 0.0], [-0.5, 0.0, 0.5]],
                "mass": [
                    [0.33333333, 0.16666667, 0.16666667],
                    [0.16666667, 0.33333333, 0.16666667],
                    [0.16666667, 0.16666667, 0.33333333],
                ],
            },
            id="triangle",
        ),
        pytest.param(
            Tetrahedron,
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            {
                "stiffness": [
                    [0.5, -0.16666667, -0.16666667, -0.16666667],
                    [-0.16666667, 0.16666667, 0.0, 0.0],
                    [-0.16666667, 0.0, 0.16666667, 0.0],
                    [-0.16666667, 0.00, 0.0, 0.16666667],
                ],
                "mass": [
                    [0.01666667, 0.00833333, 0.00833333, 0.00833333],
                    [0.00833333, 0.01666667, 0.00833333, 0.00833333],
                    [0.00833333, 0.00833333, 0.01666667, 0.00833333],
                    [0.00833333, 0.00833333, 0.00833333, 0.01666667],
                ],
            },
            id="tetra",
        ),
    ],
)
def test_matrix(element: SimplexElement, nodes: ArrayLike, matrices: dict):
    """Tests the matrix assembly."""
    e = element(nodes)
    for name, matrix in matrices.items():
        assert np.allclose(e.get_matrix(name), matrix)
