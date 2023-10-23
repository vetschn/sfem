from typing import Callable

import pytest
from numpy.typing import ArrayLike

from sfem import GrundmannMoeller
from sfem.quadrature import QuadratureScheme


@pytest.mark.parametrize(
    "quad, function, nodes, expected",
    [
        pytest.param(GrundmannMoeller(1), lambda x: 1, None, 1.0, id="1D-unit-simplex"),
        pytest.param(
            GrundmannMoeller(2), lambda x: x**2, None, 1 / 6, id="2D-unit-simplex"
        ),
        pytest.param(
            GrundmannMoeller(3), lambda x: x**3, None, 0.025, id="3D-unit-simplex"
        ),
        pytest.param(GrundmannMoeller(1), lambda x: 1, [1.2, 5.7], 4.5, id="1D-nodes"),
        pytest.param(
            GrundmannMoeller(2),
            lambda x: x**2,
            [[0.76466725, 0.22906601], [0.89590049, 0.18793218], [0.89174682, 0.3125]],
            0.006345594189778544,
            id="2D-nodes",
        ),
        pytest.param(
            GrundmannMoeller(3),
            lambda x: x**3,
            [
                [0.55024047, 1.0, 0.72099365],
                [0.5669873, 1.0, 0.58333333],
                [0.66818591, 0.6874595, 0.6874595],
                [0.51710961, 0.79882233, 0.79882233],
            ],
            0.000838036171274712,
            id="3D-nodes",
        ),
    ],
)
def test_integrate(
    quad: QuadratureScheme,
    function: Callable,
    nodes: ArrayLike,
    expected: float,
):
    assert quad.integrate(function, nodes=nodes) == pytest.approx(expected)
