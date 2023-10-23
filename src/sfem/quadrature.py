from fractions import Fraction
from functools import reduce
from math import factorial
from operator import add
from typing import Callable, Generator

import numpy as np
import numpy.linalg as npla
from numpy.typing import ArrayLike
from pytools import (
    generate_decreasing_nonnegative_tuples_summing_to,
    generate_unique_permutations,
)


def _integer_ratio(numerator: int, denominator: int) -> tuple:
    """Returns the integer ratio of two numbers.

    Parameters
    ----------
    numerator : int
        The numerator.
    denominator : int
        The denominator.

    Returns
    -------
    tuple
        The integer ratio of the numerator and denominator.

    """
    return Fraction(numerator, denominator).as_integer_ratio()


class QuadratureScheme:
    """Basic interface for a quadrature scheme.

    Parameters
    ----------
    points : array_like
        The quadrature points.
    weights : array_like
        The quadrature weights.

    Attributes
    ----------
    points : np.ndarray
        The quadrature points.
    weights : np.ndarray
        The quadrature weights.

    """

    def __init__(self, points: ArrayLike, weights: ArrayLike):
        """Initializes a quadrature rule."""
        self.points = np.array(points).squeeze()
        self.weights = np.array(weights)

    def _transform_scheme(self, nodes: ArrayLike) -> tuple:
        """Transforms the quadrature scheme from the standard simplex.

        Parameters
        ----------
        nodes : array_like
            The nodes of the element.

        Returns
        -------
        tuple
            The transformed quadrature points and weights.

        """
        nodes = np.array(nodes)
        n = nodes.shape[0] - 1
        coordinate_matrix = np.column_stack([np.ones(n + 1), nodes])
        measure = npla.det(coordinate_matrix)  # Factorial already in weights.
        weights = self.weights * measure

        inv_unit_coordinate_matrix = np.block(
            [[np.zeros(n), 1], [np.eye(n), -np.ones((n, 1))]]
        )
        transform_matrix = (inv_unit_coordinate_matrix @ coordinate_matrix).T

        points = [transform_matrix @ np.block([1, p]) for p in self.points]
        points = np.array(points)[:, 1:]
        return points, weights

    def integrate(
        self, integrand: Callable, nodes: ArrayLike = None, dtype: np.dtype = float
    ):
        """Evaluates the given function over the points.

        The integrand has to be a function that accepts a single
        argument, which is a point in the integration domain. It does
        NOT have to be a vectorized function.

        Parameters
        ----------
        integrand : Callable
            The function to be evaluated. This function must accept a
            single argument, which is a point in the integration domain.
            As long as the function's return value is summable, the
            quadrature rule can be used to integrate it.
        nodes : array_like, optional
            The nodes of the element. If given, the integrand will be
            evaluated over the simplex spanned by the nodes. If not
            given, the integrand will be evaluated over the standard
            simplex.
        dtype : np.dtype, optional
            The data type of the integral. The default is float.

        """
        points, weights = self.points, self.weights
        if nodes is not None:
            points, weights = self._transform_scheme(nodes)

        function_values = [integrand(point) for point in points]
        return np.sum([w * f for w, f in zip(weights, function_values)], dtype=dtype)


class GrundmannMoeller(QuadratureScheme):
    """Grundmann-Moeller quadrature on an n-simplex.

    This cubature rule has both negative and positive weights. It is
    exact for polynomials up to degree `2 * s + 1`.

    The integration domain is the unit simplex.

    Parameters
    ----------
    n : int
        The dimension of the integration domain, i.e. of the simplex.
    s : int
        The order of the quadrature rule.

    Attributes
    ----------
    points : np.ndarray
        The quadrature points.
    weights : np.ndarray
        The quadrature weights.

    """

    def __init__(self, n: int, s: int = 1):
        """Initializes a Grundmann-Moeller quadrature scheme.

        The formula implemented here can be seen in (4.1) of [1]_.

        References
        ----------
        .. [1] A. Grundmann and H. M. Moller, “Invariant Integration
           Formulas for the n-Simplex by Combinatorial Methods,” SIAM
           Journal on Numerical Analysis, vol. 15, no. 2, pp. 282-290,
           1978.

        """
        if n < 1:
            raise ValueError("n must be at least 1.")

        if s < 0 or s % 2 == 0:
            raise ValueError("s must be positive and odd.")

        points_and_weights = GrundmannMoeller._points_and_weights(n, s)

        nodes = np.eye(n + 1, n)
        points = []
        weights = []
        for point, weight in points_and_weights.items():
            points.append(reduce(add, (a / b * v for (a, b), v in zip(point, nodes))))
            weights.append(weight)

        super().__init__(points, weights)

    @staticmethod
    def _points_and_weights(n: int, s: int) -> dict:
        """Calculates the points and weights of the quadrature rule."""
        d = 2 * s + 1  # The degree of the quadrature rule.
        points_and_weights = {}
        for i in range(s + 1):
            # Calculate the weight.
            numerator = (d + n - 2 * i) ** d
            denominator = factorial(i) * factorial(d + n - i)
            weight = (-1) ** i * 2 ** (-2 * s) * numerator / denominator

            for beta in GrundmannMoeller._beta(n, s - i):
                # Calculate the quadrature point.
                point = tuple(
                    _integer_ratio(2 * beta_i + 1, d + n - 2 * i) for beta_i in beta
                )
                points_and_weights[point] = points_and_weights.get(point, 0.0) + weight

        return points_and_weights

    @staticmethod
    def _beta(n: int, modulus: int) -> Generator[tuple, None, None]:
        """Generates all unique integer-tuples of some modulus.

        Parameters
        ----------
        n : int
            The length of the tuples.
        modulus : int
            The modulus of the tuples.

        Yields
        ------
        tuple
            A tuple of nonnegative integers of the given modulus.
        """
        for t in generate_decreasing_nonnegative_tuples_summing_to(modulus, n + 1):
            yield from generate_unique_permutations(t)
