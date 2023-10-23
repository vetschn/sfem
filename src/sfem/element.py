from math import factorial
from typing import Callable

import numpy as np
import numpy.linalg as npla
from numpy.typing import ArrayLike

from sfem.quadrature import GrundmannMoeller


def _one(r: ArrayLike) -> float:
    """Returns one."""
    return 1.0


class SimplexElement:
    """Base class for simplex elements.

    Parameters
    ----------
    nodes : array_like
        The nodes of the element. The node dimensionality must be equal
        to the element dimensionality (e.g. a line element has 1D
        nodes).

    Attributes
    ----------
    num_nodes : int
        The number of nodes of the element. This defines the simplex
        dimensionality.
    dim : int
        The dimensionality of the element.
    name : str
        The name of the element. This is used for ``meshio``.
    nodes : array_like
        The nodes of the element. The node dimensionality must be equal
        to the element dimensionality (e.g. a line element has 1D
        nodes).
    coordinate_matrix : array_like
        The coordinate matrix of the element. This is the matrix
        containing the coordinates of the nodes as columns. The first
        column is the vector of ones.
    alpha : array_like
        The shape function coefficients.
    measure : float
        The measure of the element. This is the determinant of the
        barycentric map.

    """

    num_nodes: int = None
    dim: int = None
    name: str = None
    quadrature_scheme: GrundmannMoeller = None

    def __init__(self, nodes: ArrayLike):
        """Initializes the element."""
        self.nodes = np.asarray(nodes)
        self.coordinate_matrix = np.column_stack([np.ones(self.num_nodes), self.nodes])
        self.alpha = npla.solve(self.coordinate_matrix, np.eye(self.num_nodes)).T
        self.measure = npla.det(self.coordinate_matrix) / factorial(self.dim)

    def contains(self, r: ArrayLike) -> bool:
        """Checks if the given point is contained in the element.

        All shape functions must be non-negative at the point for it to
        be contained in the element.

        Parameters
        ----------
        r : array_like
            The point to check.

        Returns
        -------
        bool
            True if the point is contained in the element, False
            otherwise.

        """
        return np.all(self.alpha @ np.append(1.0, r) >= 0.0)

    def integrate(self, integrand: Callable, dtype: np.dtype = float) -> np.ndarray:
        """Performs cubature over the n-simplex.

        Parameters
        ----------
        integrand : callable
            The integrand to integrate. This must be a function of the
            form integrand(r) where r is the point to evaluate at.
        dtype : np.dtype, optional
            The data type of the integral. The default is float.

        Returns
        -------
        np.ndarray
            The integral of the integrand over the element.

        """
        return self.quadrature_scheme.integrate(
            integrand, nodes=self.nodes, dtype=dtype
        )

    def N(self, node: int, r: ArrayLike) -> float:
        """Evaluates the nodal shape function at the given point.

        Parameters
        ----------
        node : int
            The node of the shape function to evaluate.
        r : array_like
            The point to evaluate at.

        Returns
        -------
        float
            The value of the shape function at the given point.

        """
        if not self.contains(r):
            return 0.0
        return self.alpha[node] @ np.append(1.0, r)

    def grad_N(self, node: int, r: ArrayLike) -> np.ndarray:
        """Evaluates the shape function gradient at the given point.

        Parameters
        ----------
        node : int
            The node of the shape function to evaluate.
        r : array_like
            The point to evaluate at.

        Returns
        -------
        array_like
            The gradient of the shape function at the given point.

        """
        if not self.contains(r):
            return np.zeros(self.dim)
        return self.alpha[node, 1:]

    def _probe(self, integrand: Callable) -> np.ndarray:
        """Probes the integrand to get the output shape.

        You can't know what you're gonna get from the integrand.

        Parameters
        ----------
        integrand : callable
            The integrand to probe.

        Returns
        -------
        np.ndarray
            The output shape of the integrand.

        """
        result = integrand(0, 0, np.zeros(self.dim))
        return np.shape(np.squeeze(result))

    def _matrix(self, integrand: Callable, dtype: np.dtype) -> np.ndarray:
        """Computes the local element matrix for the given integrand.

        Parameters
        ----------
        integrand : callable
            The integrand of the matrix. This must be a function of the
            form integrand(i, j, r) where i and j are the node indices
            and r is the point to evaluate at.
        dtype : np.dtype, optional
            The data type of the matrix.

        Returns
        -------
        np.ndarray
            The local element matrix.

        """
        shape = self._probe(integrand)
        m = np.zeros((self.num_nodes, self.num_nodes, *shape), dtype=dtype)
        for i, j in np.ndindex((self.num_nodes, self.num_nodes)):
            m[i, j] = self.integrate(lambda r: integrand(i, j, r), dtype=dtype)
        return m

    def stiffness_matrix(
        self, dtype: np.dtype, function: Callable = None
    ) -> np.ndarray:
        """Computes the local element stiffness matrix.

        Parameters
        ----------
        dtype : np.dtype
            The data type of the matrix.
        function : callable, optional
            A weighting function for the stiffness matrix. If not given,
            the identity function is used.

        Returns
        -------
        np.ndarray
            The local element stiffness matrix.

        """
        if function is None:
            function = _one

        def integrand(i: int, j: int, r: ArrayLike) -> np.ndarray:
            """The integrand for the stiffness matrix."""
            return function(r) * (self.grad_N(i, r) @ self.grad_N(j, r))

        return self._matrix(integrand, dtype=dtype)

    def mass_matrix(self, dtype: np.dtype, function: Callable = None) -> np.ndarray:
        """Computes the local element mass matrix.

        Parameters
        ----------
        dtype : np.dtype
            The data type of the matrix.
        function : callable, optional
            A weighting function for the mass matrix. If not given, the
            identity function is used.

        Returns
        -------
        np.ndarray
            The local element mass matrix.

        """
        if function is None:
            function = _one

        def integrand(i: int, j: int, r: ArrayLike) -> np.ndarray:
            """The integrand for the mass matrix."""
            return function(r) * (self.N(i, r) * self.N(j, r))

        return self._matrix(integrand, dtype=dtype)

    def gradient_matrix(self, dtype: np.dtype, function: Callable = None) -> np.ndarray:
        """Computes the local element gradient matrix.

        Parameters
        ----------
        dtype : np.dtype
            The data type of the matrix.
        function : callable, optional
            A weighting function for the gradient matrix. If not given,
            the identity function is used.

        Returns
        -------
        array_like
            The local element gradient matrix.

        """
        if function is None:
            function = _one

        def integrand(i: int, j: int, r: ArrayLike) -> np.ndarray:
            """The integrand for the gradient matrix."""
            return function(r) * self.N(i, r) * self.grad_N(j, r)

        return self._matrix(integrand, dtype=dtype)

    def get_matrix(
        self, matrix_type: str, dtype: np.dtype = float, function: Callable = None
    ) -> np.ndarray:
        """Gets the given matrix type for the element.

        Parameters
        ----------
        matrix_type : str
            The type of matrix to get. Can be "stiffness", "mass", or
            "gradient".
        dtype : np.dtype, optional
            The data type of the matrix. The default is float.
        function : callable, optional
            A weighting function for the matrix. If not given, the
            identity function is used.

        Returns
        -------
        np.ndarray
            The matrix for the element.

        """
        if matrix_type == "stiffness":
            return self.stiffness_matrix(dtype, function)
        if matrix_type == "mass":
            return self.mass_matrix(dtype, function)
        if matrix_type == "gradient":
            return self.gradient_matrix(dtype, function)
        raise ValueError(f"Unknown matrix type '{matrix_type}'")
