"""
This module exposes a few PML transforms that can be used to introduce a
complex part to the node coordinates.
"""
from typing import Union

import numpy as np


def _nu(x: float, pml_coords: Union[tuple, float]) -> float:
    """Calculates the through-the-PML coordinate."""
    if isinstance(pml_coords, float):
        x_pml = pml_coords
        if x < x_pml:
            return 0.0
        return x - x_pml

    if isinstance(pml_coords, tuple):
        x_pml_start, x_pml_end = pml_coords
        if x < x_pml_start:
            return x_pml_start - x
        if x > x_pml_end:
            return x - x_pml_end
        return 0.0

    raise TypeError("pml_coords must be either a float or a tuple of floats.")


def x_berenger(
    x: float, pml_coords: Union[tuple, float], k: float, beta: float
) -> complex:
    """Applies the Berenger PML transform to the given coordinate.

    Parameters
    ----------
    x : float
        The coordinate to transform.
    pml_coords : Union[tuple, float]
        The coordinates of the PML. If a float is given, it is assumed
        that the PML starts at x = pml_coords. If a tuple is given, it
        is assumed that the PML extends beyond the given coordinates
        (the domain lies inbetween).
    k : float
        The wavenumber of the wave.
    beta : float
        The Berenger PML stretching coefficient.

    Returns
    -------
    float
        The transformed coordinate.

    """
    nu = _nu(x, pml_coords)
    return x + 1j / k * beta * nu


def x_bermudez(
    x: float,
    pml_coords: Union[tuple, float],
    k: float,
    delta_pml: float,
    eps: float = 1e-9,
) -> complex:
    """Applies the Bermudez PML transform to the given coordinate.

    Parameters
    ----------
    x : float
        The coordinate to transform.
    pml_coords : Union[tuple, float]
        The coordinates of the PML. If a float is given, it is assumed
        that the PML starts at x = pml_coords. If a tuple is given, it
        is assumed that the PML extends beyond the given coordinates
        (the domain lies inbetween).
    k : float
        The wavenumber of the wave.
    delta_pml : float
        The thickness of the PML.
    eps : float
        A small number used to avoid division by zero. Defaults to 1e-9.

    Returns
    -------
    float
        The transformed coordinate.

    """
    nu = _nu(x, pml_coords)
    return x - 1j / k * np.log(1 - nu / (delta_pml + eps))


def x_comsol(
    x: float,
    pml_coords: Union[tuple, float],
    delta_pml: float,
    kind: str = "rational",
    wavelength: float = None,
    curvature: float = 1.0,
    scaling_factor: float = 1.0,
    eps: float = 1e-9,
) -> complex:
    """Applies a PML transform as used in COMSOL.

    Parameters
    ----------
    x : float
        The coordinate to transform.
    pml_coords : Union[tuple, float]
        The coordinates of the PML. If a float is given, it is assumed
        that the PML starts at x = pml_coords. If a tuple is given, it
        is assumed that the PML extends beyond the given coordinates
        (the domain lies inbetween).
    delta_pml : float
        The thickness of the PML.
    kind : str
        The kind of PML transform to apply. Must be either "rational" or
        "polynomial". Defaults to "rational".
    wavelength : float
        A typical wavelength. Defaults to delta_pml.
    curvature : float
        The curvature parameter. Defaults to 1.0.
    scaling_factor : float
        A scaling factor. Defaults to 1.0.
    eps : float
        A small number used to avoid division by zero. Defaults to 1e-9.

    Returns
    -------
    float
        The transformed coordinate.

    Notes
    -----
    From the COMSOL documentation:

    ::

        The rational stretching is designed for propagating waves of
        mixed wavelengths and angles of incidence. The real part of the
        stretching scales the effective PML thickness to a quarter of a
        typical wavelength, while the imaginary part — responsible for
        the attenuation — is stretched out toward infinity. This means
        that provided sufficient mesh resolution, the PML absorbs any
        propagating wave perfectly.

    ::

        The polynomial stretching is generally applicable and most
        appropriate when there is a mix of different wave types in the
        model and you can afford at least 8 mesh elements across the
        PML. Also, compared to the rational stretching, it interferes
        less with the convergence of iterative linear solvers.

    """
    if wavelength is None:
        wavelength = delta_pml

    if kind == "rational":

        def f_i(xi: float) -> complex:
            """The rational stretching function used in COMSOL."""
            real = 1 / (3 * curvature * (1 - xi) + 4)
            imag = -1j / (3 * curvature * (1 - xi + eps))
            return scaling_factor * xi * (real + imag)

    elif kind == "polynomial":

        def f_i(xi: float) -> complex:
            """The polynomial stretching function used in COMSOL."""
            return scaling_factor * xi**curvature * (1 - 1j)

    else:
        raise ValueError("kind must be either 'rational' or 'polynomial'.")

    xi = _nu(x, pml_coords) / delta_pml

    delta_x = wavelength * f_i(xi) - delta_pml * xi

    # TODO: I don't like this, but it's the way I'll make it work for
    # now.
    if isinstance(pml_coords, tuple):
        x_pml_start, x_pml_end = pml_coords
        if x < x_pml_start:
            return x - delta_x.real + 1j * delta_x.imag
        if x > x_pml_end:
            return x + delta_x.real + 1j * delta_x.imag
        return x

    return x + delta_x
