from multiprocessing import Pool
from typing import Callable

import meshio
import numpy as np
import pyvista as pv
import vtk
from bsparse import sparse
from numpy.lib import recfunctions as rfn
from numpy.typing import ArrayLike

from sfem.element import SimplexElement

# The VTK cell types corresponding to the meshio cell types.
VTK_MESH_TYPES = {
    "line": vtk.VTK_LINE,
    "triangle": vtk.VTK_TRIANGLE,
    "tetra": vtk.VTK_TETRA,
}


class Mesh:
    """A finite element mesh.

    Parameters
    ----------
    nodes : array_like
        The nodes of the mesh. Each entry is interpreted as a point in
        n-dimensional space.
    elements : array_like
        The elements of the mesh. Each entry is a list of node indices
        that define the element.
    etype : SimplexElement
        The type of element making up the mesh. Must be a subclass of
        ``SimplexElement``. Mixed-element meshes are not supported.

    Attributes
    ----------
    nodes : array_like
        The nodes of the mesh. Each entry is interpreted as a point in
        n-dimensional space.
    elements : array_like
        The elements of the mesh. Each entry is a list of node indices
        that define the element.
    num_nodes : int
        The number of nodes in the mesh.
    num_elements : int
        The number of elements in the mesh.
    etype : SimplexElement
        The type of element making up the mesh. A subclass of
        ``SimplexElement``.
    typed_elements : list[SimplexElement]
        The elements of the mesh, each of type ``etype``.

    """

    def __init__(
        self, nodes: ArrayLike, elements: ArrayLike, etype: SimplexElement
    ) -> None:
        """Initializes a finite element mesh."""
        self.nodes = np.asarray(nodes)
        self.elements = np.asarray(elements)
        self.num_nodes = len(nodes)
        self.num_elements = len(elements)
        self.etype = etype
        self.typed_elements = [etype(e) for e in self.nodes[self.elements]]

    def __str__(self) -> str:
        """Returns a string representation of the mesh."""
        return (
            f"Mesh({self.num_nodes} nodes, "
            f"{self.num_elements} elements of type {self.etype.__name__})"
        )

    def __repr__(self) -> str:
        """Returns a string representation of the mesh."""
        return str(self)

    def sort_nodes(self, **argsort_kwargs: dict) -> None:
        """Sorts the nodes of the mesh according to their position.

        The matrices assembled on this mesh acquire a block structure.

        Parameters
        ----------
        argsort_kwargs : dict
            Keyword arguments passed to ``np.argsort``. The default
            values are ``axis=0`` and ``order=["x", "y", "z"]``.

        """
        node_dtype = self.nodes.dtype
        if "axis" not in argsort_kwargs:
            argsort_kwargs["axis"] = 0
        if self.etype.dim == 1:
            if "order" not in argsort_kwargs:
                argsort_kwargs["order"] = ["x"]
            nodes = rfn.unstructured_to_structured(
                self.nodes,
                dtype=np.dtype([("x", node_dtype)]),
            )
        elif self.etype.dim == 2:
            if "order" not in argsort_kwargs:
                argsort_kwargs["order"] = ["x", "y"]

            nodes = rfn.unstructured_to_structured(
                self.nodes,
                dtype=np.dtype([("x", node_dtype), ("y", node_dtype)]),
            )
        else:
            if "order" not in argsort_kwargs:
                argsort_kwargs["order"] = ["x", "y", "z"]

            nodes = rfn.unstructured_to_structured(
                self.nodes,
                dtype=np.dtype(
                    [("x", node_dtype), ("y", node_dtype), ("z", node_dtype)]
                ),
            )
        sort_index = np.argsort(nodes, **argsort_kwargs)
        nodes = nodes[sort_index]
        nodes = rfn.structured_to_unstructured(nodes)

        elements = np.argsort(sort_index)[self.elements]

        self.nodes = nodes
        self.elements = elements
        self.typed_elements = [self.etype(e) for e in nodes[elements]]

    def find_node(self, point: ArrayLike, tol: float = 1e-12) -> int:
        """Returns the index of the node closest to the given point.

        Parameters
        ----------
        point : array_like
            The point to search for.
        tol : float, optional
            The tolerance for the search, by default 1e-12.

        Returns
        -------
        int
            The index of the node closest to the given point.

        """
        if self.etype.dim == 1:
            in_tolerance = np.abs(self.nodes - point) < tol
        else:
            in_tolerance = np.linalg.norm(self.nodes - point, axis=1) < tol
        if not np.any(in_tolerance):
            raise ValueError("No node found.")
        return np.argwhere(in_tolerance)[0][0]

    def _get_matrix_shapes(self, dof_shape: tuple) -> tuple[tuple, tuple]:
        """Calculates the shape of the global and local matrices.

        Parameters
        ----------
        dof_shape : tuple
            The shape of the degrees of freedom for each node.

        Returns
        -------
        tuple[tuple, tuple]
            The shape of the global and local matrices.

        """
        if len(dof_shape) == 0:
            global_shape = (self.num_nodes, self.num_nodes)
            local_shape = (self.etype.num_nodes, self.etype.num_nodes)
            return global_shape, local_shape

        if len(dof_shape) == 1:
            raise ValueError("Cannot assemble matrix with vector entries (yet).")

        if len(dof_shape) == 2:
            if not dof_shape[0] == dof_shape[1]:
                raise ValueError("Cannot assemble matrix with non-square entries.")

            global_shape = (
                self.num_nodes * dof_shape[0],
                self.num_nodes * dof_shape[0],
            )
            local_shape = (
                self.etype.num_nodes * dof_shape[0],
                self.etype.num_nodes * dof_shape[0],
            )
            return global_shape, local_shape

        raise ValueError("Cannot assemble matrix with more than two dimensions.")

    def _get_global_indices(self, element: ArrayLike, dof_shape: tuple) -> np.ndarray:
        """Calculates the global indices of the degrees of freedom.

        Parameters
        ----------
        dof_shape : tuple
            The shape of the degrees of freedom for each node.

        Returns
        -------
        np.ndarray
            The global indices of the degrees of freedom.

        """
        element = np.asarray(element)
        offsets = np.tile(np.arange(dof_shape[0]), len(element))
        return np.repeat(element, dof_shape[0]) * dof_shape[0] + offsets

    def _reshape_local_matrix(
        self, local_matrix: np.ndarray, local_shape: tuple
    ) -> np.ndarray:
        """Reshapes the local matrix to the correct shape.

        Parameters
        ----------
        local_matrix : np.ndarray
            The local matrix.
        dof_shape : tuple
            The shape of the degrees of freedom for each node.

        Returns
        -------
        np.ndarray
            The reshaped local matrix.

        """
        # Reshape from hell. There is probably a better way to do this.
        intermediate_shape = local_matrix.shape[0], local_shape[1]
        intermediate = local_matrix.reshape(*intermediate_shape, -1, order="F")
        return intermediate.reshape(*local_shape)

    def assemble_matrix(
        self,
        matrix_type: str,
        dtype: type = float,
        function: Callable = None,
        parallel: bool = True,
    ) -> sparse.COO:
        """Assembles the specified matrix type for the mesh.

        Parameters
        ----------
        matrix_type : str
            The type of matrix to assemble. Can be "stiffness" or
            "mass", "lumped_mass".
        function : callable, optional
            A spatial weighting function for the matrix. If not given,
            the identity function is used.
        dtype : type, optional
            The data type of the matrix, by default float.
        parallel : bool, optional
            Whether to use multiprocessing to assemble the matrix in
            parallel, by default True.

        Returns
        -------
        sparse.COO
            The assembled matrix in coordinate format.

        """
        # TODO: Add support for gradient matrices.
        if matrix_type not in ("stiffness", "mass", "lumped_mass"):
            raise ValueError(
                f"Unknown matrix type '{matrix_type}'. "
                "Must be 'stiffness', 'mass', or 'lumped_mass'."
            )

        if matrix_type == "lumped_mass":
            if function is None:

                def function(x):
                    return 1

            return sparse.diag(function(self.nodes))

        global get_local_matrix

        def get_local_matrix(element: SimplexElement) -> np.ndarray:
            """Returns the local matrix for the given element."""
            return element.get_matrix(matrix_type, dtype, function)

        if parallel:
            with Pool() as pool:
                local_matrices = pool.map(get_local_matrix, self.typed_elements)
        else:
            local_matrices = list(map(get_local_matrix, self.typed_elements))

        dof_shape = np.shape(local_matrices[0][0, 0])
        dof_shape = (1, 1) if len(dof_shape) == 0 else dof_shape
        global_shape, local_shape = self._get_matrix_shapes(dof_shape)

        coords = []
        data = []
        for element, local_matrix in zip(self.elements, local_matrices):
            indices = self._get_global_indices(element, dof_shape)
            coords.extend([(i, j) for j in indices for i in indices])
            local_matrix = self._reshape_local_matrix(local_matrix, local_shape)
            data.extend(local_matrix.ravel())

        coords, inverse = np.unique(coords, axis=0, return_inverse=True)

        data = np.array(data, dtype=dtype)
        if np.iscomplexobj(data):
            # NumPy's bincount does not support complex numbers.
            data_real = np.bincount(inverse, weights=data.real)
            data_imag = np.bincount(inverse, weights=data.imag)
            data = data_real + 1j * data_imag
        else:
            data = np.bincount(inverse, weights=data)

        ind = np.nonzero(data)
        matrix = sparse.COO(
            coords[:, 0][ind],
            coords[:, 1][ind],
            data[ind],
            shape=global_shape,
            dtype=dtype,
        )

        return matrix

    def view_quality(
        self,
        pl: pv.Plotter = None,
        notebook: bool = False,
        measure: str = "scaled_jacobian",
    ) -> pv.Plotter:
        """Visualizes the mesh quality using pyvista.

        Parameters
        ----------
        pl : pv.Plotter, optional
            The plotter to use, by default None. If None, a new plotter
            is created.
        **plotter_kwargs
            Keyword arguments to pass to the plotter.

        """
        display = False
        if pl is None:
            pl = pv.Plotter(notebook=notebook)
            display = True

        # If the mesh is not 3D, we need to add a (second and) third
        # dimension.
        if self.nodes.ndim == 1:
            nodes = np.zeros((self.num_nodes, 3))
            nodes[:, 0] = self.nodes
        elif self.nodes.shape[1] == 2:
            nodes = np.zeros((self.num_nodes, 3))
            nodes[:, :2] = self.nodes
        else:
            nodes = self.nodes

        ugrid = pv.UnstructuredGrid(
            {VTK_MESH_TYPES[self.etype.name]: self.elements},
            nodes,
        )
        qual = ugrid.compute_cell_quality(quality_measure=measure)
        pl.add_mesh(
            qual.extract_all_edges(),
            scalars="CellQuality",
            cmap="RdYlGn",
            show_edges=True,
            opacity=0.75,
            clim=[0, 1],
        )
        pl.add_axes(xlabel="x", ylabel="y", zlabel="z")
        pl.add_bounding_box()

        if display:
            pl.show()

        return pl

    def view(self, pl: pv.Plotter = None, notebook: bool = False) -> pv.Plotter:
        """Visualizes the mesh.

        Parameters
        ----------
        pl : pv.Plotter, optional
            The plotter to use, by default None. If None, a new plotter
            is created.
        **plotter_kwargs
            Keyword arguments to pass to the plotter.

        Returns
        -------
        pv.Plotter
            The plotter object.

        """
        display = False
        if pl is None:
            pl = pv.Plotter(notebook=notebook)
            display = True

        # If the mesh is not 3D, we need to add a (second and) third
        # dimension.
        if self.nodes.ndim == 1:
            nodes = np.zeros((self.num_nodes, 3))
            nodes[:, 0] = self.nodes
        elif self.nodes.shape[1] == 2:
            nodes = np.zeros((self.num_nodes, 3))
            nodes[:, :2] = self.nodes
        else:
            nodes = self.nodes

        ugrid = pv.UnstructuredGrid(
            {VTK_MESH_TYPES[self.etype.name]: self.elements},
            nodes,
        )

        pl.add_mesh(ugrid, color=True, opacity=0.1, show_edges=True)
        if self.nodes.shape[1] == 3:
            pl.add_mesh(ugrid.extract_all_edges(), line_width=1)
        pl.add_axes(xlabel="x", ylabel="y", zlabel="z")
        pl.add_bounding_box()

        if display:
            pl.show()

        return pl

    @classmethod
    def from_meshio(cls, mesh: meshio.Mesh, etype: SimplexElement) -> "Mesh":
        """Creates a finite element mesh from a meshio mesh.

        Parameters
        ----------
        mesh : meshio.Mesh
            The mesh to create the finite element mesh from.
        etype : SimplexElement
            The type of element making up the mesh. Must be a subclass
            of ``SimplexElement``. Mixed-element meshes are not
            supported.

        Returns
        -------
        Mesh
            The finite element mesh.

        """
        return cls(mesh.points, mesh.cells_dict[etype.name], etype)
