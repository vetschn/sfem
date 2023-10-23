import meshio

from sfem import Line, Mesh, Tetrahedron, Triangle


def test_tetrahedral(datadir):
    """Tests tetrahedral meshes."""
    meshio_mesh = meshio.read(datadir / "tetra.msh", "gmsh")
    mesh = Mesh.from_meshio(meshio_mesh, Tetrahedron)

    assert mesh.find_node([0.5, 0.5, 0.5]) == 272
    mesh.sort_nodes()
    assert mesh.find_node([0.5, 0.5, 0.5]) == 173

    stiffness_matrix = mesh.assemble_matrix("stiffness")
    assert stiffness_matrix.shape == (341, 341)
    assert stiffness_matrix.nnz == 3841


def test_triangular(datadir):
    """Tests triangular meshes."""
    meshio_mesh = meshio.read(datadir / "triangle.msh", "gmsh")
    mesh = Mesh.from_meshio(meshio_mesh, Triangle)

    assert mesh.find_node([1.0, 0.25]) == 12
    mesh.sort_nodes()
    assert mesh.find_node([1.0, 0.25]) == 91

    stiffness_matrix = mesh.assemble_matrix("stiffness")
    assert stiffness_matrix.shape == (mesh.num_nodes, mesh.num_nodes)
    assert stiffness_matrix.nnz == 616


def test_linear(datadir):
    meshio_mesh = meshio.read(datadir / "line.msh", "gmsh")
    mesh = Mesh.from_meshio(meshio_mesh, Line)

    assert mesh.find_node([0.5]) == 6
    mesh.sort_nodes()
    assert mesh.find_node([0.5]) == 5

    stiffness_matrix = mesh.assemble_matrix("stiffness")
    assert stiffness_matrix.shape == (mesh.num_nodes, mesh.num_nodes)
    assert stiffness_matrix.nnz == 31
