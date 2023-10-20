from sfem.element import SimplexElement
from sfem.mesh import Mesh
from sfem.quadrature import GrundmannMoeller


class Line(SimplexElement):
    num_nodes = 2
    dim = 1
    name = "line"
    quadrature_scheme = GrundmannMoeller(1)

    def __init__(self, nodes):
        super().__init__(nodes)


class Triangle(SimplexElement):
    num_nodes = 3
    dim = 2
    name = "triangle"
    quadrature_scheme = GrundmannMoeller(2)

    def __init__(self, nodes):
        super().__init__(nodes)


class Tetrahedron(SimplexElement):
    num_nodes = 4
    dim = 3
    name = "tetra"
    quadrature_scheme = GrundmannMoeller(3)

    def __init__(self, nodes):
        super().__init__(nodes)


__all__ = ["Line", "Triangle", "Tetrahedron", "Mesh", "SimplexElement"]
