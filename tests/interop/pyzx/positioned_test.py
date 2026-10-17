from collections.abc import Callable
from fractions import Fraction

import pytest
import pyzx as zx
from pyzx import VertexType
from pyzx.graph.graph_s import GraphS

from tqec.computation.block_graph import BlockGraph
from tqec.gallery import cnot, cz, move_rotation, three_cnots
from tqec.interop.pyzx._testing import make_positioned_zx_graph
from tqec.interop.pyzx.positioned import PositionedZX
from tqec.utils.exceptions import TQECError
from tqec.utils.position import Position3D


def test_positions_not_specified() -> None:
    g = GraphS()
    g.add_vertex(VertexType.Z)
    with pytest.raises(TQECError, match=r".* do not match.*"):
        PositionedZX(g, {})


def test_positions_not_neighbors() -> None:
    with pytest.raises(TQECError, match=r".* must be neighbors.*"):
        make_positioned_zx_graph(
            vertex_types=[VertexType.Z, VertexType.Z],
            positions=[Position3D(0, 0, 0), Position3D(2, 0, 0)],
            edges=[(0, 1)],
            hadamard_edges=[False],
        )


def test_unsupported_vertex_type() -> None:
    with pytest.raises(TQECError, match=r"Unsupported vertex type and phase.*"):
        make_positioned_zx_graph(
            vertex_types=[VertexType.Z],
            positions=[Position3D(0, 0, 0)],
            phases={0: Fraction(1, 3)},
        )


def test_boundary_not_dangle() -> None:
    with pytest.raises(TQECError, match=r".* must be dangling.*"):
        make_positioned_zx_graph(
            vertex_types=[VertexType.BOUNDARY, VertexType.Z, VertexType.X],
            positions=[Position3D(0, 0, 0), Position3D(1, 0, 0), Position3D(0, 1, 0)],
            edges=[(0, 1), (0, 2)],
            hadamard_edges=[False, False],
            inputs=(0,),
        )


def test_y_connect_in_space() -> None:
    with pytest.raises(TQECError, match=r".* must connect to the time direction.*"):
        make_positioned_zx_graph(
            vertex_types=[VertexType.Z, VertexType.Z],
            positions=[Position3D(0, 0, 0), Position3D(1, 0, 0)],
            phases={1: Fraction(1, 2)},
            edges=[(0, 1)],
            hadamard_edges=[False],
        )


def test_3d_corner() -> None:
    with pytest.raises(TQECError, match=r".* has a 3D corner.*"):
        make_positioned_zx_graph(
            vertex_types=[VertexType.Z] * 4,
            positions=[
                Position3D(0, 0, 0),
                Position3D(1, 0, 0),
                Position3D(0, 1, 0),
                Position3D(0, 0, 1),
            ],
            edges=[(0, 1), (0, 2), (0, 3)],
            hadamard_edges=[False] * 3,
        )


@pytest.mark.parametrize(
    "build_graph, inputs, outputs, qasm",
    [
        pytest.param(cnot, (0, 6), (3, 9), "qreg q[2];\ncx q[0], q[1];", id="cnot"),
        pytest.param(cz, (0, 3), (2, 5), "qreg q[2];\ncz q[0], q[1];", id="cz"),
        pytest.param(move_rotation, (0,), (4,), "qreg q[1];", id="move_rotation"),
        pytest.param(
            three_cnots,
            (1, 4, 8),
            (0, 7, 11),
            "qreg q[3];\ncx q[0], q[1];\ncx q[1], q[2];\ncx q[0], q[2];",
            id="three_cnots",
        ),
    ],
)
def test_from_block_graph_matches_circuit(
    build_graph: Callable[[], BlockGraph],
    inputs: tuple[int, ...],
    outputs: tuple[int, ...],
    qasm: str,
) -> None:
    g = build_graph().to_zx_graph().g
    g.set_inputs(inputs)
    g.set_outputs(outputs)
    assert zx.compare_tensors(zx.qasm(qasm), g)
