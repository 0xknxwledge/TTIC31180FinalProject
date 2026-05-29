import numpy as np

from frtdbn.viz import (
    change_network,
    plot_change_network,
    plot_regime_heatmaps,
    plot_structure_over_time,
    top_edges,
)

DELTA = np.array([[0.0, 0.5, 0.0], [0.0, 0.0, -0.9], [0.2, 0.0, 0.0]])
NAMES = ["A", "B", "C"]


def test_top_edges_ranks_by_absolute_weight():
    top = top_edges(DELTA, NAMES, k=2)
    assert top[0] == ("B", "C", -0.9)   # largest |weight|
    assert top[1] == ("A", "B", 0.5)


def test_change_network_builds_digraph_of_top_edges():
    g = change_network(DELTA, NAMES, k=2)
    assert g.number_of_edges() == 2
    assert g.has_edge("B", "C")
    assert np.isclose(g["B"]["C"]["weight"], -0.9)


def test_plot_regime_heatmaps_writes_file(tmp_path):
    p = tmp_path / "heat.png"
    plot_regime_heatmaps(np.zeros((3, 3)), DELTA, NAMES, p)
    assert p.exists() and p.stat().st_size > 0


def test_plot_change_network_writes_file(tmp_path):
    p = tmp_path / "net.png"
    plot_change_network(DELTA, NAMES, p, k=2)
    assert p.exists() and p.stat().st_size > 0


def test_plot_structure_over_time_writes_file(tmp_path):
    p = tmp_path / "time.png"
    plot_structure_over_time([np.zeros((3, 3)), DELTA, DELTA], ["t0", "t1", "t2"], NAMES, p)
    assert p.exists() and p.stat().st_size > 0
