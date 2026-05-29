import numpy as np

from frtdbn.benchmark import run_grid, summarize_grid


def _tiny_grid():
    return run_grid(
        d=8,
        p=1,
        n_values=[40],
        seeds=[0],
        lbfgs_max_iter=8,
        outer_max_iter=2,
    )


def test_run_grid_covers_loss_by_fusion_2x2():
    rows = _tiny_grid()
    combos = {(r["loss"], r["gamma_w"] > 0 or r["gamma_a"] > 0) for r in rows}
    assert combos == {
        ("gaussian", False),
        ("gaussian", True),
        ("student_t", False),
        ("student_t", True),
    }


def test_run_grid_rows_expose_expected_columns():
    row = _tiny_grid()[0]
    for key in (
        "seed", "n", "d", "p", "nu", "loss", "gamma", "standardize",
        "varsort_raw", "varsort_fit", "auroc_change_w", "auroc_change_a",
        "max_h", "seconds",
    ):
        assert key in row, key
    assert "gamma_w" in row
    assert "gamma_a" in row
    assert "nu" in row
    assert 0.0 <= row["auroc_change_w"] <= 1.0


def test_summarize_grid_computes_mean_and_standard_error():
    rows = [
        {"p": 1, "nu": 5.0, "solver": "admm", "loss": "gaussian", "gamma": 0.0,
         "auroc_change_w": 0.8, "auroc_change_a": 0.5, "max_h": 0.01, "seconds": 0.3},
        {"p": 1, "nu": 5.0, "solver": "admm", "loss": "gaussian", "gamma": 0.0,
         "auroc_change_w": 0.6, "auroc_change_a": 0.5, "max_h": 0.01, "seconds": 0.3},
    ]
    summ = summarize_grid(rows)
    assert len(summ) == 1
    cell = summ[0]
    assert (cell["solver"], cell["loss"], cell["fusion"]) == ("admm", "gaussian", "indep")
    assert cell["p"] == 1
    assert cell["n_fits"] == 2
    assert np.isclose(cell["auroc_w_mean"], 0.7)
    # sample stdev of [0.8, 0.6] is sqrt(0.02); se = stdev / sqrt(2) = 0.1
    assert np.isclose(cell["auroc_w_se"], 0.1, atol=1e-9)


def test_summarize_grid_single_fit_has_zero_se():
    rows = [{"p": 1, "nu": 5.0, "solver": "admm", "loss": "gaussian", "gamma": 0.1,
             "auroc_change_w": 0.9, "auroc_change_a": 0.5, "max_h": 0.0, "seconds": 0.1}]
    cell = summarize_grid(rows)[0]
    assert cell["fusion"] == "fused"
    assert cell["n_fits"] == 1
    assert cell["auroc_w_se"] == 0.0


def test_summarize_grid_separates_solver_loss_fusion_cells():
    rows = run_grid(
        d=8, p=1, n_values=[40], seeds=[0],
        solvers=("lbfgs_smooth", "admm"), lbfgs_max_iter=8, outer_max_iter=2,
    )
    summ = summarize_grid(rows)
    keys = {(c["solver"], c["loss"], c["fusion"]) for c in summ}
    assert len(keys) == 8


def test_run_grid_sweeps_change_edges_and_reports_delta_l1():
    rows = run_grid(
        d=8, p=1, n_values=[40], seeds=[0],
        solvers=("admm",), losses=("student_t",), gammas=(0.04,),
        change_edges_values=(1, 3), lbfgs_max_iter=8, outer_max_iter=2,
    )
    assert {r["change_edges"] for r in rows} == {1, 3}
    assert all("delta_w_l1" in r and r["delta_w_l1"] >= 0 for r in rows)


def test_run_grid_includes_solver_axis():
    rows = run_grid(
        d=8, p=1, n_values=[40], seeds=[0],
        solvers=("lbfgs_smooth", "admm"),
        lbfgs_max_iter=8, outer_max_iter=2,
    )
    assert {r["solver"] for r in rows} == {"lbfgs_smooth", "admm"}
    assert len(rows) == 2 * 2 * 2 * 2  # solver x loss x gamma_w x gamma_a


def test_standardization_drives_var_sortability_to_one_half():
    rows = _tiny_grid()
    # The generator is var-sortable by construction (variance grows down the
    # topological order); standardizing each column flattens that gradient.
    assert max(r["varsort_raw"] for r in rows) > 0.55
    for r in rows:
        assert abs(r["varsort_fit"] - 0.5) < 1e-9
