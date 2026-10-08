"""Known-answer tests for the gridness scorer and grid scale (paper Supplementary 3d)."""
import numpy as np
import pytest

from grid_cells import scores

NBINS = 32
HALF = 1.1


def scorer():
    return scores.GridScorer(NBINS, ((-HALF, HALF), (-HALF, HALF)), scores.paper_mask_parameters())


def lattice_ratemap(spacing_m, angles_deg=(0, 60, 120), phase=(0.13, -0.27)):
    centres = (np.arange(NBINS) + 0.5) / NBINS * 2 * HALF - HALF
    x, y = np.meshgrid(centres, centres, indexing="ij")
    k = 4 * np.pi / (np.sqrt(3) * spacing_m)
    return sum(np.cos(k * ((x - phase[0]) * np.cos(np.deg2rad(a))
                           + (y - phase[1]) * np.sin(np.deg2rad(a))))
               for a in angles_deg)


@pytest.mark.parametrize("spacing", [0.5, 0.7, 0.9])
def test_hexagonal_lattice_is_grid_like(spacing):
    score, _, _ = scorer().get_scores(lattice_ratemap(spacing))
    assert score > 0.37 * 2, score


def test_noise_is_not_grid_like():
    rng = np.random.default_rng(0)
    vals = [scorer().get_scores(rng.normal(size=(NBINS, NBINS)))[0] for _ in range(20)]
    assert np.mean(vals) < 0.2 and np.max(vals) < 0.37


def test_single_gaussian_bump_is_not_grid_like():
    centres = (np.arange(NBINS) + 0.5) / NBINS * 2 * HALF - HALF
    x, y = np.meshgrid(centres, centres, indexing="ij")
    bump = np.exp(-((x - 0.2) ** 2 + (y + 0.1) ** 2) / (2 * 0.2 ** 2))
    assert scorer().get_scores(bump)[0] < 0.37


def test_square_lattice_scores_below_hexagonal():
    sq = lattice_ratemap(0.7, angles_deg=(0, 90))
    hexa = lattice_ratemap(0.7)
    assert scorer().get_scores(sq)[0] < 0.37 < scorer().get_scores(hexa)[0]


def test_grid_scale_tracks_spacing():
    s = scorer()
    bin_m = 2 * HALF / NBINS
    small = scores.grid_scale(s.calculate_sac(lattice_ratemap(0.45)), bin_m)
    large = scores.grid_scale(s.calculate_sac(lattice_ratemap(0.9)), bin_m)
    assert small == pytest.approx(0.45, rel=0.15)
    assert large == pytest.approx(0.9, rel=0.15)
    assert small / large == pytest.approx(0.5, rel=0.15)
