"""Training targets, the pixel mask and the token stats (docs/conventions.md section 5)."""

import pytest

torch = pytest.importorskip("torch")

from dawn.data.color import linear_to_lab  # noqa: E402
from dawn.data.resample import area_downsample  # noqa: E402
from dawn.flow.render import RenderSettings, render  # noqa: E402
from dawn.flow.targets import (  # noqa: E402
    FitSettings,
    build_targets,
    compute_token_stats,
    load_or_compute_stats,
    pixel_mask,
    prepare_batch,
)
from dawn.grid.normalize import standardize  # noqa: E402
from tests.flow_helpers import TINY_INPUT, load_configs, synthetic_batch, tiny_layout  # noqa: E402

POLICY = load_configs()["pairs"]
FIT = FitSettings(ridge_alpha=1e-3, refine_steps=1)


def _prepared(batch=2, **kw):
    return prepare_batch(synthetic_batch(batch, **kw), TINY_INPUT, POLICY)


def test_pixel_mask_is_valid_and_not_moving_for_box_pairs_and_not_vehicles_for_fallback():
    raw = synthetic_batch(2, fallback=[False, True])
    mask = pixel_mask(raw, POLICY)
    assert mask.shape == (2, 1, *TINY_INPUT) and mask.dtype == torch.bool
    assert torch.equal(mask[0], raw["valid"][0] & ~raw["moving"][0])
    assert torch.equal(mask[1], raw["valid"][1] & ~raw["vehicles"][1])
    assert mask[0].sum() > mask[1].sum()  # the vehicles mask is the bigger one


def test_prepare_batch_downsamples_by_area_and_turns_the_mask_into_fractions():
    raw = synthetic_batch(2, size=(64, 80))
    half = prepare_batch(raw, (32, 40), POLICY)
    assert half["thermal"].shape == (2, 1, 32, 40) and half["rgb"].shape == (2, 3, 32, 40)
    assert torch.allclose(half["thermal"], area_downsample(raw["thermal"], (32, 40)))
    assert torch.allclose(half["rgb"], area_downsample(raw["rgb"], (32, 40)))  # linear light
    expected = area_downsample(pixel_mask(raw, POLICY).float(), (32, 40))
    assert torch.allclose(half["mask"], expected)
    assert ((half["mask"] > 0) & (half["mask"] < 1)).any()  # fractions, not thresholded
    full = prepare_batch(raw, (64, 80), POLICY)  # at the stored size nothing is resampled
    assert torch.equal(full["thermal"], raw["thermal"]) and full["mask"].dtype == torch.float32
    assert full["image_id"] == raw["image_id"] and full["fallback"].dtype == torch.bool


def test_prepare_batch_refuses_to_upsample():
    with pytest.raises(ValueError, match="larger"):
        prepare_batch(synthetic_batch(1, size=(32, 40)), (64, 80), POLICY)


def test_targets_have_the_layout_shapes_and_the_true_lightness():
    layout, batch = tiny_layout(), _prepared()
    t = build_targets(batch["thermal"], batch["rgb"], batch["mask"], layout, FIT)
    gy, gx = layout.grid
    assert t.parts["grid"].shape == (2, 2, layout.num_bins, gy, gx)
    assert t.parts["ab"].shape == (2, 2, *layout.color_size) and "corr" not in t.parts
    assert t.node_weights.shape == (2, layout.num_bins, gy, gx)
    assert t.fraction.shape == (2, 1, *layout.color_size)
    assert torch.allclose(t.lab, linear_to_lab(batch["rgb"]), atol=1e-4)
    assert torch.isclose(t.node_weights.sum(), batch["mask"].sum(), rtol=1e-4)
    # L = L*/100 is an exact affine function of thermal in the synthetic data: the oracle gets it.
    err = ((t.L_grid - t.lab[:, :1] / 100) * batch["mask"]).abs().sum() / batch["mask"].sum()
    assert err < 2e-3


def test_masked_pixels_have_no_influence_on_any_target():
    layout, batch = tiny_layout(True), _prepared()
    clean = build_targets(batch["thermal"], batch["rgb"], batch["mask"], layout, FIT)
    rgb = batch["rgb"].clone()
    rgb[batch["mask"].expand_as(rgb) == 0] = 0.9  # recolour every fully masked pixel
    dirty = build_targets(batch["thermal"], rgb, batch["mask"], layout, FIT)
    for name in clean.parts:
        assert torch.allclose(clean.parts[name], dirty.parts[name], atol=1e-5), name


def test_the_colour_target_is_the_masked_area_mean_of_ab_with_a_neutral_fill():
    layout, batch = tiny_layout(), _prepared()
    t = build_targets(batch["thermal"], batch["rgb"], batch["mask"], layout, FIT)
    ab = linear_to_lab(batch["rgb"])[:, 1:]
    num = area_downsample(ab * batch["mask"], layout.color_size)
    frac = area_downsample(batch["mask"], layout.color_size)
    covered = frac > 0
    assert torch.allclose(t.fraction, frac)
    assert torch.allclose(
        t.parts["ab"][covered.expand_as(num)],
        (num / frac.clamp(min=1e-9))[covered.expand_as(num)],
        atol=1e-4,
    )
    assert (~covered).any(), "the synthetic masks should fully cover at least one colour pixel"
    assert (t.parts["ab"][(~covered).expand_as(num)] == 0).all()  # neutral where there is no data


def test_the_correction_target_is_the_masked_mean_of_true_minus_fitted_lightness():
    layout, batch = tiny_layout(True), _prepared()
    t = build_targets(batch["thermal"], batch["rgb"], batch["mask"], layout, FIT)
    assert t.parts["corr"].shape == (2, 1, *layout.color_size)
    residual = t.lab[:, :1] / 100 - t.L_grid
    num = area_downsample(residual * batch["mask"], layout.color_size)
    frac = area_downsample(batch["mask"], layout.color_size)
    expected = torch.where(frac > 0, num / frac.clamp(min=1e-9), torch.zeros_like(num))
    assert torch.allclose(t.parts["corr"], expected, atol=1e-5)
    # Make the lightness unrepresentable by the grid (one bright blob) and the correction wakes up.
    rgb = batch["rgb"].clone()
    rgb[..., 20:36, 20:44] = 0.9
    blob = build_targets(batch["thermal"], rgb, batch["mask"], layout, FIT)
    assert blob.parts["corr"].abs().max() > 10 * t.parts["corr"].abs().max()


def test_oracle_parts_reproduce_the_target_when_the_model_is_exact():
    """Fit, render, compare: lightness exact by construction, colour limited by its resolution."""
    layout, batch = tiny_layout(), _prepared()
    t = build_targets(batch["thermal"], batch["rgb"], batch["mask"], layout, FIT)
    settings = RenderSettings(layout, bgu_grid=(4, 8, 10))
    out = render(t.parts, batch["thermal"], TINY_INPUT, settings)
    m = batch["mask"]
    l_err = ((out["lab"][:, 0] - t.lab[:, 0]).abs() * m[:, 0]).sum() / m.sum()
    assert l_err < 0.3  # L* units out of 100
    ab_err = ((out["lab"][:, 1:] - t.lab[:, 1:]).abs() * m).sum() / (2 * m.sum())
    assert ab_err < 3.0  # the a*b* here vary smoothly over several colour pixels


def test_pack_and_standardize_give_z1_with_unit_scale_stats():
    layout, batch = tiny_layout(True), _prepared(4)
    t = build_targets(batch["thermal"], batch["rgb"], batch["mask"], layout, FIT)
    loader = [batch]
    stats = compute_token_stats(loader, layout, FIT)
    mean, std = layout.stats_vector(stats)
    z1 = standardize(layout.pack(t.parts), mean, std)
    assert z1.shape == (4, layout.num_tokens, layout.dim)
    bright = z1[..., layout.segments["brightness"]]
    assert torch.allclose(bright.mean(dim=(0, 1)), torch.zeros(bright.shape[-1]), atol=5e-3)
    assert (bright.std(dim=(0, 1)) < 1.5).all() and bright.std() > 0.5


def test_token_stats_match_a_direct_computation_and_ignore_masked_colour_pixels():
    layout, batch = tiny_layout(), _prepared(3)
    t = build_targets(batch["thermal"], batch["rgb"], batch["mask"], layout, FIT)
    stats = compute_token_stats([batch], layout, FIT)
    grid = t.parts["grid"]  # (B, 2, K, gy, gx): pooled over samples and cells, not over bins
    assert torch.allclose(stats["brightness"]["mean"], grid.mean(dim=(0, 3, 4)), atol=1e-5)
    assert torch.allclose(stats["brightness"]["std"], grid.std(dim=(0, 3, 4)), atol=1e-4)
    ab, w = t.parts["ab"], t.fraction
    mean = (ab * w).sum(dim=(0, 2, 3)) / w.sum()
    assert torch.allclose(stats["color"]["mean"], mean, atol=1e-4)
    assert "correction" not in stats


def test_token_stats_are_cached_by_layout_and_fit_and_recomputed_on_a_mismatch(tmp_path):
    layout, batch = tiny_layout(), _prepared()
    path = tmp_path / "stats.pt"
    calls = []

    def loader():
        calls.append(1)
        return [batch]

    first = load_or_compute_stats(path, loader, layout, FIT, TINY_INPUT, key_extra={"n": 2})
    again = load_or_compute_stats(path, loader, layout, FIT, TINY_INPUT, key_extra={"n": 2})
    assert len(calls) == 1 and path.exists()
    assert torch.equal(first["color"]["mean"], again["color"]["mean"])
    load_or_compute_stats(path, loader, tiny_layout(True), FIT, TINY_INPUT, key_extra={"n": 2})
    assert len(calls) == 2  # the layout changed
    load_or_compute_stats(
        path, loader, tiny_layout(True), FitSettings(1e-3, 0), TINY_INPUT, key_extra={"n": 2}
    )
    assert len(calls) == 3  # the fit settings changed
