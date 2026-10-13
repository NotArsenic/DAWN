"""Tests for box matching, similarity and homography fitting, acceptance gates, and warp masks."""

import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")
pytest.importorskip("PIL")
omegaconf = pytest.importorskip("omegaconf")

from dawn.data import align  # noqa: E402
from dawn.data.dronevehicle import Box  # noqa: E402
from tests.fake_dronevehicle import similarity  # noqa: E402

H, W = 512, 640
SIZE = (H, W)
CFG = align.AlignConfig()
EDGE_CFG = align.AlignConfig(grad_min_gain=0.0, grad_max_best_shift_px=1)


def rect(cx, cy, w=22.0, h=12.0, cat="car") -> Box:
    """Construct a rectangular Box centered at coordinates."""
    return Box(
        cat,
        (
            (cx - w / 2, cy - h / 2),
            (cx + w / 2, cy - h / 2),
            (cx + w / 2, cy + h / 2),
            (cx - w / 2, cy + h / 2),
        ),
        "polygon",
    )


def moved(box: Box, T: np.ndarray) -> Box:
    """Transform box vertices using affine transform matrix."""
    pts = align.apply_transform(T, box.vertices)
    return Box(box.category, tuple(map(tuple, pts)), box.kind)


def spread(n: int, cat="car") -> list[Box]:
    """Generate separated bounding boxes across the frame for testing."""
    rng = np.random.default_rng(0)
    cols = int(math.ceil(n / 3))
    out = []
    for i in range(n):
        r, c = divmod(i, cols)
        out.append(
            rect(
                60 + c * (520 / max(cols - 1, 1)) + rng.uniform(-8, 8),
                70 + r * 180 + rng.uniform(-8, 8),
                cat=cat,
            )
        )
    return out


def assert_same_map(fit, T, tol_px=0.5):
    """Assert that two transforms map coordinates to within a pixel tolerance."""
    xs, ys = np.meshgrid(np.linspace(0, 640, 9), np.linspace(0, 512, 7))
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1)
    err = np.abs(align.apply_transform(fit, pts) - align.apply_transform(T, pts)).max()
    assert err < tol_px, f"transforms differ by {err:.3f} px"


def pulled(rgb_boxes, T):
    """Transform bounding boxes using the inverse of a spatial transform."""
    inv = np.linalg.inv(T)
    return [moved(b, inv) for b in rgb_boxes]


# --- config --------------------------------------------------------------------------------------


def test_config_reads_the_align_section_of_data_yaml():
    path = Path(__file__).resolve().parents[1] / "configs" / "data.yaml"
    cfg = align.AlignConfig.from_config(
        omegaconf.OmegaConf.to_container(omegaconf.OmegaConf.load(path))["align"]
    )
    assert cfg.max_match_dist_px == 40
    assert cfg.copy_tol_px == 0.5
    assert cfg.exclude_copied_boxes is True
    assert (cfg.min_matches, cfg.min_inlier_ratio) == (6, 0.6)
    assert cfg.grad_min_gain == -0.2
    assert (cfg.max_median_residual_px, cfg.max_scale_deviation) == (2.0, 0.05)
    assert (cfg.max_rotation_deg, cfg.max_shift_px) == (2.0, 30.0)
    assert (cfg.homography_min_matches, cfg.grad_search_px) == (10, 4)
    assert cfg.grad_max_best_shift_px == 3
    assert (cfg.moving_min_residual_px, cfg.moving_dilate_px) == (4, 3)


# --- copied boxes --------------------------------------------------------------------------------


def test_a_box_is_copied_when_class_matches_and_all_four_vertices_agree():
    rgb = [rect(100, 100), rect(300, 200, cat="bus")]
    thermal = [rect(100, 100), rect(300, 200, cat="car")]  # second: same place, other class
    t_copied, r_copied = align.copied_flags(thermal, rgb, tol=0.5)
    assert t_copied.tolist() == [True, False]
    assert r_copied.tolist() == [True, False]


@pytest.mark.parametrize("offset, copied", [(0.5, True), (0.51, False), (0.0, True)])
def test_copy_tolerance_is_inclusive_and_applies_to_every_vertex(offset, copied):
    base = rect(100, 100)
    verts = [list(v) for v in base.polygon]
    verts[2][1] += offset  # only one vertex is off
    shifted = Box("car", tuple(map(tuple, verts)), "polygon")
    t_copied, _ = align.copied_flags([shifted], [base], tol=0.5)
    assert bool(t_copied[0]) is copied


def test_labels_identical_needs_every_box_on_both_sides_copied():
    a = [rect(100, 100), rect(300, 200)]
    assert align.labels_identical(list(a), a, tol=0.5)
    assert not align.labels_identical(a[:1], a, tol=0.5)  # different sizes
    assert not align.labels_identical([a[0], rect(310, 200)], a, tol=0.5)
    assert not align.labels_identical([], [], tol=0.5)  # no labels at all is not "copied"


# --- matching ------------------------------------------------------------------------------------


def test_matches_are_mutual_nearest_neighbours_of_the_same_class_within_the_gate():
    rgb = [rect(100, 100), rect(300, 100), rect(500, 100, cat="bus")]
    thermal = [rect(103, 98), rect(299, 104), rect(502, 99, cat="car")]  # third class differs
    m = align.match_boxes(thermal, rgb, CFG)
    assert m.thermal_idx.tolist() == [0, 1]
    assert m.rgb_idx.tolist() == [0, 1]
    np.testing.assert_allclose(m.thermal_xy[0], [103, 98])
    np.testing.assert_allclose(m.rgb_xy[1], [300, 100])
    assert (m.n_thermal, m.n_rgb, m.n) == (3, 3, 2)


def test_a_thermal_box_that_is_not_its_rgb_boxs_nearest_neighbour_is_not_matched():
    rgb = [rect(100, 100)]
    thermal = [rect(110, 100), rect(104, 100)]  # both want rgb 0; only the closer is mutual
    m = align.match_boxes(thermal, rgb, CFG)
    assert m.thermal_idx.tolist() == [1]


@pytest.mark.parametrize("distance, matched", [(40.0, True), (40.01, False)])
def test_gate_is_inclusive_and_comes_from_the_config(distance, matched):
    m = align.match_boxes([rect(100 + distance, 100)], [rect(100, 100)], CFG)
    assert (m.n == 1) is matched
    loose = align.AlignConfig(max_match_dist_px=60)
    assert align.match_boxes([rect(100 + distance, 100)], [rect(100, 100)], loose).n == 1


def test_copied_boxes_are_excluded_from_matching_on_both_sides():
    rgb = [rect(100, 100), rect(300, 200)]
    thermal = [rect(100, 100), rect(303, 201)]  # first is an exact copy
    m = align.match_boxes(thermal, rgb, CFG)
    assert m.thermal_idx.tolist() == [1]
    assert m.rgb_idx.tolist() == [1]
    assert m.n_copied == 1
    assert not m.labels_identical


def test_exclusion_can_be_switched_off_for_a_dry_run():
    rgb = [rect(100, 100), rect(300, 200)]
    thermal = [rect(100, 100), rect(303, 201)]
    off = align.AlignConfig(exclude_copied_boxes=False)
    m = align.match_boxes(thermal, rgb, off)
    assert m.n == 2
    assert m.n_copied == 1  # still counted, just not excluded


def test_identical_label_sets_have_no_matches_and_say_so():
    boxes = spread(8)
    m = align.match_boxes(list(boxes), boxes, CFG)
    assert m.n == 0 and m.labels_identical and m.n_copied == 8


def test_no_boxes_at_all_is_no_matches():
    m = align.match_boxes([], [], CFG)
    assert m.n == 0 and not m.labels_identical
    assert m.thermal_xy.shape == (0, 2)


# --- similarity ----------------------------------------------------------------------------------


def test_fit_similarity_recovers_a_known_transform_from_points():
    T = similarity(1.02, 1.3, 7.0, -4.0)
    src = np.random.default_rng(1).uniform(0, 600, (9, 2))
    np.testing.assert_allclose(
        align.fit_similarity(src, align.apply_transform(T, src)), T, atol=1e-9
    )


def test_fit_similarity_from_two_points_is_exact():
    T = similarity(0.97, -0.8, -3.0, 2.0)
    src = np.array([[100.0, 50.0], [400.0, 300.0]])
    np.testing.assert_allclose(
        align.fit_similarity(src, align.apply_transform(T, src)), T, atol=1e-9
    )


def test_fit_similarity_never_returns_a_reflection():
    # A mirror image has no rotation that explains it: the least-squares scale collapses to zero
    # (so the scale check rejects it) instead of the fit flipping its sign.
    src = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    dst = src * np.array([-1.0, 1.0])
    T = align.fit_similarity(src, dst)
    assert np.linalg.det(T[:2, :2]) >= 0
    assert align.transform_summary(T, SIZE)[0] == pytest.approx(0.0, abs=1e-9)


def test_fit_similarity_rejects_coincident_points():
    with pytest.raises(ValueError, match="distinct"):
        align.fit_similarity(np.ones((3, 2)), np.zeros((3, 2)))


def test_summary_of_a_similarity():
    scale, rot, shift = align.transform_summary(similarity(1.03, -1.5, 0.0, 0.0), SIZE)
    assert scale == pytest.approx(1.03)
    assert rot == pytest.approx(-1.5)
    # (320, 256) -> a * (320, 256) rotated: displacement is not zero for a scale about the origin
    assert shift == pytest.approx(
        np.hypot(*(similarity(1.03, -1.5, 0, 0) @ [320, 256, 1])[:2] - [320, 256])
    )


def test_shift_is_the_displacement_of_the_frame_centre_not_the_raw_translation():
    # Rotate by 1 degree about the frame centre: the translation column is large, the centre stays.
    R = similarity(1.0, 1.0, 0.0, 0.0)
    centre = np.array([320.0, 256.0])
    t = centre - R[:2, :2] @ centre
    R[:2, 2] = t
    assert np.hypot(*t) > 4.0  # the raw translation is not small
    _, _, shift = align.transform_summary(R, SIZE)
    assert shift == pytest.approx(0.0, abs=1e-9)


def test_summary_of_a_pure_translation():
    T = similarity(1.0, 0.0, 3.0, -4.0)
    scale, rot, shift = align.transform_summary(T, SIZE)
    assert (scale, rot) == (pytest.approx(1.0), pytest.approx(0.0))
    assert shift == pytest.approx(5.0)


# --- RANSAC --------------------------------------------------------------------------------------


def _cloud(n, T, n_out, seed=0, noise=0.3):
    rng = np.random.default_rng(seed)
    src = rng.uniform([20, 20], [620, 490], (n, 2))
    dst = align.apply_transform(T, src) + rng.normal(0, noise, (n, 2))
    dst[:n_out] += rng.uniform(15, 40, (n_out, 2)) * rng.choice([-1, 1], (n_out, 2))
    return src, dst


def test_ransac_recovers_the_transform_despite_outliers():
    T = similarity(1.012, 0.7, 5.0, -3.0)
    src, dst = _cloud(24, T, n_out=7)
    fit, inliers = align.ransac_similarity(src, dst, inlier_px=4, enumerate_max=40, iterations=500)
    assert_same_map(fit, T)
    assert inliers.sum() == 17
    assert not inliers[:7].any()  # the planted outliers are out


def test_ransac_by_enumeration_needs_no_random_generator_and_is_deterministic():
    T = similarity(1.0, 0.5, 2.0, 2.0)
    src, dst = _cloud(18, T, n_out=5, seed=3)
    a = align.ransac_similarity(src, dst, 4, 40, 500, generator=None)
    b = align.ransac_similarity(src, dst, 4, 40, 500, generator=None)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_ransac_above_the_enumeration_limit_samples_from_a_seeded_generator():
    from dawn.data.seeding import generator_for

    T = similarity(0.99, -0.4, -2.0, 4.0)
    src, dst = _cloud(60, T, n_out=15, seed=4)
    run = lambda tag: align.ransac_similarity(  # noqa: E731
        src, dst, 4, 40, 300, generator=generator_for(tag, "ransac")
    )
    a, b = run("train_00001"), run("train_00001")
    np.testing.assert_array_equal(a[0], b[0])
    assert_same_map(a[0], T)
    assert a[1].sum() >= 44


def test_ransac_with_exactly_two_matches_returns_the_exact_fit():
    T = similarity(1.0, 0.0, 6.0, -1.0)
    src = np.array([[100.0, 100.0], [300.0, 250.0]])
    fit, inliers = align.ransac_similarity(src, align.apply_transform(T, src), 4, 40, 10)
    np.testing.assert_allclose(fit, T, atol=1e-9)
    assert inliers.all()


# --- homography ----------------------------------------------------------------------------------


def _homography(g=1e-4):
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [g, 0.0, 1.0]])


def test_fit_homography_recovers_a_known_homography():
    Hm = np.array([[1.01, 0.01, 4.0], [-0.01, 0.99, -3.0], [2e-5, -1e-5, 1.0]])
    src = np.random.default_rng(5).uniform(0, 600, (12, 2))
    np.testing.assert_allclose(
        align.fit_homography(src, align.apply_transform(Hm, src)), Hm, atol=1e-8
    )


def test_ransac_homography_ignores_outliers():
    Hm = _homography()
    src = np.random.default_rng(6).uniform(20, 600, (20, 2))
    dst = align.apply_transform(Hm, src)
    dst[:5] += 30.0
    from dawn.data.seeding import generator_for

    fit, inliers = align.ransac_homography(src, dst, 4, 500, generator_for("x", "ransac"))
    assert inliers.sum() == 15 and not inliers[:5].any()
    np.testing.assert_allclose(align.apply_transform(fit, src[5:]), dst[5:], atol=0.05)


def test_summary_of_a_homography_uses_the_local_jacobian_at_the_frame_centre():
    scale, rot, shift = align.transform_summary(_homography(1e-4), SIZE)
    w = 1 + 1e-4 * 320
    assert scale == pytest.approx(math.sqrt(1 / w**3), rel=1e-6)
    assert shift == pytest.approx(math.hypot(320 / w - 320, 256 / w - 256))
    assert abs(rot) < 1.0


# --- acceptance ----------------------------------------------------------------------------------

GOOD = align.GeometryMetrics(
    n_matches=12,
    labels_identical=False,
    n_inliers=12,
    inlier_ratio=1.0,
    median_residual=0.5,
    median_residual_all=0.5,
    scale=1.0,
    rotation_deg=0.0,
    shift_px=5.0,
)


def _with(**kw):
    return align.GeometryMetrics(**{**GOOD.__dict__, **kw})


def test_a_good_fit_has_no_failed_checks():
    assert align.failed_checks(GOOD, CFG) == []


@pytest.mark.parametrize(
    "change, reason",
    [
        ({"n_matches": 6}, None),
        ({"n_matches": 5}, "few_matches"),
        ({"inlier_ratio": 0.6}, None),
        ({"inlier_ratio": 0.59}, "inlier_ratio"),
        ({"median_residual": 2.0}, None),
        ({"median_residual": 2.01}, "residual"),
        ({"scale": 1.05}, None),
        ({"scale": 0.95}, None),
        ({"scale": 1.0501}, "scale"),
        ({"scale": 0.9499}, "scale"),
        ({"rotation_deg": 1.99}, None),
        ({"rotation_deg": -1.99}, None),
        ({"rotation_deg": 2.0}, "rotation"),
        ({"rotation_deg": -2.0}, "rotation"),
        ({"shift_px": 29.99}, None),
        ({"shift_px": 30.0}, "shift"),
    ],
)
def test_each_threshold_at_its_boundary(change, reason):
    failed = align.failed_checks(_with(**change), CFG)
    assert failed == ([] if reason is None else [reason])


def test_failures_are_listed_in_plan_order_and_the_first_is_the_reason():
    bad = _with(
        n_matches=3,
        inlier_ratio=0.2,
        median_residual=9.0,
        scale=1.2,
        rotation_deg=5.0,
        shift_px=50.0,
    )
    order = ["few_matches", "inlier_ratio", "residual", "scale", "rotation", "shift"]
    assert align.failed_checks(bad, CFG) == order
    assert align.first_reason(bad, CFG) == "few_matches"
    assert align.first_reason(GOOD, CFG) == "ok"


def test_identical_label_sets_report_labels_copied_instead_of_few_matches():
    copied = _with(n_matches=0, labels_identical=True)
    assert align.failed_checks(copied, CFG)[0] == "labels_copied"
    assert align.first_reason(copied, CFG) == "labels_copied"


def test_nan_metrics_fail_rather_than_pass():
    nan = float("nan")
    unfit = _with(inlier_ratio=nan, median_residual=nan, scale=nan, rotation_deg=nan, shift_px=nan)
    assert align.failed_checks(unfit, CFG) == [
        "inlier_ratio",
        "residual",
        "scale",
        "rotation",
        "shift",
    ]


def test_thresholds_come_from_the_config():
    loose = align.AlignConfig(max_median_residual_px=3.0, min_matches=3)
    assert align.failed_checks(_with(median_residual=2.5, n_matches=4), loose) == []


# --- align_boxes ---------------------------------------------------------------------------------


def test_align_boxes_recovers_a_known_transform_from_boxes():
    T = similarity(1.01, 0.6, 4.0, -3.0)
    rgb = spread(12)
    rng = np.random.default_rng(1)
    thermal = [
        Box(
            b.category,
            tuple((x + rng.normal(0, 0.2), y + rng.normal(0, 0.2)) for x, y in b.polygon),
            "polygon",
        )
        for b in pulled(rgb, T)
    ]
    result = align.align_boxes(thermal, rgb, CFG, "train_00001", SIZE)
    assert result.reason == "ok"
    assert result.model == "similarity"
    assert_same_map(result.T, T)
    assert result.metrics.n_matches == 12 and result.metrics.n_inliers == 12
    assert result.metrics.median_residual < 0.5
    assert result.metrics.scale == pytest.approx(1.01, abs=1e-3)
    assert result.metrics.rotation_deg == pytest.approx(0.6, abs=0.05)


def test_align_boxes_flags_a_moving_vehicle_and_still_accepts():
    T = similarity(1.0, 0.0, 3.0, 2.0)
    rgb = spread(10)
    thermal = pulled(rgb, T)
    thermal[4] = moved(thermal[4], similarity(1.0, 0.0, 14.0, 0.0))  # this one moved
    result = align.align_boxes(thermal, rgb, CFG, "id", SIZE)
    assert result.reason == "ok"
    assert result.metrics.n_matches == 10 and result.metrics.n_inliers == 9
    assert result.n_moving == 1


def test_few_matches_is_the_reason_with_fewer_than_six():
    T = similarity(1.0, 0.0, 3.0, 2.0)
    rgb = spread(5)
    result = align.align_boxes(pulled(rgb, T), rgb, CFG, "id", SIZE)
    assert result.reason == "few_matches"
    assert result.metrics.n_matches == 5
    assert result.T is not None  # the fit is still made and logged


def test_one_match_has_no_fit():
    result = align.align_boxes([rect(100, 100)], [rect(105, 100)], CFG, "id", SIZE)
    assert result.reason == "few_matches"
    assert result.T is None and result.model == "none"
    assert math.isnan(result.metrics.median_residual)


def test_identical_labels_are_labels_copied_and_a_dry_run_would_accept_them():
    rgb = spread(9)
    result = align.align_boxes(list(rgb), rgb, CFG, "id", SIZE)
    assert result.reason == "labels_copied"
    dry = align.align_boxes(
        list(rgb), rgb, align.AlignConfig(exclude_copied_boxes=False), "id", SIZE
    )
    assert dry.reason == "ok"  # zero residual by construction: why copies are not evidence
    assert dry.metrics.median_residual == pytest.approx(0.0, abs=1e-9)


def test_scale_off_by_eight_percent_is_rejected_for_scale():
    T = similarity(1.08, 0.0, 0.0, 0.0)
    rgb = spread(10)
    result = align.align_boxes(pulled(rgb, T), rgb, CFG, "id", SIZE)
    assert result.reason == "scale"


def test_homography_is_tried_only_when_the_similarity_fails_on_residual_alone():
    Hm = _homography(6e-5)  # a similarity leaves a 2.3 px median residual, nothing else fails
    rgb = spread(14)
    thermal = [moved(b, np.linalg.inv(Hm)) for b in rgb]
    sim_only = align.align_boxes(
        thermal, rgb, align.AlignConfig(homography_enabled=False), "id", SIZE
    )
    assert sim_only.reason == "residual"  # the precondition: a similarity cannot fit this
    assert align.failed_checks(sim_only.sim, CFG) == ["residual"]
    result = align.align_boxes(thermal, rgb, CFG, "id", SIZE)
    assert result.model == "homography"
    assert result.reason == "ok"
    assert result.T.shape == (3, 3) and not np.allclose(result.T[2, :2], 0)
    assert result.hom is not None and result.hom.median_residual < 0.1
    assert result.sim.median_residual > 2.0  # the similarity attempt is logged too


def test_homography_is_not_tried_below_its_minimum_number_of_matches():
    Hm = _homography(6e-5)
    rgb = spread(8)  # 8 < 10
    result = align.align_boxes([moved(b, np.linalg.inv(Hm)) for b in rgb], rgb, CFG, "id", SIZE)
    assert result.hom is None
    assert result.model == "similarity"
    assert result.reason == "residual"


def test_homography_is_not_tried_when_other_checks_fail_too():
    T = similarity(1.08, 0.0, 0.0, 0.0)  # scale fails, residual fine
    rgb = spread(12)
    result = align.align_boxes(pulled(rgb, T), rgb, CFG, "id", SIZE)
    assert result.hom is None and result.reason == "scale"


def test_align_boxes_is_deterministic_above_the_enumeration_limit():
    T = similarity(1.0, 0.2, 2.0, 1.0)
    rgb = [rect(20 + 14 * (i % 40), 40 + 40 * (i // 40) + 3 * (i % 7)) for i in range(60)]
    thermal = pulled(rgb, T)
    a = align.align_boxes(thermal, rgb, CFG, "val_00007", SIZE)
    b = align.align_boxes(thermal, rgb, CFG, "val_00007", SIZE)
    np.testing.assert_array_equal(a.T, b.T)
    assert a.metrics == b.metrics


def test_unmatched_boxes_are_counted_and_not_part_of_the_fit():
    T = similarity(1.0, 0.0, 3.0, 2.0)
    rgb = spread(10) + [rect(30, 480)]
    thermal = pulled(rgb[:10], T) + [rect(620, 20)]  # one extra box on each side, far apart
    result = align.align_boxes(thermal, rgb, CFG, "id", SIZE)
    assert (result.n_unmatched_rgb, result.n_unmatched_thermal) == (1, 1)


# --- warp ----------------------------------------------------------------------------------------


def _ramp():
    return (torch.arange(W, dtype=torch.float32) / 1000).expand(H, W).unsqueeze(0).contiguous()


def _texture(seed=0):
    return torch.rand(1, H, W, generator=torch.Generator().manual_seed(seed))


def test_identity_warp_reproduces_the_input_exactly_and_everything_is_valid():
    thermal = _texture()
    warped, valid = align.warp_thermal(thermal, np.eye(3), SIZE)
    assert warped.dtype == torch.float32 and warped.shape == (1, H, W)
    assert torch.equal(warped, thermal)
    assert valid.dtype == torch.bool and valid.all()


def test_an_integer_shift_equals_a_shifted_copy_and_the_valid_mask_is_exact():
    thermal = _texture(1)
    dx, dy = 5, -3  # thermal pixel (c, r) lands on RGB pixel (c + 5, r - 3)
    warped, valid = align.warp_thermal(thermal, similarity(1.0, 0.0, dx, dy), SIZE)
    # RGB pixel (row r', col c') takes thermal (r' + 3, c' - 5) where that exists.
    expected_valid = torch.zeros(H, W, dtype=torch.bool)
    expected_valid[: H - 3, 5:] = (
        True  # source rows 3..H-1 -> rows 0..H-4; source cols 0..W-6 -> 5..
    )
    inner = expected_valid.clone()  # then eroded by one pixel against the invalid region
    inner[H - 4, :] = False  # last valid row touches the invalid rows below
    inner[:, 5] = False  # first valid column touches the invalid columns on its left
    assert torch.equal(valid[0], inner)
    rows, cols = slice(0, H - 4), slice(6, W)
    assert torch.equal(warped[0, rows, cols], thermal[0, 3 : H - 1, 1 : W - 5])


def test_a_half_pixel_shift_is_bilinear_and_exact_on_a_ramp():
    warped, valid = align.warp_thermal(_ramp(), similarity(1.0, 0.0, 0.5, 0.0), SIZE)
    # output(c') = input(c' - 0.5) = (c' - 0.5) / 1000 away from the left edge
    expected = ((torch.arange(W, dtype=torch.float32) - 0.5) / 1000)[None, None]
    assert torch.allclose(warped[:, :, 3:], expected.expand(1, H, W)[:, :, 3:], atol=1e-6)
    assert valid.all()  # a sub-half-pixel shift keeps the whole frame


def test_a_scale_about_the_origin_samples_the_ramp_at_the_inverse_position():
    warped, _ = align.warp_thermal(_ramp(), similarity(2.0, 0.0, 0.0, 0.0), SIZE)
    cols = torch.arange(W, dtype=torch.float32)
    expected = cols / 2 / 1000  # RGB column c' came from thermal column c' / 2
    assert torch.allclose(warped[0, 10], expected, atol=1e-6)


def test_the_validity_mask_marks_pixels_pulled_from_outside_the_thermal_frame():
    # Scale 0.9 about the origin: RGB columns beyond 0.9 * W have no thermal source.
    _, valid = align.warp_thermal(_texture(), similarity(0.9, 0.0, 0.0, 0.0), SIZE)
    assert valid[0, 10, 100] and not valid[0, 10, W - 1] and not valid[0, H - 1, 10]
    edge = int(0.9 * W)
    assert valid[0, 10, edge - 3] and not valid[0, 10, edge + 3]


def test_warp_is_the_inverse_mapping_so_boxes_and_pixels_move_together():
    T = similarity(1.0, 0.0, 12.0, 7.0)
    thermal = torch.zeros(1, H, W)
    thermal[:, 100:120, 200:230] = 1.0  # a hot block in thermal coordinates
    warped, _ = align.warp_thermal(thermal, T, SIZE)
    ys, xs = torch.nonzero(warped[0] > 0.5, as_tuple=True)
    assert (xs.min().item(), xs.max().item()) == (212, 241)  # T moves x by +12
    assert (ys.min().item(), ys.max().item()) == (107, 126)  # and y by +7


def test_warp_accepts_a_homography():
    Hm = _homography(1e-4)
    warped, valid = align.warp_thermal(_texture(), Hm, SIZE)
    assert warped.shape == (1, H, W) and valid.any()


# --- moving mask ---------------------------------------------------------------------------------


def _moving_setup():
    T = similarity(1.0, 0.0, 0.0, 0.0)
    rgb = [rect(100, 100), rect(300, 200), rect(500, 300)]
    thermal = [rect(101, 100), rect(310, 200), rect(501, 300)]  # box 1 is 10 px off, others 1 px
    matches = align.match_boxes(thermal, rgb, CFG)
    return T, rgb, thermal, matches


def test_only_boxes_whose_own_residual_exceeds_the_threshold_are_moving():
    T, _, _, matches = _moving_setup()
    flags = align.moving_flags(matches, T, min_residual_px=4)
    assert flags.tolist() == [False, True, False]


def test_moving_mask_is_the_union_of_both_polygons_dilated():
    T, rgb, thermal, matches = _moving_setup()
    mask, n_moving = align.moving_mask(thermal, rgb, matches, T, SIZE, CFG)
    assert n_moving == 1
    assert mask.dtype == torch.bool and mask.shape == (1, H, W)
    # centre of the moving RGB box and of its thermal counterpart (10 px apart) are both masked
    assert mask[0, 200, 300] and mask[0, 200, 310]
    assert not mask[0, 100, 100] and not mask[0, 300, 500]  # steady vehicles are not


def test_moving_mask_dilates_by_a_euclidean_disk():
    T, rgb, thermal, matches = _moving_setup()
    mask, _ = align.moving_mask(thermal, rgb, matches, T, SIZE, CFG)
    top, left = 200 - 6, 300 - 11  # the RGB box's top edge row and left edge column
    assert mask[0, top - 3, 300]  # 3 px above the edge
    assert not mask[0, top - 4, 300]  # 4 px above
    assert mask[0, top - 2, left - 2]  # corner pixel at distance 2.83 <= 3
    assert not mask[0, top - 3, left - 3]  # at distance 4.24: a square dilation would keep it


def test_moving_mask_uses_the_thermal_polygon_pushed_through_the_fit():
    T = similarity(1.0, 0.0, 20.0, 0.0)  # thermal is 20 px left of RGB
    rgb = [rect(300, 200)]
    thermal = [rect(300 - 20 + 10, 200)]  # after T it sits 10 px from the RGB box: moving
    matches = align.match_boxes(thermal, rgb, CFG)
    mask, n = align.moving_mask(thermal, rgb, matches, T, SIZE, CFG)
    assert n == 1
    assert mask[0, 200, 310] and mask[0, 200, 300]  # RGB box and the warped thermal box
    assert not mask[0, 200, 290 - 11 - 6]  # unwarped thermal position is not masked


def test_unmatched_and_copied_boxes_are_never_masked():
    T = similarity(1.0, 0.0, 0.0, 0.0)
    rgb = [rect(100, 100), rect(300, 200)]
    thermal = [rect(100, 100), rect(420, 200)]  # first copied, second 120 px away: unmatched
    matches = align.match_boxes(thermal, rgb, CFG)
    mask, n = align.moving_mask(thermal, rgb, matches, T, SIZE, CFG)
    assert n == 0 and not mask.any()


def test_raster_of_an_integer_polygon_covers_its_vertices_inclusively():
    box = rect(50, 50, w=10, h=6)
    mask = align.rasterize([box.vertices], SIZE)  # rows 47..53, columns 45..55
    assert mask[47, 45] and mask[53, 55] and mask[47, 55] and mask[53, 45]  # corners are inside
    assert mask.sum() == 7 * 11
    assert not mask[46, 50] and not mask[54, 50] and not mask[50, 44] and not mask[50, 56]


# --- edge check ----------------------------------------------------------------------------------


def _luma(seed=0):
    rng = np.random.default_rng(seed)
    from scipy import ndimage

    base = ndimage.gaussian_filter(rng.random((H, W)), 3)
    base = (base - base.min()) / np.ptp(base)
    for _ in range(25):
        y, x = int(rng.integers(0, H - 80)), int(rng.integers(0, W - 90))
        base[y : y + int(rng.integers(20, 70)), x : x + int(rng.integers(20, 80))] = rng.random()
    return base.astype(np.float32)


def _shifted(img, dx, dy):
    return np.roll(np.roll(img, dy, axis=0), dx, axis=1)


def test_a_correct_warp_improves_the_gradient_correlation_and_passes():
    luma = _luma()
    thermal = torch.from_numpy(_shifted(luma, -4, 3))[None]  # content sits at T^-1 of the RGB one
    T = similarity(1.0, 0.0, 4.0, -3.0)  # thermal (c, r) -> RGB (c + 4, r - 3)
    result = align.edge_check(thermal, luma, T, EDGE_CFG)
    assert result.after > result.before
    assert result.gain > 0.1
    assert (result.best_dx, result.best_dy) == (0, 0)
    assert result.passed


def test_content_displaced_from_the_labels_fails_on_the_best_shift():
    luma = _luma(1)
    thermal = torch.from_numpy(_shifted(luma, 3, 0))[None]
    result = align.edge_check(thermal, luma, np.eye(3), EDGE_CFG)  # labels say: no shift
    assert result.gain == pytest.approx(0.0, abs=1e-9)
    # best_dx is the shift to apply to the warped thermal: its content must move 3 px left
    assert (result.best_dx, result.best_dy) == (-3, 0)
    assert not result.passed


def test_a_warp_that_breaks_an_aligned_pair_has_negative_gain_and_fails():
    luma = _luma(2)
    thermal = torch.from_numpy(luma.copy())[None]  # already aligned
    result = align.edge_check(thermal, luma, similarity(1.0, 0.0, 5.0, 5.0), EDGE_CFG)
    assert result.gain < 0
    assert not result.passed


def test_one_pixel_residual_misalignment_still_passes():
    luma = _luma(3)
    thermal = torch.from_numpy(_shifted(luma, 1, 0))[None]
    result = align.edge_check(thermal, luma, np.eye(3), EDGE_CFG)
    assert abs(result.best_dx) == 1
    assert result.passed  # |best shift| <= grad_check.max_best_shift_px (1)


def test_edge_check_thresholds_come_from_the_config():
    luma = _luma(3)
    thermal = torch.from_numpy(_shifted(luma, 1, 0))[None]
    assert align.edge_check(thermal, luma, np.eye(3), EDGE_CFG).passed
    strict = align.AlignConfig(grad_max_best_shift_px=0)
    assert not align.edge_check(thermal, luma, np.eye(3), strict).passed

    t2 = torch.from_numpy(_shifted(luma, -4, 3))[None]
    T = similarity(1.0, 0.0, 4.0, -3.0)
    assert align.edge_check(t2, luma, T, EDGE_CFG).passed
    demanding = align.AlignConfig(grad_min_gain=10.0)  # a correlation gain can never reach this
    assert not align.edge_check(t2, luma, T, demanding).passed


def test_rgb_edge_luma_is_the_srgb_coded_luma_of_a_linear_image():
    rgb = torch.zeros(3, 4, 4)
    rgb[1] = 1.0  # pure green, linear
    luma = align.rgb_edge_luma(rgb)
    assert luma.shape == (4, 4) and luma.dtype == np.float32
    assert luma[0, 0] == pytest.approx(0.8625, abs=1e-3)  # sRGB code of linear 0.7152, /255


# --- edge-correlation fallback: the translation estimator ----------------------------------------

FB_CFG = align.AlignConfig(
    fallback_search_px=6, fallback_min_prominence=0.02, fallback_max_quadrant_dev_px=2
)


def _brute_ncc(ref, padded, s):
    """Pearson correlation for every shift by looping: grid[dy + s, dx + s]."""
    h, w = ref.shape
    out = np.zeros((2 * s + 1, 2 * s + 1))
    for dy in range(-s, s + 1):
        for dx in range(-s, s + 1):
            cand = padded[s - dy : s - dy + h, s - dx : s - dx + w]
            out[dy + s, dx + s] = np.corrcoef(cand.ravel(), ref.ravel())[0, 1]
    return out


def test_ncc_grid_matches_a_brute_force_loop_over_shifts():
    rng = np.random.default_rng(0)
    s = 3
    ref = rng.random((24, 30))
    padded = rng.random((24 + 2 * s, 30 + 2 * s))
    got = align._ncc_grid(torch.from_numpy(ref), torch.from_numpy(padded), s).numpy()
    np.testing.assert_allclose(got, _brute_ncc(ref, padded, s), atol=1e-9)


def test_ncc_grid_peaks_at_the_shift_that_realigns_the_content():
    rng = np.random.default_rng(1)
    s = 4
    base = rng.random((60, 70))
    # the content of `padded` is `ref` displaced by (+2 cols, -3 rows): realigning moves it by
    # (-2, +3), and grid[dy + s, dx + s] is the correlation after moving the content by (dx, dy)
    ref = base[10:50, 10:60]
    padded = np.zeros((40 + 2 * s, 50 + 2 * s))
    padded[:, :] = base[10 - s + 3 : 50 + s + 3, 10 - s - 2 : 60 + s - 2]
    grid = align._ncc_grid(torch.from_numpy(ref), torch.from_numpy(padded), s).numpy()
    iy, ix = np.unravel_index(grid.argmax(), grid.shape)
    assert (ix - s, iy - s) == (-2, 3)
    assert grid.max() == pytest.approx(1.0, abs=1e-9)


def _textured(seed=0):
    return _luma(seed)  # blurred noise with sharp rectangles: plenty of edges


def _thermal_from(luma, dx, dy):
    """A thermal whose content sits where RGB content sits, displaced by (-dx, -dy)."""
    return torch.from_numpy(_shifted(luma, -dx, -dy))[None]


@pytest.mark.parametrize("dx, dy", [(0, 0), (3, -2), (-5, 4), (6, 0)])
def test_the_estimator_recovers_a_known_translation(dx, dy):
    luma = _textured(2)
    est = align.edge_translation(_thermal_from(luma, dx, dy), luma, FB_CFG)
    assert (est.dx, est.dy) == (dx, dy)
    assert est.prominence > 0.02 and est.quad_dev == 0
    assert est.corr > est.corr_at_zero or (dx, dy) == (0, 0)
    assert est.saturated == (max(abs(dx), abs(dy)) == FB_CFG.fallback_search_px)


def test_a_shift_beyond_the_window_is_flagged_as_saturated_not_trusted():
    luma = _textured(3)
    est = align.edge_translation(_thermal_from(luma, 9, 0), luma, FB_CFG)
    assert est.saturated
    assert not align.fallback_accepts(est, FB_CFG)


def test_unrelated_content_has_no_prominent_peak_or_disagreeing_quadrants():
    a, b = _textured(4), _textured(5)
    est = align.edge_translation(torch.from_numpy(b)[None], a, FB_CFG)
    assert not align.fallback_accepts(est, FB_CFG)


def test_quadrants_that_move_differently_are_rejected_even_with_a_clear_global_peak():
    luma = _textured(6)
    thermal = _shifted(luma, -2, 0).copy()  # most of the frame shifted by +2 columns
    thermal[:, 320:] = _shifted(luma, 3, 0)[:, 320:]  # the right half the other way
    est = align.edge_translation(torch.from_numpy(thermal)[None], luma, FB_CFG)
    assert est.quad_dev >= 3
    assert not align.fallback_accepts(est, FB_CFG)


def _est(**kw):
    base = dict(
        dx=1, dy=-1, corr=0.7, corr_at_zero=0.5, prominence=0.05, quad_dev=1, saturated=False
    )
    return align.TranslationEstimate(**{**base, **kw})


@pytest.mark.parametrize(
    "change, accepted",
    [
        ({}, True),
        ({"prominence": 0.02}, True),  # inclusive
        ({"prominence": 0.0199}, False),
        ({"quad_dev": 2}, True),  # inclusive
        ({"quad_dev": 3}, False),
        ({"saturated": True}, False),
        ({"prominence": float("nan")}, False),
    ],
)
def test_the_fallback_acceptance_rule_at_its_boundaries(change, accepted):
    assert align.fallback_accepts(_est(**change), FB_CFG) is accepted


def test_the_fallback_can_be_switched_off():
    off = align.AlignConfig(fallback_enabled=False)
    assert not align.fallback_accepts(_est(), off)


def test_translation_matrix_moves_thermal_pixels_onto_rgb():
    T = align.translation_matrix(4, -3)
    np.testing.assert_allclose(align.apply_transform(T, np.array([[10.0, 20.0]])), [[14.0, 17.0]])
    scale, rot, shift = align.transform_summary(T, SIZE)
    assert (scale, rot, shift) == (pytest.approx(1.0), pytest.approx(0.0), pytest.approx(5.0))


def test_config_reads_the_fallback_section_of_data_yaml():
    path = Path(__file__).resolve().parents[1] / "configs" / "data.yaml"
    cfg = align.AlignConfig.from_config(
        omegaconf.OmegaConf.to_container(omegaconf.OmegaConf.load(path))["align"]
    )
    assert cfg.fallback_enabled is True
    assert (cfg.fallback_search_px, cfg.fallback_min_prominence) == (10, 0.02)
    assert cfg.fallback_max_quadrant_dev_px == 2


# --- the vehicles mask --------------------------------------------------------------------------


def rotated(cx, cy, w, h, deg, cat="car") -> Box:
    """An oriented rectangle: the polygon is rotated, its bounding box is not the vehicle."""
    a = math.radians(deg)
    corners = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    pts = corners @ rot.T + [cx, cy]
    return Box(cat, tuple(map(tuple, pts)), "polygon")


VM = align.VehicleMaskParams(
    dilate_px=3.0,
    extend_px=17.4,
    window_margin_px=12,
    search_px=6,
    max_residual_px=2,
    min_prominence=0.02,
    min_window_px=32,
)
NO_IMAGES = None  # without images the static test cannot run


def _scene(seed=0, cars=()):
    """RGB-like luma with a few bright car rectangles `(cx, cy, w, h)` on a textured road."""
    luma = _luma(seed).copy()
    for cx, cy, w, h in cars:
        luma[cy - h // 2 : cy + h // 2 + 1, cx - w // 2 : cx + w // 2 + 1] = 1.0
    return luma


def _images(rgb_luma, thermal_luma, sigma=1.0):
    """What `vehicle_mask` needs for the static test: gradient maps and the validity mask."""
    return (
        align.gradient_magnitude(rgb_luma, sigma),
        align.gradient_magnitude(thermal_luma, sigma),
        np.ones(rgb_luma.shape, bool),
    )


def vmask(rgb, thermal, T, images=None, params=VM):
    return align.vehicle_mask(rgb, thermal, T, SIZE, params, CFG, images)


def test_every_rgb_polygon_is_covered_after_dilation():
    boxes = [rotated(100, 100, 40, 14, 30), rotated(300, 200, 30, 12, -60), rect(500, 400)]
    mask, _ = vmask(boxes, [], None)
    assert mask.dtype == torch.bool and mask.shape == (1, H, W)
    inside = align.rasterize([b.vertices for b in boxes], SIZE)
    assert mask[0].numpy()[inside].all()  # every pixel of every polygon is masked
    assert mask.sum() > inside.sum()  # and the dilation added a margin


def test_the_mask_is_empty_on_a_frame_with_no_boxes():
    mask, counts = vmask([], [], None)
    assert not mask.any() and counts == align.VehicleBranches()
    assert not vmask([], [], np.eye(3))[0].any()  # with a transform too


def test_oriented_polygons_are_rasterized_not_their_bounding_boxes():
    box = rotated(200, 200, 60, 10, 45)  # a thin diagonal vehicle: its bbox is mostly empty road
    params = align.VehicleMaskParams(dilate_px=0.0, extend_px=0.0)
    mask = vmask([box], [], None, params=params)[0][0].numpy()
    xs, ys = box.vertices[:, 0], box.vertices[:, 1]
    assert not mask[int(ys.min()) + 2, int(xs.min()) + 2]  # a bounding-box corner is not covered
    assert mask[200, 200]
    assert mask.sum() < 0.5 * (xs.max() - xs.min()) * (ys.max() - ys.min())


def test_a_matched_vehicle_is_masked_at_both_positions_with_the_small_margin_only():
    T = similarity(1.0, 0.0, 0.0, 0.0)
    rgb, thermal = [rect(300, 200)], [rect(318, 200)]  # in thermal the same vehicle is 18 px away
    mask, counts = vmask(rgb, thermal, T)
    m = mask[0].numpy()
    assert counts.matched == 1 and counts.static == counts.moved == counts.unsure == 0
    assert m[200, 300] and m[200, 318]  # where RGB has it and where the thermal pixels are
    # rect(300, 200) spans columns 289..311, its thermal twin 307..329; only 3 px is added
    assert m[200, 289 - 3] and not m[200, 289 - 3 - 1]
    assert (
        m[200, 329 + 3] and not m[200, 329 + 3 + 1]
    )  # no 17 px motion margin: both positions known


def test_thermal_only_boxes_are_masked_where_their_pixels_land():
    T = similarity(1.0, 0.0, 25.0, 0.0)  # thermal pixel (c, r) lands on (c + 25, r)
    mask, counts = vmask([], [rect(100, 100)], T)
    assert counts.thermal_only == 1 and counts.matched == 0
    m = mask[0].numpy()
    assert m[100, 125] and not m[100, 100]


def test_thermal_boxes_that_are_copies_of_rgb_boxes_add_nothing():
    T = similarity(1.0, 0.0, 25.0, 0.0)
    rgb = [rect(100, 100), rect(300, 200)]
    copies = [rect(100, 100), rect(300, 200)]  # labels_copied: the thermal boxes are the RGB ones
    with_copies, counts = vmask(rgb, copies, T)
    rgb_only, _ = vmask(rgb, [], None)
    assert torch.equal(with_copies, rgb_only)  # warping a copy would put a spurious box 25 px away
    assert counts.thermal_only == 0 and counts.matched == 0


def test_a_static_vehicle_gets_the_small_margin():
    cars = [(300, 250, 44, 20)]
    luma = _scene(1, cars)
    mask, counts = vmask([rect(300, 250, 44, 20)], [], similarity(1, 0, 0, 0), _images(luma, luma))
    assert counts.static == 1 and counts.moved == counts.unsure == 0
    m = mask[0].numpy()
    assert m[250, 300 + 22 + 3] and not m[250, 300 + 22 + 3 + 2]  # the 3 px margin and no more


def test_a_vehicle_that_shows_up_elsewhere_in_thermal_is_extended_along_its_long_axis():
    car = (300, 250, 44, 20)
    rgb_luma = _scene(2, [car])
    thermal_luma = _scene(2, [(304, 250, 44, 20)])  # in thermal the car sits 4 px further right
    box = rect(300, 250, 44, 20)  # long axis horizontal
    mask, counts = vmask([box], [], similarity(1, 0, 0, 0), _images(rgb_luma, thermal_luma))
    assert counts.moved == 1 and counts.static == 0
    m = mask[0].numpy()
    assert m[250, 300 + 22 + 17] and m[250, 300 - 22 - 17]  # 17.4 px past each end of the long axis
    assert not m[250, 300 + 22 + 17 + 5]
    assert not m[250 - 10 - 8, 300] and m[250 - 10 - 3, 300]  # sideways: only the 3 px margin


def test_a_weak_peak_or_a_tiny_box_is_unsure_and_treated_like_moved():
    flat = np.full((H, W), 0.5, np.float32)  # no texture: no clear correlation peak
    box = rect(300, 250, 44, 20)
    mask, counts = vmask([box], [], similarity(1, 0, 0, 0), _images(flat, flat))
    assert counts.unsure == 1 and counts.static == 0
    assert mask[0].numpy()[250, 300 + 22 + 17]  # extended all the same
    luma = _scene(3, [(300, 250, 6, 5)])  # a 6x5 px "box": its window is under min_window_px
    tiny = align.VehicleMaskParams(**{**VM.__dict__, "window_margin_px": 4})
    _, counts = vmask([rect(300, 250, 6, 5)], [], similarity(1, 0, 0, 0), _images(luma, luma), tiny)
    assert counts.unsure == 1


def test_without_images_every_unmatched_vehicle_is_unsure():
    _, counts = vmask([rect(300, 250), rect(100, 100)], [], similarity(1, 0, 0, 0), NO_IMAGES)
    assert counts.unsure == 2


def test_extension_follows_an_oriented_long_axis():
    box = rotated(300, 250, 44, 18, 40)
    u = np.array([math.cos(math.radians(40)), math.sin(math.radians(40))])
    mask, _ = vmask([box], [], None)  # no images: unsure -> extended
    m = mask[0].numpy()
    tip = np.array([300.0, 250.0]) + u * (22 + 17)
    assert m[int(round(tip[1])), int(round(tip[0]))]
    side = np.array([300.0, 250.0]) + np.array([-u[1], u[0]]) * (9 + 17)
    assert not m[int(round(side[1])), int(round(side[0]))]  # not isotropic


def test_every_rgb_box_lands_in_exactly_one_branch():
    car = (300, 250, 44, 20)
    luma = _scene(4, [car])
    rgb = [rect(300, 250, 44, 20), rect(100, 100), rect(450, 400)]
    thermal = [rect(103, 101), rect(451, 399)]  # two matched, one copied-free unmatched RGB box
    _, counts = vmask(rgb, thermal, similarity(1, 0, 0, 0), _images(luma, luma))
    assert counts.matched == 2
    assert counts.matched + counts.static + counts.moved + counts.unsure == len(rgb)


def test_vehicle_fraction_and_the_dense_flag():
    mask = torch.zeros(1, 10, 10, dtype=torch.bool)
    mask[:, :3, :] = True
    assert align.vehicle_fraction(mask) == pytest.approx(0.3)
    assert not align.is_dense(0.30, 0.30)  # "over" 30%, so exactly 30% is not flagged
    assert align.is_dense(0.3001, 0.30)


def test_moving_displacements_are_the_residuals_of_the_moving_boxes_only():
    T = similarity(1.0, 0.0, 0.0, 0.0)
    rgb = [rect(100, 100), rect(300, 200), rect(500, 300)]
    thermal = [rect(101, 100), rect(318, 200), rect(499, 300)]  # box 1 moved 18 px
    matches = align.match_boxes(thermal, rgb, CFG)
    moved = align.moving_displacements(matches, T, CFG.moving_min_residual_px)
    assert moved.tolist() == pytest.approx([18.0])
    assert align.moving_displacements(matches, None, 4.0).size == 0
