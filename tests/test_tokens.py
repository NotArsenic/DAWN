"""docs/conventions.md section 2.3: the one token layout (dawn/flow/tokens.py).

Everything asks the layout for its sizes, with one exception: `test_layout_matches_the_plan` pins
the plan's numbers (16 brightness values, then 32 colour values, 1,280 tokens) as the spec.
"""

import pytest

torch = pytest.importorskip("torch")

from dawn.flow.tokens import TokenLayout  # noqa: E402

TINY = {"num_bins": 3, "grid": (2, 3), "block": 4}


def _parts(layout, batch=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    gy, gx = layout.grid
    parts = {
        "grid": torch.randn(batch, 2, layout.num_bins, gy, gx, generator=g),
        "ab": torch.randn(batch, 2, *layout.color_size, generator=g),
    }
    if layout.correction:
        parts["corr"] = torch.randn(batch, 1, *layout.color_size, generator=g)
    return parts


def test_layout_matches_the_plan():
    off = TokenLayout(num_bins=8, grid=(32, 40), block=4, correction=False)
    on = TokenLayout(num_bins=8, grid=(32, 40), block=4, correction=True)
    assert off.num_tokens == on.num_tokens == 1280
    assert off.color_size == (128, 160)
    assert (off.dim, on.dim) == (48, 64)
    assert off.segments == {"brightness": slice(0, 16), "color": slice(16, 48)}
    assert on.segments == {
        "brightness": slice(0, 16),
        "color": slice(16, 48),
        "correction": slice(48, 64),
    }


@pytest.mark.parametrize("correction", [False, True])
def test_pack_unpack_round_trip(correction):
    layout = TokenLayout(**TINY, correction=correction)
    parts = _parts(layout)
    tokens = layout.pack(parts)
    assert tokens.shape == (2, layout.num_tokens, layout.dim)
    back = layout.unpack(tokens)
    assert set(back) == set(parts)
    for name in parts:
        assert torch.equal(back[name], parts[name]), name


def test_the_base_layout_is_the_same_with_or_without_the_correction():
    off, on = TokenLayout(**TINY), TokenLayout(**TINY, correction=True)
    parts = _parts(on)
    base = {k: v for k, v in parts.items() if k != "corr"}
    assert torch.equal(on.pack(parts)[..., : off.dim], off.pack(base))


@pytest.mark.parametrize("correction", [False, True])
def test_token_n_is_cell_y_x_and_the_colour_block_it_covers(correction):
    """A marker pixel lands in token y*gx + x at the value the layout says."""
    layout = TokenLayout(**TINY, correction=correction)
    gy, gx = layout.grid
    block = layout.block
    for y, x, py, px, ch in [(0, 0, 0, 0, 0), (1, 2, 3, 1, 1), (1, 0, 2, 3, 0), (0, 2, 1, 2, 1)]:
        parts = {k: torch.zeros_like(v) for k, v in _parts(layout, batch=1).items()}
        parts["ab"][0, ch, block * y + py, block * x + px] = 7.0
        tokens = layout.pack(parts)
        n = y * gx + x
        v = layout.segments["color"].start + ch * block * block + py * block + px
        assert tokens[0, n, v] == 7.0 and tokens.abs().sum() == 7.0
        if correction:
            parts["ab"].zero_()
            parts["corr"][0, 0, block * y + py, block * x + px] = 5.0
            tokens = layout.pack(parts)
            v = layout.segments["correction"].start + py * block + px
            assert tokens[0, n, v] == 5.0 and tokens.abs().sum() == 5.0


def test_brightness_values_are_bin_major_gain_then_offset():
    layout = TokenLayout(**TINY)
    gy, gx = layout.grid
    parts = {k: torch.zeros_like(v) for k, v in _parts(layout, batch=1).items()}
    parts["grid"][0, 1, 2, 1, 2] = 3.0  # offset, bin 2, cell (1, 2)
    tokens = layout.pack(parts)
    assert tokens[0, 1 * gx + 2, 2 * 2 + 1] == 3.0 and tokens.abs().sum() == 3.0


def test_pack_and_unpack_reject_wrong_shapes_and_parts():
    layout = TokenLayout(**TINY, correction=True)
    parts = _parts(layout)
    with pytest.raises(ValueError, match="corr"):
        layout.pack({k: v for k, v in parts.items() if k != "corr"})
    with pytest.raises(ValueError, match="ab"):
        layout.pack({**parts, "ab": parts["ab"][..., :-1]})
    with pytest.raises(ValueError, match="tokens"):
        layout.unpack(torch.zeros(1, layout.num_tokens, layout.dim - 1))
    off = TokenLayout(**TINY)
    with pytest.raises(ValueError, match="corr"):
        off.pack(parts)  # a correction nobody asked for is a layout mismatch, not ignored


def test_stats_vector_follows_the_value_order():
    layout = TokenLayout(**TINY, correction=True)
    stats = {
        "brightness": {
            "mean": torch.arange(2 * 3, dtype=torch.float32).reshape(2, 3),  # [c, k]
            "std": torch.arange(2 * 3, dtype=torch.float32).reshape(2, 3) + 1,
        },
        "color": {"mean": torch.tensor([10.0, 20.0]), "std": torch.tensor([1.0, 2.0])},
        "correction": {"mean": torch.tensor([-1.0]), "std": torch.tensor([0.5])},
    }
    mean, std = layout.stats_vector(stats)
    assert mean.shape == std.shape == (layout.dim,)
    for c in range(2):
        for k in range(3):
            assert mean[k * 2 + c] == stats["brightness"]["mean"][c, k]
            assert std[k * 2 + c] == stats["brightness"]["std"][c, k]
    color = layout.segments["color"]
    assert (mean[color][:16] == 10.0).all() and (mean[color][16:] == 20.0).all()
    assert (std[color][:16] == 1.0).all() and (std[color][16:] == 2.0).all()
    assert (mean[layout.segments["correction"]] == -1.0).all()


def test_stats_vector_rejects_missing_or_misshapen_stats():
    layout = TokenLayout(**TINY)
    good = {
        "brightness": {"mean": torch.zeros(2, 3), "std": torch.ones(2, 3)},
        "color": {"mean": torch.zeros(2), "std": torch.ones(2)},
    }
    layout.stats_vector(good)
    with pytest.raises(ValueError, match="brightness"):
        layout.stats_vector(
            {**good, "brightness": {"mean": torch.zeros(2, 4), "std": torch.ones(2, 4)}}
        )
    with pytest.raises(ValueError, match="color"):
        layout.stats_vector({"brightness": good["brightness"]})


@pytest.mark.parametrize("correction", [False, True])
def test_weights_are_normalised_per_segment_and_sample_and_broadcast_correctly(correction):
    layout = TokenLayout(**TINY, correction=correction)
    gy, gx = layout.grid
    g = torch.Generator().manual_seed(4)
    node_weights = torch.rand(2, 3, gy, gx, generator=g) * 10
    node_weights[1] = 0  # sample 1 has no brightness weight at all
    fraction = torch.rand(2, 1, *layout.color_size, generator=g)
    weights = layout.weights(node_weights, fraction)
    assert weights.shape == (2, layout.num_tokens, layout.dim)
    for name, seg in layout.segments.items():
        mean = weights[:, :, seg].mean(dim=(1, 2))
        if name == "brightness":
            assert torch.isclose(mean[0], torch.tensor(1.0), atol=1e-5)
            assert mean[1] == 0.0 and torch.isfinite(weights).all()
        else:
            assert torch.allclose(mean, torch.ones(2), atol=1e-5), name
    # Brightness: node weights, the same for gain and offset of one (cell, bin).
    bright = weights[0, :, layout.segments["brightness"]].reshape(gy * gx, 3, 2)
    assert torch.equal(bright[..., 0], bright[..., 1])
    ratio = bright[:, :, 0] / node_weights[0].permute(1, 2, 0).reshape(gy * gx, 3)
    assert torch.allclose(ratio, ratio[0, 0].expand_as(ratio), rtol=1e-4)
    # Colour: the mask fraction of the pixel, the same for a* and b*.
    color = weights[0, :, layout.segments["color"]].reshape(gy * gx, 2, 4, 4)
    assert torch.equal(color[:, 0], color[:, 1])
    block = fraction[0, 0, 0:4, 0:4]  # token 0 covers colour pixels (0:4, 0:4)
    assert torch.allclose(color[0, 0] / block, (color[0, 0] / block)[0, 0].expand(4, 4), rtol=1e-4)


def test_descriptor_round_trips_and_a_mismatch_is_a_clear_error():
    layout = TokenLayout(**TINY, correction=True)
    descriptor = layout.descriptor()
    assert TokenLayout.from_descriptor(descriptor) == layout
    layout.check(descriptor)
    for other in (
        TokenLayout(**TINY),
        TokenLayout(num_bins=3, grid=(2, 4), block=4, correction=True),
    ):
        with pytest.raises(ValueError, match="layout"):
            other.check(descriptor)


def test_from_config_reads_the_resolution_from_the_encoder_input_size():
    import omegaconf

    cfg = omegaconf.OmegaConf.to_container(omegaconf.OmegaConf.load("configs/flow.yaml"))
    layout = TokenLayout.from_config(cfg)
    h, w = cfg["encoder"]["input_size"]
    assert layout.grid == (h // cfg["encoder"]["patch_size"], w // cfg["encoder"]["patch_size"])
    assert layout.num_bins == cfg["grid"]["num_bins"] and layout.block == cfg["color"]["block"]
    assert layout.correction is cfg["brightness_correction"]["enabled"]
    cfg["brightness_correction"]["enabled"] = True
    assert TokenLayout.from_config(cfg).dim == layout.dim + layout.block**2
    cfg["encoder"]["input_size"] = [100, 100]
    with pytest.raises(ValueError, match="multiple"):
        TokenLayout.from_config(cfg)
