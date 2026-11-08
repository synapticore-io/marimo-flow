import numpy as np

from marimo_flow.core.cavity_reference import centerlines, minimum, solve_cavity

# Oueslati et al., C. R. Mecanique 339 (2011) 655-665, Table 1, Re = 100.
OUESLATI_U_MIN = -0.21361
OUESLATI_W_MIN = -0.24971


def _extrema(n: int) -> tuple[float, float]:
    s, ux, uz = centerlines(solve_cavity(100, n))
    return minimum(s, ux)[0], minimum(s, uz)[0]


def test_re100_matches_literature_after_richardson():
    """Second-order Richardson extrapolation of 16^3/32^3 hits the 3D benchmark."""
    coarse, fine = _extrema(16), _extrema(32)
    u_ext, w_ext = (f + (f - c) / 3 for c, f in zip(coarse, fine, strict=True))
    assert abs(u_ext - OUESLATI_U_MIN) / abs(OUESLATI_U_MIN) < 0.015
    assert abs(w_ext - OUESLATI_W_MIN) / abs(OUESLATI_W_MIN) < 0.015
    # The fine grid alone sits between the coarse grid and the literature value.
    assert coarse[0] > fine[0] > OUESLATI_U_MIN


def test_smooth_lid_flow_is_mirror_symmetric():
    vel = solve_cavity(10, 16, smooth=True, t_end=6.0)
    # The lid profile is symmetric in y, so u_x, u_z mirror and u_y flips sign.
    mirrored = vel[:, ::-1]
    assert np.allclose(mirrored[..., [0, 2]], vel[..., [0, 2]], atol=1e-12)
    assert np.allclose(mirrored[..., 1], -vel[..., 1], atol=1e-12)
    s, ux, _ = centerlines(vel)
    u_min, z_min = minimum(s, ux)
    assert -0.12 < u_min < -0.09
    assert 0.5 < z_min < 0.6
