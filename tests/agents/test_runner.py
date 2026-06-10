"""Tests for the graph runner helpers."""

from __future__ import annotations

from marimo_flow.agents.runner import _safe_filename


def test_safe_filename_strips_windows_reserved_chars():
    # Plain snapshot labels stay untouched.
    assert _safe_filename("end") == "end"
    assert _safe_filename("node-RouteNode") == "node-RouteNode"
    # Reserved chars fold to underscores (defence-in-depth for NTFS).
    assert _safe_filename('a<b>c"d|e?f*g/h\\i') == "a_b_c_d_e_f_g_h_i"
    assert ":" not in _safe_filename("node-TriageNode:deadbeef")
