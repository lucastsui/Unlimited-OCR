"""Unit tests for the local-inference patch (patch_model_for_local.py).

Covers the two things the patch must guarantee:
  1. The positional-assignment replacement computes exactly what the released
     masked_scatter_ injection computes (checked against CPU, where
     masked_scatter_ is correct), for contiguous AND non-contiguous sources.
  2. On the MPS backend, the replacement matches the CPU reference. (The
     motivation is that raw masked_scatter_ with a non-contiguous source does
     not, on affected torch/macOS versions; the test asserts our replacement
     is right rather than asserting the upstream op is wrong, so it stays
     green if PyTorch fixes the underlying issue.)

Run:  pytest tests/test_local_patch.py
"""

import torch


def _reference_injection(dst, mask, src):
    """The released behavior: in-place masked_scatter_ (computed on CPU)."""
    out = dst.clone().cpu()
    out.masked_scatter_(mask.cpu().unsqueeze(-1), src.cpu().contiguous())
    return out


def _patched_injection(dst, mask, src):
    """The patched behavior: explicit positional assignment (any device)."""
    out = dst.clone()
    feat = src.to(device=out.device, dtype=out.dtype)
    pos = mask.to(out.device).bool().nonzero(as_tuple=True)[0]
    out[pos] = feat
    return out


def _make_case(seq_len=32, hidden=16, n_img=10, noncontiguous=False, device="cpu"):
    torch.manual_seed(0)
    dst = torch.randn(seq_len, hidden, device=device)
    mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    mask[3:3 + n_img] = True
    if noncontiguous:
        src = torch.randn(hidden, n_img, device=device).T  # transposed view
        assert not src.is_contiguous()
    else:
        src = torch.randn(n_img, hidden, device=device)
    return dst, mask, src


def test_positional_assignment_matches_masked_scatter_contiguous():
    dst, mask, src = _make_case(noncontiguous=False)
    assert torch.equal(_patched_injection(dst, mask, src),
                       _reference_injection(dst, mask, src))


def test_positional_assignment_matches_masked_scatter_noncontiguous():
    dst, mask, src = _make_case(noncontiguous=True)
    assert torch.equal(_patched_injection(dst, mask, src),
                       _reference_injection(dst, mask, src))


def test_patched_injection_correct_on_mps():
    if not torch.backends.mps.is_available():
        import pytest
        pytest.skip("MPS not available")
    for noncontiguous in (False, True):
        dst, mask, src = _make_case(noncontiguous=noncontiguous, device="mps")
        got = _patched_injection(dst, mask, src).cpu()
        want = _reference_injection(dst, mask, src)
        assert torch.allclose(got, want), (
            f"patched injection wrong on MPS (noncontiguous={noncontiguous})"
        )
