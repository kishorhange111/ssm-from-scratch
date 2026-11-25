import pytest
import torch

from ssm.model import LM, Config, Mamba2
from ssm.scan import scan_parallel, scan_sequential, segsum, ssd_chunked
from ssm.train import mqar_batch, train_lm, train_mqar


def _inputs(b=2, l=37, h=3, p=4, n=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    kw = {"dtype": torch.float64, "generator": g}
    return (torch.randn(b, l, h, p, **kw), -torch.rand(b, l, h, **kw),
            torch.randn(b, l, h, n, **kw), torch.randn(b, l, h, n, **kw))  # fmt: skip


def test_parallel_scan_matches_recurrence():
    x, a, B, C = _inputs()
    y0, s0 = scan_sequential(x, a, B, C)
    y1, s1 = scan_parallel(x, a, B, C)
    torch.testing.assert_close(y1, y0)
    torch.testing.assert_close(s1, s0)


@pytest.mark.parametrize("chunk", [1, 4, 16, 64])  # 37 is not a multiple of any of them
def test_chunked_ssd_matches_recurrence(chunk):
    x, a, B, C = _inputs()
    y0, s0 = scan_sequential(x, a, B, C)
    y2, s2 = ssd_chunked(x, a, B, C, chunk)
    torch.testing.assert_close(y2, y0)
    torch.testing.assert_close(s2, s0)


def test_ssd_carries_initial_state():
    x, a, B, C = _inputs(l=20)
    _, mid = scan_sequential(x[:, :12], a[:, :12], B[:, :12], C[:, :12])
    y_ref, s_ref = scan_sequential(x[:, 12:], a[:, 12:], B[:, 12:], C[:, 12:], h0=mid)
    y, s = ssd_chunked(x[:, 12:], a[:, 12:], B[:, 12:], C[:, 12:], chunk=4, h0=mid)
    torch.testing.assert_close(y, y_ref)
    torch.testing.assert_close(s, s_ref)


def test_segsum():
    x = torch.tensor([1.0, 2.0, 3.0])
    s = segsum(x)
    assert s[2, 0] == 5.0 and s[2, 1] == 3.0 and s[1, 1] == 0.0
    assert torch.isinf(s[0, 1])


def test_ssd_is_attention_with_a_decay_mask():
    """Duality: with one chunk, y = (exp(segsum(a)) ∘ C Bᵀ) x, i.e. masked linear attention."""
    x, a, B, C = _inputs(b=1, l=8, h=1, p=2, n=3)
    L = torch.exp(segsum(a[0, :, 0]))
    M = L * (C[0, :, 0] @ B[0, :, 0].t())
    torch.testing.assert_close(ssd_chunked(x, a, B, C, chunk=8)[0][0, :, 0], M @ x[0, :, 0])


def test_block_recurrent_step_matches_parallel_forward():
    torch.manual_seed(0)
    m = Mamba2(32, d_state=8, head_dim=8, chunk=4).double()
    u = torch.randn(2, 11, 32, dtype=torch.float64)
    st, outs = m.init_state(2, dtype=torch.float64), []
    for t in range(11):
        o, st = m.step(u[:, t], st)
        outs.append(o)
    torch.testing.assert_close(torch.stack(outs, 1), m(u))


def test_lm_decoding_matches_forward():
    torch.manual_seed(0)
    lm = LM(Config(d_model=32, n_layers=2, d_state=8)).double()
    idx = torch.randint(0, 256, (2, 9))
    sts, outs = [b.init_state(2, dtype=torch.float64) for b in lm.mix], []
    for t in range(9):
        o, sts = lm.step(idx[:, t], sts)
        outs.append(o)
    torch.testing.assert_close(torch.stack(outs, 1), lm(idx))


def test_mqar_batch_is_well_formed():
    x, y = mqar_batch(4, 8, torch.Generator().manual_seed(0))
    assert x.shape == (4, 32)
    for i in range(4):
        pairs = dict(zip(x[i, :16:2].tolist(), x[i, 1:16:2].tolist(), strict=True))
        assert len(pairs) == 8  # distinct keys
        for t in range(16, 32, 2):
            assert y[i, t] == pairs[int(x[i, t])] == x[i, t + 1]
        assert (y[i, :16] == -100).all() and (y[i, 17::2] == -100).all()


def test_tiny_runs_complete():
    r = train_mqar(Config(d_model=32, n_layers=1, d_state=8, mlp=False), 4, 20, "cpu", bs=8)
    assert 0.0 <= r["accuracy"] <= 1.0
    g = torch.Generator().manual_seed(0)
    data = torch.randint(0, 256, (20000,), generator=g).numpy().astype("uint8")
    r = train_lm(Config(d_model=32, n_layers=1, d_state=8), 5, "cpu", ctx=32, bs=4,
                 data=(data, data))  # fmt: skip
    assert r["val_bpb"] > 0
