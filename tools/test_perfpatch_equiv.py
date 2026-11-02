"""test_perfpatch_equiv.py - prove mikelou1's hotspot rewrites are numerically
identical to baseline (CPU, no GPU). Gate for adoption into our solver."""
import torch

torch.manual_seed(0)

ARC_TOKENS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
EOS_ID = 15
_CACHE = {}


def arc_ids(device):
    if str(device) not in _CACHE:
        _CACHE[str(device)] = torch.tensor(ARC_TOKENS, dtype=torch.long,
                                           device=device)
    return _CACHE[str(device)]


def nll_base(logits, scores):
    n = logits.size(0)
    return (torch.tensor(scores, dtype=torch.float32).view(n, 1)
            - logits.float().cpu().log_softmax(-1))


def nll_fast(logits, scores):
    n = logits.size(0)
    logits_f = logits.float()
    arc_logits = logits_f.index_select(-1, arc_ids(logits.device))
    return (torch.as_tensor(scores, dtype=torch.float32,
                            device=logits.device).view(n, 1)
            + torch.logsumexp(logits_f, dim=-1, keepdim=True)
            - arc_logits).cpu()


def scores_base(batch_logits, queries, answers, pad=13):
    bl = batch_logits.float().cpu().log_softmax(-1)
    out = []
    for logits, q, a in zip(bl, queries, answers):
        al = logits[len(q) - 1:len(q) - 1 + len(a)]
        out.append(-al[torch.arange(len(a)), a].sum().item())
    return out


def scores_fast(batch_logits, queries, answers):
    bl = batch_logits.float()
    norm = torch.logsumexp(bl, dim=-1)
    out = []
    for r, (q, a) in enumerate(zip(queries, answers)):
        pos = torch.arange(len(q) - 1, len(q) - 1 + len(a))
        tgt = torch.tensor(a, dtype=torch.long)
        out.append(-(bl[r, pos, tgt] - norm[r, pos]).sum().item())
    return out


def main():
    import itertools
    for trial in range(5):
        n, V = 4, 151936
        logits = torch.randn(n, V) * 3
        scores = [float(x) for x in torch.randn(n) * 2]
        a = nll_base(logits, scores)[:, ARC_TOKENS]
        b = nll_fast(logits, scores)
        assert a.shape == b.shape == (n, len(ARC_TOKENS)), (a.shape, b.shape)
        assert torch.allclose(a, b, atol=1e-4), (trial, (a - b).abs().max())
    print("NLL equivalent OK")
    # integrated loop check: baseline token-id indexing vs patched positional
    # indexing must yield identical (score, token) pairs, incl. EOS=15
    max_score = 1.6
    for trial in range(5):
        n, V = 4, 200
        logits = torch.randn(n, V) * 3
        scores = [float(x) for x in torch.randn(n) * 2]
        full = nll_base(logits, scores)
        gath = nll_fast(logits, scores)
        for i in range(n):
            ref, new = [], []
            for t in ARC_TOKENS:
                s = full[i, t].item()
                if s < max_score:
                    ref.append((s, t))
            for pos, t in enumerate(ARC_TOKENS):
                s = gath[i, pos].item()
                if s < max_score:
                    new.append((s, t))
            assert [t for _, t in ref] == [t for _, t in new], (trial, i)
            assert all(abs(a - b) < 1e-5 for (a, _), (b, _) in zip(ref, new)), (trial, i)
    print("loop equivalent OK (incl. EOS path)")
    for trial in range(5):
        B, L = 3, 40
        bl = torch.randn(B, L, 512) * 3
        qs = [[1] * 10, [2] * 15, [3] * 8]
        an = [[4] * 6, [5] * 9, [6] * 5]
        a = scores_base(bl, qs, an)
        b = scores_fast(bl, qs, an)
        assert len(a) == len(b) == 3
        assert all(abs(x - y) < 1e-3 for x, y in zip(a, b)), (a, b)
    print("scores equivalent OK")


if __name__ == "__main__":
    main()
