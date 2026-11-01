"""tools/flint/compare.py - strata-flint against llama.cpp on the same Flint GGUF and token ids (standard library only).

    # llama.cpp side (the dump tool is tools/qwen35/q36_llama_dump.cpp, built against a llama.cpp that knows `flint`)
    flint_llama_dump --model F.gguf --tokens 1,2,3 --n 1 --ngl 0 --dump-layers ref.layers --dump-full-logits ref.logits
    # Strata side
    strata-flint --model F.gguf --tokens 1,2,3 --n 1 --dump-layers st.layers --dump-full-logits st.logits
    python tools/flint/compare.py --layers ref.layers st.layers --logits ref.logits st.logits

The layer files are [position][layer][n_embd] f32 (the residual after each layer); the logit files hold the last
prompt position's [vocab] f32 followed by one row per generated / forced token (both tools write the same rows when
strata-flint is given llama.cpp's ids with --force); every common row is compared, and the means are printed.
"""
import argparse
import array
import math


def load(path, n=None):
    a = array.array("f")
    with open(path, "rb") as f:
        a.fromfile(f, n) if n else a.frombytes(f.read())
    return a


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", nargs=2)
    ap.add_argument("--logits", nargs=2)
    ap.add_argument("--n-layers", type=int, default=49)
    ap.add_argument("--n-embd", type=int, default=2048)
    ap.add_argument("--vocab", type=int, default=150144)
    a = ap.parse_args()
    if a.layers:
        r, s = load(a.layers[0]), load(a.layers[1])
        row = a.n_layers * a.n_embd
        n = min(len(r), len(s)) // row
        print(f"positions compared: {n}")
        print("layer  rel.err(last pos)  rel.err(worst pos)")
        for l in range(a.n_layers):
            errs = []
            for p in range(n):
                o = p * row + l * a.n_embd
                d = sum((r[o + i] - s[o + i]) ** 2 for i in range(a.n_embd))
                m = sum(r[o + i] ** 2 for i in range(a.n_embd))
                errs.append(math.sqrt(d) / (math.sqrt(m) + 1e-12))
            print(f"{l:5d}  {errs[-1]:.5f}            {max(errs):.5f}")
    if a.logits:
        r, s = load(a.logits[0]), load(a.logits[1])
        n = min(len(r), len(s)) // a.vocab
        tot_kl, agree = 0.0, 0
        print("row   top1(ref) top1(strata)   KL(ref||strata)   max|dlogit|")
        for i in range(n):
            rr, ss = r[i * a.vocab:(i + 1) * a.vocab], s[i * a.vocab:(i + 1) * a.vocab]

            def lsm(x):
                m = max(x)
                z = math.log(sum(math.exp(v - m) for v in x))
                return [v - m - z for v in x]

            pr, ps = lsm(rr), lsm(ss)
            kl = sum(math.exp(x) * (x - y) for x, y in zip(pr, ps))
            ra = max(range(a.vocab), key=rr.__getitem__)
            sa = max(range(a.vocab), key=ss.__getitem__)
            tot_kl += kl
            agree += ra == sa
            print(f"{i:3d}   {ra:8d}  {sa:8d}      {kl:12.6f}   {max(abs(x - y) for x, y in zip(rr, ss)):9.4f}")
        print(f"rows {n}: mean KL {tot_kl / max(n, 1):.6f}, top-1 agreement {agree}/{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
