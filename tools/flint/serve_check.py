"""tools/flint/serve_check.py - the engine protocol of strata-flint --serve, driven the way serve/server.py drives an engine.

    python tools/flint/serve_check.py --engine build/strata-flint --model flint-UD-Q4_K_XL.gguf

Checks, on the real model (it needs the GPU and the file): READY, a greedy request against the command-line mode's text, the
reuse of the cache by a second request that extends the first, a prompt that diverges (read again), a STOP, and the refusal of
a bad request.  Exit code 0 when all pass.
"""
import argparse
import queue
import subprocess
import sys
import threading
import time

PROMPT = [630, 6963, 297, 9865, 332]       # "The capital of France is"


class Engine:
    def __init__(self, exe, model, extra):
        self.p = subprocess.Popen([exe, "--serve", "--model", model, "--max-context", "4096", *extra], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.q = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.p.stdout:
            self.q.put(line.rstrip("\n"))
        self.q.put(None)

    def line(self, timeout=600):
        l = self.q.get(timeout=timeout)
        if l is None:
            raise RuntimeError("the engine ended")
        return l

    def send(self, text):
        self.p.stdin.write(text + "\n")
        self.p.stdin.flush()

    def gen(self, max_new, ids, keys="", stop_after=None):
        self.send(f"GEN {max_new} {keys + ' ' if keys else ''}{','.join(map(str, ids))}")
        toks, pp, done, stopped = [], [], None, False
        while True:
            l = self.line()
            if l.startswith("T "):
                toks.append(int(l.split()[1]))
                if stop_after and len(toks) == stop_after and not stopped:
                    self.send("STOP")
                    stopped = True
            elif l.startswith("PP "):
                pp.append(l)
            elif l.startswith("DONE"):
                done = l.split()
                break
            elif l.startswith("ERR"):
                return toks, pp, l
        return toks, pp, done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--extra", nargs="*", default=[])
    a = ap.parse_args()
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail else ""))
        ok = ok and bool(cond)

    # the command-line mode's greedy text is the reference
    ref = subprocess.run([a.engine, "--model", a.model, "--tokens", ",".join(map(str, PROMPT)), "--n", "8"],
                         capture_output=True, text=True, timeout=900).stdout.split("generated:")[1].split()
    ref = [int(x) for x in ref]
    t0 = time.time()
    e = Engine(a.engine, a.model, a.extra)
    first = e.line()
    check("READY", first.startswith("READY "), f"({first}, {time.time() - t0:.0f} s)")
    toks, pp, done = e.gen(8, PROMPT)
    check("greedy request equals the command-line mode", toks == ref, f"{toks} vs {ref}")
    check("PP progress and DONE", bool(pp) and done and done[0] == "DONE" and int(done[1]) == 8, f"{done}")
    check("first request reuses nothing", done and int(done[8]) == 0)
    # the second request extends the first (prompt + its answer): only the new end is read
    ext = PROMPT + toks[:6] + [11, 12]
    toks2, pp2, done2 = e.gen(4, ext)
    check("an extending request reuses the cache", done2 and int(done2[8]) >= len(PROMPT) + 5 and int(done2[14]) <= 3,
          f"reused {done2[8] if done2 else '?'}, read {done2[14] if done2 else '?'}")
    # the same request again, read from the cache: the same greedy text
    toks3, _, done3 = e.gen(4, ext)
    check("the same request again gives the same text", toks3 == toks2, f"{toks3} vs {toks2}")
    # a prompt that diverges at its start is read again
    other = [630, 6963, 9999, 9865, 332]
    _, _, done4 = e.gen(2, other)
    check("a diverging prompt is read from the start", done4 and int(done4[8]) in (0, 2), f"reused {done4[8] if done4 else '?'}")
    # sampling parameters are accepted
    toks5, _, done5 = e.gen(4, PROMPT, "temperature=0.7 top_p=0.9 top_k=20 seed=7")
    check("a sampled request", done5 and int(done5[1]) == 4 and len(toks5) == 4)
    # STOP ends a request
    toks6, _, done6 = e.gen(200, PROMPT, stop_after=3)
    check("STOP ends the request", done6 and done6[5] == "cancel" and len(toks6) < 200, f"{len(toks6)} tokens, {done6[5] if done6 else '?'}")
    # a bad request is refused, and the engine goes on
    e.send("GEN 4 99999999")
    bad = e.line()
    check("an id outside the vocabulary is refused", bad.startswith("ERR"), bad)
    e.send("HELLO")
    check("an unknown command is refused", e.line().startswith("ERR"))
    toks7, _, done7 = e.gen(2, PROMPT)
    check("the engine goes on after an error", done7 and int(done7[1]) == 2)
    e.p.stdin.close()
    e.p.wait(timeout=60)
    check("the engine ends when its input ends", e.p.returncode == 0, f"rc {e.p.returncode}")
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
