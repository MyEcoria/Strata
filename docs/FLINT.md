# Flint (experimental)

Flint is Recursive's coding model (Hugging Face `RecursiveSI/flint`, GGUF architecture `flint`). It has 39.8 billion
parameters and uses 2.5 billion of them for each token (49 layers, 2 dense and 47 with 256 experts, 6 used per token).
Strata runs it with its own engine, `strata-flint`. This page says what works, what was measured and what does not exist yet.

> **Status.** A first engine, tested on one machine. It runs from the command line on token ids and behind Strata's server
> (OpenAI and Anthropic APIs, reasoning, tool calls, streaming). One request at a time, no pictures. Linux with an AMD card
> only. There is no ready-made engine package that includes it yet, so it is installed from a source checkout. Speed on the test
> machine (16 GB card, 15 GB RAM): a long prompt is read at about 1,150 tokens/s and the answer comes at 55 to 65 tokens/s,
> because there every expert fits in the card or in page-locked RAM. With less RAM the rest is read from the disk and the answer
> is slower (see "Measured").

## What you need

- Linux, an AMD card Strata supports ([AMD_HIP.md](AMD_HIP.md)) or, untested, an NVIDIA card, and a source build of Strata
  (`cmake -DSTRATA_ENABLE_HIP=ON ...`; on Ubuntu `tools/install_amd_build_deps_ubuntu.sh` installs the build packages).
- A Flint GGUF. The one that was tested is Unsloth's `flint-UD-Q4_K_XL.gguf` (25.05 GB, Hugging Face `unsloth/flint-GGUF`).
  Q4 or above is recommended; the smaller files were not tested.
- RAM and disk: the experts that do not fit in the card are kept in RAM, then read from the file. An NVMe disk is needed.

## Serve it

From a source checkout (this builds the engine, downloads the 25 GB file, checks its SHA-256 and writes the config):

```sh
./setup.sh --backend hip --family flint --model UD-Q4_K_XL
```

`strata launch --family flint --model UD-Q4_K_XL` is the same choice in the Terminal, but it downloads a ready-made engine, and
none of the released engine packages includes Flint yet: it says so and stops. Once it runs, the server is the usual one:
`http://127.0.0.1:8080/v1` (OpenAI) and `/v1/messages` (Anthropic). The model card's sampling (temperature 1.0, top_p 0.95,
top_k 20) is the default.

What the server does for Flint (`serve/flint_format.py`, `serve/test_flint.py`): the model opens its turn with `<think>`; the
reasoning is returned apart from the answer. Its tool calls (`<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...`) are
returned as tool calls, also while streaming. The chat template is the model's own (GLM-style: `<|system|>`, `<|user|>`,
`<|assistant|>`, `<|observation|>`). The engine keeps the tokens it holds between requests, so a follow-up turn reads only its new
end. Because the 35 sliding-window layers keep a ring of 1024 cells, a conversation is continued only from at most 512 tokens
before the end of what the engine holds; a prompt that diverges earlier is read again from the start.

Checked on the machine below, through `setup.py` and the server: a question answered with its reasoning apart, a tool call
returned as a tool call and the next turn reading 145 of its tokens from the cache, a streamed reply (first chunk after 0.3 s), the
same question over the Anthropic API. Not measured: how well Flint chooses and fills in tools in a long agent session.

## Run the engine alone

```sh
strata-flint --model flint-UD-Q4_K_XL.gguf --tokens 630,6963,297,9865,332 --n 16
strata-flint --help
```

The ids come from the model's tokenizer (for example `llama-tokenize --ids`). The output is the generated ids.
`tools/flint/serve_check.py` drives the engine protocol on the real model.

## How the experts are stored

Flint does not fit in 16 GB at Q4 (the experts are 21.8 GiB). `strata-flint` keeps them on three levels:

1. **VRAM.** A cache with the same share of the experts in every MoE layer (about 100 to 160 of 256 on a 16 GB card), least
   recently used first.
2. **Host RAM.** The experts that are not in the VRAM cache: page-locked memory that the GPU reads in place, over the bus,
   when a layer asks for one of them (a layer's experts that are in neither level are read from the file into it first). An
   expert asked for 12 times outside the VRAM cache is swapped with the least recently used one of the cache, in the background
   (`STRATA_FLINT_PROMOTE_AFTER`). The area is sized from `MemAvailable` less 2.5 GiB, at most 70% of the RAM, and never more than
   the experts the VRAM cache leaves out (`--host-cache GiB`; `--no-pin-host` turns the page-locking off, which is slower).
3. **The file.** An expert in neither level is read with parallel `O_DIRECT` reads. The disk reached about 1.7 GB/s for these.

At start the first experts of every layer fill the first two levels (about 21 GiB, 14 s, once). On the test machine the two levels
hold all of Flint's experts (VRAM 11.3 GiB, RAM 10.2 GiB), so a decode never touches the file. The token-embedding table (0.3 GiB)
is kept in RAM so that its VRAM goes to the cache: every 0.3 GiB of VRAM was worth about 3 tokens/s of decode.

**A long prompt** (more than 64 tokens) takes another path, built from the pieces Strata uses for its other models on this card:
the dense projections are FP16 GEMMs on the matrix cores, the attention is batched GEMMs per KV head with the sink in the softmax,
and the routed experts go through llama.cpp's int8 MMQ kernels in groups of 16 experts. A thread has each group's experts waiting
in a ring of buffers (copied from the VRAM cache, from the host tier, or read from the file by a second thread a few layers
ahead), so the GPU does not wait for the host. The prompt is cut into near-equal chunks of at most 2048 tokens (`--chunk`):
every chunk is a pass over all the experts. A bigger chunk reads a long prompt faster (1,345 tokens/s at 4096) but its buffers take
VRAM from the expert cache, and the decode loses speed (about 3 tokens/s per 1000 tokens of chunk). Needs a build with
`STRATA_PREFILL_MMQ` (setup turns it on for AMD).

## Measured

Machine: RX 9070 (gfx1201, 16 GB), Ryzen 5 5600, 15 GB RAM (about 2.6 GB left while Flint runs), NVMe (about 1.7 GB/s for
1-2 MB random reads), Ubuntu 26.10, engine built from this tree, defaults, a context of 16,384. File: `flint-UD-Q4_K_XL.gguf`.

| | Result |
| --- | --- |
| Prompt, 6,321 tokens of code, through the server | 5.5 s, **1,151 tokens/s** |
| Prompt, 6,379 tokens, `strata-flint` | 5.5 s, 1,160 tokens/s |
| Prompt, 1,000 tokens | 0.95 s (1,056 tokens/s) |
| The same question about the same text, asked again | 6,310 of 6,318 tokens reused, 8 read in 0.06 s |
| Decode, 8 different questions (stories, code, a table, a poem), 700 sampled tokens each (temperature 1, top_p 0.95, top_k 20) | **63.7 tokens/s** on average (59 to 66 per answer), no read from the file |
| Decode, through the server, answering about the 6,321-token file | 54.6 and 61.3 tokens/s |
| Decode, greedy, a text that repeats | 60 tokens/s after 600 steps (the best case, not the typical one) |
| Start (model loaded, caches filled) | about 20 s |

The decode went through these steps on this machine (the sampled questions above, except the first lines): the experts swapped
into VRAM before each use 27 tokens/s, a lean step that reads the host tier in place 34, with a bigger host tier (9.9 GiB) 49, with
the embedding table in RAM 50, with a tier large enough for every expert (10.2 GiB) and short prompts read in place 63.7. The
step is now 15.7 ms: 13.4 ms of GPU (10.7 ms is the floor measured with every expert in VRAM, `--bench-floor`; the rest is the GPU
reading about 40 experts per step from RAM over the bus) and the host's share.

**What limits it.** Memory. With a tier of 8.8 GiB (what the machine's free RAM gave before it was sized better) 2% of the choices
went to the file and the decode was 34 tokens/s; with 9.9 GiB, 49. A machine with less RAM or a smaller card than this one will
be between those numbers and 64. The VRAM is part of it too: a context of 32,768 cells instead of 16,384 takes 0.43 GiB of the
cache (about 2.5 tokens/s). A lookahead that reads the next layers' guessed experts from the file ahead of time was tried and
removed: the experts that reach the file are the rarely used ones, and the guess finds them 1 time in 7 (lookahead of 3 layers:
23 tokens/s). A second stream for the shared expert (42 against 49 tokens/s), reading with the page cache (45 against 49) and splitting the file
reads (the same) lost or gave nothing; a replacement policy by frequency (simulated on a recorded trace: 5% fewer reads) is not worth
its code; fusing the attention's residual add, the norm and the quantizer into one kernel gave 0.4 tokens/s.

**What limits the prompt.** Every chunk streams all the experts through the card; the compute alone is 1,729 tokens/s at 2048.
With every expert in memory, a prompt of up to a thousand tokens takes about a second.

## Checked against llama.cpp

The reference is llama.cpp's own Flint graph (CPU backend), same GGUF, same ids. `tools/flint/compare.py` compares the logits;
`strata-flint --nll` gives the loss of every prompt token for a perplexity comparison.

| Check | Result |
| --- | --- |
| Perplexity, 1100 tokens of code: Strata against llama.cpp with its attention in F32 and an F32 KV cache | mean loss 2.4909 against 2.4904 |
| The same against llama.cpp's default (F16 KV cache) | 2.4909 against 2.4593 |
| llama.cpp against itself, between those two settings | the same size of difference (mean -0.031, per-token spread 0.66; Strata against llama.cpp: spread 0.67) |
| Teacher-forced, 40 steps, 13-token prompt | most likely token identical on 39 of 40 steps |
| Teacher-forced, 8 steps, 1100-token prompt (the sliding-window ring wraps) | most likely token identical on 8 of 8 steps |
| Prompt chunks of 64 against 512 tokens; experts resident against cached | identical logits |

Flint is very sensitive to rounding: some positions get activations of about 10^4 in layer 46, and a small change can change
which experts are chosen. Per-token comparisons (KL, single logits) are noisy even between two llama.cpp settings, so use the
means. The noise floor of llama.cpp on a GPU, with the same q8_1 activations as Strata, was **not** measured.

## Not done

- The decode speed needs the RAM described above. Not tried on another machine; a card with more memory or a machine with more
  RAM should be at least as fast, one with less is slower.
- No ready-made engine package includes Flint, so `strata launch` and the app cannot install it yet.
- One request at a time (no `--parallel`), no pictures, no speculative decoding. Greedy and sampled decoding only.
- A conversation that diverges more than 512 tokens before its end is read again from the start (see above): about 5 s for
  6,000 tokens.
- Not built or run on CUDA, Windows or Mac.
- The page-locked tier leaves little RAM free (2.6 GB on 15 GB): `--host-cache` takes less if the machine is also used for
  other things.
- Only the Q4_K_XL file was run. The other Flint files were not tested and are not offered by setup.
