# Flint (experimental)

Flint is Recursive's coding model (Hugging Face `RecursiveSI/flint`, GGUF architecture `flint`). It has 39.8 billion
parameters and uses 2.5 billion of them for each token (49 layers, 2 dense and 47 with 256 experts, 6 used per token).
Strata runs it with its own engine, `strata-flint`. This page says what works, what was measured and what does not exist yet.

> **Status.** A first engine, tested on one machine. It runs from the command line on token ids and behind Strata's server
> (OpenAI and Anthropic APIs, reasoning, tool calls, streaming). One request at a time, no pictures. Linux with an AMD card
> only. There is no ready-made engine package that includes it yet, so it is installed from a source checkout. Speed: a long
> prompt is read at over 1,100 tokens/s; the answer comes at about 25-30 tokens/s (see "Measured": what limits it).

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

Flint does not fit in 16 GB at Q4 (the experts are 23.4 GB). `strata-flint` keeps them on three levels:

1. **VRAM.** A cache with the same share of the experts in every MoE layer (about 95 to 150 of 256 on a 16 GB card), least
   recently used first.
2. **Host RAM.** What the VRAM cache gives up is copied down here by asynchronous DMA, and comes back the same way. The area is
   page-locked (`--no-pin-host` turns that off, which is slower) and sized from `MemAvailable` less 3 GiB, at most two thirds of the RAM (`--host-cache GiB`).
3. **The file.** Experts in neither level are read with parallel `O_DIRECT` reads. The disk reached about 1.8 GB/s for these
   reads whatever their size.

At start the first experts of every layer fill the first two levels (about 19 GiB, 12 s, once): a prompt then finds them there,
and the decode starts with something to hit. If everything fits in VRAM, the experts are simply resident.

**A long prompt** (more than 64 tokens) takes another path, built from the pieces Strata uses for its other models on this card:
the dense projections are FP16 GEMMs on the matrix cores, the attention is batched GEMMs per KV head with the sink in the softmax,
and the routed experts go through llama.cpp's int8 MMQ kernels in groups of 16 experts. A thread has each group's experts waiting
in a ring of buffers (copied from the VRAM cache, from the host tier, or read from the file by a second thread a few layers
ahead), so the GPU does not wait for the host. The prompt is cut into near-equal chunks of at most 4096 tokens (`--chunk`):
every chunk is a pass over all the experts. Needs a build with `STRATA_PREFILL_MMQ` (setup turns it on for AMD).

## Measured

Machine: RX 9070 (gfx1201, 16 GB), Ryzen 5 5600, 15 GB RAM, NVMe (3 GB/s sequential, about 1.8 GB/s for 1-2 MB random reads),
Ubuntu 26.10, engine built from this tree. File: `flint-UD-Q4_K_XL.gguf`.

| | Result |
| --- | --- |
| Prompt, 6,281 tokens of code, through the server (default chunk) | 5.4 s, **1,153 tokens/s** |
| Prompt, 6,379 tokens, `strata-flint`, two chunks of 3,190 | 5.5 s, 1,164 tokens/s |
| Prompt, 1,000 tokens | 2.1 s (487 tokens/s) |
| Prompt, 300 tokens | 2.0 s (151 tokens/s) |
| Prompt, 100 tokens | 1.9 s (52 tokens/s) |
| The same question asked again about the same text | 6,270 of 6,278 tokens reused, 8 read in 0.23 s |
| Decode, through the server | 24 to 31 tokens/s |
| Decode, `strata-flint`, 700 tokens | 25.6 tokens/s (chunk 4096), 27.6 (chunk 2048) |
| Start (caches filled) | about 19 s |

**What limits the prompt.** Every chunk streams all the experts through the card. With the caches full, 10% of them (about
2.2 GB) still come from the file: 1.6 s per chunk, hidden behind the compute of a big chunk but not of a small one. So any
prompt of up to about a thousand tokens takes about 2 s, and the speed climbs with the length up to the compute: 2,048 tokens in
1.18 s, 1,729 tokens/s, with no file reads.

**What limits the decode.** A step costs about 35 ms. The GPU part, measured with every expert a hit and no host work, is
**10.7 ms** (93 tokens/s; it was 21 ms until a router kernel that did its top-6 selection in one thread was replaced by a parallel
one); a HIP graph does not change it (10.7 ms): the kernels, about 10 µs each, are the cost, not their launches. The rest is the
experts that are not in the VRAM cache: about 45 per step come back from the host tier, about 6 to 10 are read from the file
(14 ms per step at the start, about 9 ms once the cache has adapted: 1.8 MB per expert at 1.8 GB/s). Flint's routing is flat
(a cache of 48, 96 and 160 experts per layer hits 50%, 75% and 93% of the choices), so the 4 GB that do not fit in VRAM and RAM are
still asked for by about 2% of the choices. **Reading experts straight from the host memory** (an experiment, `STRATA_FLINT_DIRECT=1`)
measured slower (40 ms), and a lookahead that guesses the next layer's experts (`--prefetch`) gave no gain: they remove copies,
not the file reads. With all the experts in VRAM and RAM there would be no file read and the step would be close to the 10.7 ms
plus the host-tier traffic; that needs about 4 GB more RAM, or a smaller file than Q4_K_XL. Not measured.

Other levels tried on the same machine, for the history: VRAM cache only 85 ms per token; with a host tier that is not
page-locked 55 ms; with the page-locked tier 36 ms; with the router fixed 35 ms (the step was 46 ms before).

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

- **Decode is at 25-30 tokens/s, not 60.** The GPU part would allow about 90, the file reads and expert copies take the rest (see
  above). Closing that needs more RAM or a smaller file, not a faster kernel.
- No ready-made engine package includes Flint, so `strata launch` and the app cannot install it yet.
- One request at a time (no `--parallel`), no pictures, no speculative decoding. Greedy and sampled decoding only.
- A conversation that diverges more than 512 tokens before its end is read again from the start (see above): about 5 s for
  6,000 tokens.
- A short prompt costs about 2 s whatever its length (up to ~1,000 tokens), because every chunk is a pass over all the experts.
- Not built or run on CUDA, Windows or Mac.
- More RAM would help: the second level is limited by what the system can spare. Not measured.
- Only the Q4_K_XL file was run. The other Flint files were not tested and are not offered by setup.
