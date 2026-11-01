# Flint (experimental)

Flint is Recursive's coding model (Hugging Face `RecursiveSI/flint`, GGUF architecture `flint`). It has 39.8 billion
parameters and uses 2.5 billion of them for each token (49 layers, 2 dense and 47 with 256 experts, 6 used per token).
Strata runs it with its own engine, `strata-flint`. This page says what works, what was measured and what does not exist yet.

> **Status.** A first engine, tested on one machine. It runs from the command line on token ids. `strata launch`, the
> server and the apps do **not** use it yet, and there is no chat template. Linux only. Built and run on AMD (HIP) only.

## What you need

- Linux, an AMD card Strata supports ([AMD_HIP.md](AMD_HIP.md)) or, untested, an NVIDIA card, and a source build of Strata
  (`cmake -DSTRATA_ENABLE_HIP=ON ...`; on Ubuntu `tools/install_amd_build_deps_ubuntu.sh` installs the build packages).
- A Flint GGUF. The one that was tested is Unsloth's `flint-UD-Q4_K_XL.gguf` (25.05 GB, Hugging Face `unsloth/flint-GGUF`).
  Q4 or above is recommended; the smaller files were not tested.
- RAM and disk: the experts that do not fit in the card are kept in RAM, then read from the file. An NVMe disk is needed.

## Run it

```sh
strata-flint --model flint-UD-Q4_K_XL.gguf --tokens 630,6963,297,9865,332 --n 16
strata-flint --help
```

The ids come from the model's tokenizer (for example `llama-tokenize --ids`). The output is the generated ids.

## How the experts are stored

Flint does not fit in 16 GB at Q4 (the experts are 23.4 GB). `strata-flint` keeps them on three levels:

1. **VRAM.** A cache with the same share of the experts in every MoE layer (110 to 173 of 256 on a 16 GB card), least
   recently used first.
2. **Host RAM.** What the VRAM cache gives up is copied down here by asynchronous DMA, and comes back the same way. The area is
   page-locked (`--no-pin-host` turns that off, which is slower) and sized from `MemAvailable` less 3.5 GiB (`--host-cache GiB`).
3. **The file.** Experts in neither level are read with parallel `O_DIRECT` reads. The disk reached about 1.8 GB/s for these
   reads whatever their size.

If everything fits in VRAM, the experts are simply resident and the other levels are not used.

## Measured

Machine: RX 9070 (gfx1201, 16 GB), Ryzen 5 5600, 15 GB RAM, NVMe (3 GB/s sequential), Ubuntu 26.10, ROCm packages of the
distribution, engine built from this tree. File: `flint-UD-Q4_K_XL.gguf`. Greedy decoding of a code prompt.

| | Result |
| --- | --- |
| Decode, steady (after about 600 tokens) | 36 ms per token, about 27 tokens/s |
| Decode, averaged over the first 1500 tokens | 23 tokens/s |
| Decode, first tokens right after a long prompt | about 11 tokens/s (the cache holds the prompt's experts) |
| Prompt, 1100 tokens of code | 59 tokens/s |
| Prompt, 6379 tokens of code | 94 tokens/s (68 s) |

Where the decode time goes: about 22 ms of GPU work, which is limited by the number of kernel launches (about 1500 per token)
and one host synchronisation per MoE layer, then the rest for the expert copies. Other levels, measured on the same machine:
VRAM cache only 85 ms per token; with a host tier that is not page-locked 55 ms; with the page-locked tier 36 ms.
A lookahead that guesses the next layer's experts (`--prefetch N`) was tried and gave no gain on top of that. Flint's routing is
flat: a cache of 48, 96 and 160 experts per layer hits 50%, 75% and 93% of the choices.

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

- No server, no `strata launch --family flint`, no chat template or reasoning handling, no tokenizer in the engine.
- Not built or run on CUDA, Windows or Mac. Greedy decoding only.
- No CUDA or HIP graphs, so the decode is limited by launch overhead (see above).
- More RAM would help: the second level is limited by what the system can spare. Not measured.
- Only the Q4_K_XL file was run. The other Flint files and `--family` presets were not tested.
