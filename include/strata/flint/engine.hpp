// include/strata/flint/engine.hpp - the Flint (GGUF architecture `flint`) engine path as a library.
// See src/flint/flint_engine.cu.
#pragma once

namespace strata::flint {

/// The engine's command line (strata-flint --help lists it all): --model PATH, --tokens 1,2,3 [--n N] [--force IDS]
/// [--dump-full-logits F] [--nll F], --max-context N, --chunk N, --expert-vram GiB (VRAM for the expert cache; default:
/// what is free after a warmup, less 0.6 GiB), --host-cache GiB (host-RAM tier; default: MemAvailable less 3.5 GiB),
/// --no-pin-host, --prefetch N, --prof.  Returns the process exit code.
int run(int argc, char** argv);

}  // namespace strata::flint
