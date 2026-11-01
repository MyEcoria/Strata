// src/flint/flint_engine.cu - strata-flint: the engine path for Flint (GGUF architecture `flint`).
//
// Flint is a 49-layer decoder: 2 dense SwiGLU layers then 47 MoE layers (256 routed experts of 512, 6 per token, a shared
// expert of 1024, sigmoid router with the chosen weights renormalised).  35 layers attend over a 513-token window and 14
// over the whole context; every layer has per-head RMS norms on Q and K, a learned sink logit per head, and (per
// `flint.attention.rope_pattern`) NEOX rotary embedding of the whole 128-wide head or none.  Pre-norm residual blocks.
// The reference is llama.cpp's src/models/flint.cpp.
//
// What it does:
//   * weights straight from the GGUF into the native MMVQ (Q8_0 / K-quants / i-quants), the 1-D tensors and the router as f32;
//   * the routed experts as Strata's native expert blobs [gate | up | down], run by native_expert_grouped, on three tiers:
//     a per-expert LRU cache in VRAM (the same share of the experts in every MoE layer), a page-locked host-RAM tier
//     (exclusive with the VRAM one: an expert the VRAM cache gives up is copied down by asynchronous DMA, a host hit comes
//     back the same way), and the file itself (parallel O_DIRECT preads into a pinned buffer) for the rest.  If every
//     expert fits in VRAM they are all resident and the tiers are not used;
//   * the prompt in chunks of up to 512 tokens through the same layer code as decode; a chunk can choose more experts of one
//     layer than the cache has slots, so the experts are run in rounds through the cache (groups of at most 16 entries);
//   * attention: split-K over 128-key chunks with a log-sum-exp merge that also folds in the sink logit, in blocks of 16
//     queries; the sliding-window layers keep a ring of 1024 cells, the full layers one cell per position; the cache is FP16
//     (as llama.cpp's default);
//   * no CUDA / HIP graphs yet: ~1500 kernel launches and one host sync per MoE layer per decoded token.
//
// Modes: --tokens IDS [--n N] [--force IDS] [--dump-full-logits F] [--dump-layers F] [--nll F] [--bench-tg N] [--prof].
// Checked against llama.cpp (tools/flint/compare.py, docs/FLINT.md).
#include "strata/artifact/gguf_reader.hpp"
#include "strata/flint/engine.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/dp4a.hpp"
#include "strata/kernels/q8_1_finite.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "strata/qwen35/sampling.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <deque>
#include <random>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <condition_variable>
#include <map>
#include <csignal>
#include <iostream>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace K = strata::kernels;
namespace MMQ = strata::prefill::mmq;

#define CK(x)                                                                                                         \
    do {                                                                                                              \
        cudaError_t e_ = (x);                                                                                         \
        if (e_ != cudaSuccess) {                                                                                      \
            std::fprintf(stderr, "strata-flint: %s failed: %s (%s:%d)\n", #x, cudaGetErrorString(e_), __FILE__, __LINE__); \
            std::exit(1);                                                                                             \
        }                                                                                                             \
    } while (0)

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

constexpr int kThreads = 256;
constexpr int kHD = 128;        // head width (checked at load)
constexpr int kAttChunk = 128;  // keys per attention chunk
constexpr int kRing = 1024;     // cells of a sliding-window layer's cache (window 513 + a chunk of 8, with room)
constexpr int kCols = 8;        // columns one MMVQ launch takes
constexpr size_t kGroupEntries = 16;   // activations one expert group takes (the MMVQ's MAX_NCOLS in a HIP build)
constexpr int kQB = 16;         // queries per attention launch (bounds the partials)
constexpr int kMaxT = 8;        // the warmup / decode chunk
constexpr int kMaxK = 8;        // experts per token (6 here)

// ------------------------------------------------------------------------------------------------ small kernels
__device__ __forceinline__ float silu(float v) { return v / (1.0f + expf(-v)); }

__device__ float block_sum(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}
__device__ float block_max(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}

#include "flint_kv.cuh"   // the optional quantised KV cache (--kv-quant tqt | hq)

/// y[r] = rms(x[r]) * w, one block per row of n.
__global__ void k_rms(const float* x, const float* w, float* y, int n, float eps) {
    __shared__ float sh[kThreads];
    const float* xr = x + (size_t) blockIdx.x * n;
    float* yr = y + (size_t) blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i] * xr[i];
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * r * w[i];
}
__global__ void k_add(float* x, const float* y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

/// Embedding rows of a Q8_0 / BF16 / F16 / F32 table: out[t][i] for token tok[t].
__global__ void k_embed(const int32_t* tok, const uint8_t* table, size_t row_bytes, int type, float* out, int n) {
    const uint8_t* row = table + (size_t) tok[blockIdx.x] * row_bytes;
    float* o = out + (size_t) blockIdx.x * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v;
        if (type == 8) {
            const uint8_t* b = row + (size_t) (i / 32) * 34;
            v = __half2float(*(const __half*) b) * (float) (int8_t) b[2 + i % 32];
        } else if (type == 30) {
            const uint32_t u = (uint32_t) (row[2 * i] | (row[2 * i + 1] << 8)) << 16;
            v = __uint_as_float(u);
        } else if (type == 1) {
            v = __half2float(*(const __half*) (row + 2 * i));
        } else {
            v = *(const float*) (row + 4 * i);
        }
        o[i] = v;
    }
}

/// y[t][o] = W[o] . x[t] for f32 W [n_out][n_in] (n_in a multiple of 128): one warp per row, 8 rows per block,
/// grid (ceil(n_out / 8), T).
__global__ void k_gemv_f32(const float* W, const float* x, float* y, int n_in, int n_out) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int o = blockIdx.x * 8 + warp;
    if (o >= n_out) return;
    const float4* wr = (const float4*) (W + (size_t) o * n_in);
    const float4* xr = (const float4*) (x + (size_t) blockIdx.y * n_in);
    float acc = 0.f;
    for (int i = lane; i < n_in / 4; i += 32) {
        const float4 a = wr[i], b = xr[i];
        acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
    if (lane == 0) y[(size_t) blockIdx.y * n_out + o] = acc;
}

// ------------------------------------------------------------------------------------------------ fused decode kernel
/// k_add (x += y), k_rms (h = rms(x) * w) and the q8_1 image of h (what native_quantize_q8_1 writes) in one launch, one block of 256
/// per row.  The arithmetic is that of the three kernels it replaces, in the same order (the Qwen3.6 engine's k_add_rms_q, written
/// to give their bits; the logits of a forced run are identical with and without it: STRATA_FLINT_NO_FUSE=1).
struct Q8Blk {
    __half2 ds;
    int8_t qs[32];
};
static_assert(sizeof(Q8Blk) == 36, "q8_1 block");
__device__ __forceinline__ float fw_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o, 32);
    return v;
}
__device__ __forceinline__ float fw_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o, 32));
    return v;
}
__device__ __forceinline__ void q8_put(Q8Blk* __restrict__ y, int i, float v, int lane) {
    const float amax = fw_max(fabsf(v));
    const float sum = fw_sum(v);
    const float d = strata::kernels::q8_1_finite(amax / 127.0f);
    int8_t q = 0;
    if (amax != 0.0f) {
        const float qf = roundf(v / d);
        q = (int8_t) (qf > 127.0f ? 127.0f : qf < -127.0f ? -127.0f : qf);
    }
    y[i >> 5].qs[lane] = q;
    if (lane == 0) y[i >> 5].ds = strata::kernels::q8_1_ds(d, sum);
}
__global__ void __launch_bounds__(256) k_add_rms_q8(float* x, const float* y, const float* w, float* h, Q8Blk* hq, int n, float eps) {
    __shared__ float sh[256];
    float* xr = x + (size_t) blockIdx.x * n;
    const float* yr = y + (size_t) blockIdx.x * n;
    float* hr = h + (size_t) blockIdx.x * n;
    Q8Blk* qr = hq + (size_t) blockIdx.x * (n >> 5);
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += 256) {
        const float v = xr[i] + yr[i];
        xr[i] = v;
        s += v * v;
    }
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) {
        if ((int) threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o];
        __syncthreads();
    }
    const float r = rsqrtf(sh[0] / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += 256) {
        const float hv = __fmul_rn(__fmul_rn(xr[i], r), w[i]);   // no FMA with the quantizer's sums
        hr[i] = hv;
        q8_put(qr, i, ((volatile float*) hr)[i], threadIdx.x & 31);
    }
}
unsigned rows8(int n) { return (unsigned) ((n + 7) / 8); }
unsigned grid1(int64_t n, int b = kThreads) { return (unsigned) ((n + b - 1) / b); }

/// Q / K / V of T tokens at positions p0 .. p0 + T - 1: the per-head RMS norm of Q and K, NEOX rotary embedding when
/// `rope`, Q out as f32, K and V into the FP16 cache [HKV][ncell][128] (cell = position % ring when ring > 0, else the
/// position).  Grid (H + HKV, T), 128 threads.
__global__ void k_prep(const float* qx, const float* kx, const float* vx, const float* qw, const float* kw, float* qout,
                       __half* kc, __half* vc, int H, int HKV, int p0, int rope, float base, float eps, int ring, int ncell,
                       __half* kst = nullptr, __half* vst = nullptr) {
    __shared__ float sh[kHD];
    __shared__ float row[kHD];
    const int t = blockIdx.y, hb = blockIdx.x, tid = threadIdx.x;
    const int pos = p0 + t;
    const bool isq = hb < H;
    const float* src = isq ? qx + ((size_t) t * H + hb) * kHD : kx + ((size_t) t * HKV + (hb - H)) * kHD;
    float v = src[tid];
    const float ss = block_sum(v * v, sh);
    v = v * rsqrtf(ss / (float) kHD + eps) * (isq ? qw[tid] : kw[tid]);
    if (rope) {
        row[tid] = v;
        __syncthreads();
        const int i = tid & (kHD / 2 - 1);
        const float x0 = row[i], x1 = row[i + kHD / 2];
        const float theta = (float) pos * powf(base, -(float) (2 * i) / (float) kHD);
        const float c = cosf(theta), s = sinf(theta);
        v = tid < kHD / 2 ? x0 * c - x1 * s : x0 * s + x1 * c;
    }
    if (isq) {
        qout[((size_t) t * H + hb) * kHD + tid] = v;
    } else {
        const int kvh = hb - H;
        if (kst) {   // a quantised layer: K and V wait in the staging for the encoder
            const size_t o = ((size_t) t * HKV + kvh) * kHD + tid;
            kst[o] = __float2half(v);
            vst[o] = __float2half(vx[o]);
            return;
        }
        const int cell = ring > 0 ? pos % ring : pos;
        const size_t o = ((size_t) kvh * ncell + cell) * kHD + tid;
        kc[o] = __float2half(v);
        vc[o] = __float2half(vx[((size_t) t * HKV + kvh) * kHD + tid]);
    }
}

/// Attention, part 1: one block per (chunk of 128 keys, token x head).  The keys a token sees are positions
/// max(0, pos - window + 1) .. pos (window 0: all of them).  Writes the chunk's max score, sum of exp and the
/// exp-weighted V sum.  Grid (nchunk, T * H), 128 threads.
__global__ void k_att_part(const float* q, const __half* kc, const __half* vc, int H, int HKV, int p0, int window, int ring,
                           int ncell, float scale, float* pacc, float* pm, float* pl, int nchunk) {
    __shared__ float qs[kHD], ps[kAttChunk], sh[kAttChunk];
    const int c = blockIdx.x, th = blockIdx.y, tid = threadIdx.x;
    const int t = th / H, h = th % H, kvh = h / (H / HKV);
    const int pos = p0 + t;
    const int lo = window > 0 ? max(0, pos - window + 1) : 0;
    const int n = pos - lo + 1;
    const size_t idx = (size_t) th * nchunk + c;
    const int start = c * kAttChunk;
    if (start >= n) {
        if (tid == 0) {
            pm[idx] = -INFINITY;
            pl[idx] = 0.f;
        }
        return;
    }
    const int cnt = min(kAttChunk, n - start);
    qs[tid] = q[(size_t) th * kHD + tid];
    __syncthreads();
    float s = -INFINITY;
    if (tid < cnt) {
        const int kp = lo + start + tid;
        const int cell = ring > 0 ? kp % ring : kp;
        const __half* kr = kc + ((size_t) kvh * ncell + cell) * kHD;
        float a = 0.f;
        for (int d = 0; d < kHD; ++d) a += qs[d] * __half2float(kr[d]);
        s = a * scale;
    }
    const float m = block_max(s, sh);
    const float p = tid < cnt ? expf(s - m) : 0.f;
    ps[tid] = p;
    const float l = block_sum(p, sh);
    float acc = 0.f;
    for (int j = 0; j < cnt; ++j) {
        const int kp = lo + start + j;
        const int cell = ring > 0 ? kp % ring : kp;
        acc += ps[j] * __half2float(vc[((size_t) kvh * ncell + cell) * kHD + tid]);
    }
    pacc[idx * kHD + tid] = acc;
    if (tid == 0) {
        pm[idx] = m;
        pl[idx] = l;
    }
}

/// Attention, part 2: merge the chunks and the head's sink logit (an extra softmax entry with no value).
/// Grid T * H, 128 threads.  out [T][H][128].
__global__ void k_att_merge(const float* pacc, const float* pm, const float* pl, const float* sinks, int H, int nchunk,
                            float* out) {
    const int th = blockIdx.x, tid = threadIdx.x, h = th % H;
    const float sink = sinks[h];
    float M = sink;
    for (int c = 0; c < nchunk; ++c) M = fmaxf(M, pm[(size_t) th * nchunk + c]);
    float denom = expf(sink - M), num = 0.f;
    for (int c = 0; c < nchunk; ++c) {
        const size_t idx = (size_t) th * nchunk + c;
        if (pm[idx] == -INFINITY) continue;
        const float w = expf(pm[idx] - M);
        denom += pl[idx] * w;
        num += pacc[idx * kHD + tid] * w;
    }
    out[(size_t) th * kHD + tid] = num / denom;
}

/// The router: p = sigmoid(logit) for each of the n_expert experts (at most 512), the top-k by p (ties: the lower index), their
/// weights p / max(sum, 6.1035156e-5) * scale.  One block of 256 threads per token: k rounds of an argmax over the block.
__global__ void k_route(const float* logits, int n_expert, int k, float scale, int32_t* ids, float* wts) {
    __shared__ float p[512], rv[256];
    __shared__ int ri[256];
    __shared__ int sel[kMaxK];
    __shared__ float selw[kMaxK];
    const int tid = threadIdx.x, bd = blockDim.x;
    const float* lr = logits + (size_t) blockIdx.x * n_expert;
    for (int i = tid; i < n_expert; i += bd) p[i] = 1.0f / (1.0f + expf(-lr[i]));
    __syncthreads();
    for (int j = 0; j < k; ++j) {
        float bv = -INFINITY;
        int bi = -1;
        for (int i = tid; i < n_expert; i += bd)
            if (p[i] > bv) {   // (strictly greater: of equal values the lowest index stays)
                bv = p[i];
                bi = i;
            }
        rv[tid] = bv;
        ri[tid] = bi;
        __syncthreads();
        for (int s = bd / 2; s > 0; s >>= 1) {
            if (tid < s) {
                const float ov = rv[tid + s];
                const int oi = ri[tid + s];
                if (ov > rv[tid] || (ov == rv[tid] && oi >= 0 && (ri[tid] < 0 || oi < ri[tid]))) {
                    rv[tid] = ov;
                    ri[tid] = oi;
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            sel[j] = ri[0];
            selw[j] = rv[0];
            if (ri[0] >= 0) p[ri[0]] = -INFINITY;
        }
        __syncthreads();
    }
    if (tid == 0) {
        float sum = 0.f;
        for (int j = 0; j < k; ++j) sum += selw[j];
        const float denom = fmaxf(sum, 6.103515625e-5f);
        for (int j = 0; j < k; ++j) {
            ids[(size_t) blockIdx.x * k + j] = sel[j];
            wts[(size_t) blockIdx.x * k + j] = selw[j] / denom * scale;
        }
    }
}

/// Groups the T * k (token, expert) entries by expert for native_expert_grouped, resident layer: expert x's blob at
/// blobs + x * blob.  One block (the entries are few).
__global__ void k_group(const int32_t* ids, int n_ent, int K_, int n_expert, const uint8_t* blobs, unsigned long long blob,
                        unsigned long long* grp_ptr, int32_t* grp_start, int32_t* n_groups, int32_t* ent_dst,
                        int32_t* ent_tok) {
    __shared__ int cnt[512], off[512];
    for (int x = threadIdx.x; x < n_expert; x += blockDim.x) cnt[x] = 0;
    __syncthreads();
    for (int e = threadIdx.x; e < n_ent; e += blockDim.x) atomicAdd(&cnt[ids[e]], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int g = 0, pos = 0;
        for (int x = 0; x < n_expert; ++x) {
            off[x] = pos;
            if (cnt[x] == 0) continue;
            grp_ptr[g] = (unsigned long long) (blobs + (size_t) x * blob);
            grp_start[g] = pos;
            pos += cnt[x];
            ++g;
        }
        grp_start[g] = pos;
        *n_groups = g;
        for (int e = 0; e < n_ent; ++e) {
            const int p = off[ids[e]]++;
            ent_dst[p] = e;
            ent_tok[p] = e / K_;
        }
    }
}

/// x[t] += shared[t] + sum_k w[t][k] * parts[t * k + j]   (the MoE block's output added to the residual).
__global__ void k_combine(float* x, const float* shared, const float* parts, const float* wts, int K_, int n) {
    const int t = blockIdx.x;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = shared[(size_t) t * n + i];
        for (int j = 0; j < K_; ++j) v += wts[t * K_ + j] * parts[((size_t) t * K_ + j) * n + i];
        x[(size_t) t * n + i] += v;
    }
}


// ------------------------------------------------------------------------------------------------ prompt kernels
/// The input of a down projection, row by row: a row whose largest value is over `target` is scaled down to it (its q8_1 block
/// sums, kept in half precision, and the FP16 GEMM operands would overflow: layer 46 of Flint has rows of 1e5 and more), and
/// rsc[r] gets the factor that puts the product back (1 for a row left as it is).  One block per row.
__global__ void k_row_fit(float* h, int n, float* rsc, float target) {
    __shared__ float sh[kThreads];
    float* row = h + (size_t) blockIdx.x * n;
    float mx = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) mx = fmaxf(mx, fabsf(row[i]));
    mx = block_max(mx, sh);
    const float sc = mx > target ? target / mx : 1.0f;
    if (sc != 1.0f)
        for (int i = threadIdx.x; i < n; i += blockDim.x) row[i] *= sc;
    if (threadIdx.x == 0) rsc[blockIdx.x] = 1.0f / sc;
}
/// y[r] *= rsc[r] (n values a row): the products of rows k_row_fit scaled down, put back.
__global__ void k_row_mul(float* y, const float* rsc, int n) {
    const float f = rsc[blockIdx.x];
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[(size_t) blockIdx.x * n + i] *= f;
}
/// Counts the values of x[n] that are not finite.
__global__ void k_nonfinite(const float* x, int64_t n, int* cnt) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && !isfinite(x[i])) atomicAdd(cnt, 1);
}
/// x f32 -> FP16 (the prompt GEMMs' activations).
__global__ void k_to_f16(const float* x, __half* y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i]);
}
/// h = silu(g) * u (h may be g).
__global__ void k_swiglu(const float* g, const float* u, float* h, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] = silu(g[i]) * u[i];
}
/// The prompt's routed entries sorted by expert for MMQ: bounds[x] .. bounds[x + 1] are expert x's rows; sorted row r holds
/// entry dst[r] (= t * K + k) of token src[r]; inv[e] = entry e's sorted row (where its products land).
__global__ void k_route_count(const int32_t* ids, int n_ent, int32_t* cnt) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < n_ent) atomicAdd(&cnt[ids[e]], 1);
}
__global__ void k_route_scan(const int32_t* cnt, int n_expert, int32_t* bounds, int32_t* cur) {
    if (threadIdx.x != 0) return;
    int pos = 0;
    for (int x = 0; x < n_expert; ++x) {
        bounds[x] = pos;
        cur[x] = pos;
        pos += cnt[x];
    }
    bounds[n_expert] = pos;
}
__global__ void k_route_scatter(const int32_t* ids, int n_ent, int K_, int32_t* cur, int32_t* dst, int32_t* src, int32_t* inv) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_ent) return;
    const int r = atomicAdd(&cur[ids[e]], 1);
    dst[r] = e;
    src[r] = e / K_;
    inv[e] = r;
}
/// Benchmark only (--bench-floor): k_group with expert x's blob in slot x % S of the layer's VRAM cache, as if every expert were a
/// hit: the time of a token with no host work at all.
__global__ void k_group_mod(const int32_t* ids, int n_ent, int K_, int n_expert, const uint8_t* blobs, unsigned long long blob, int S,
                            unsigned long long* grp_ptr, int32_t* grp_start, int32_t* n_groups, int32_t* ent_dst, int32_t* ent_tok) {
    __shared__ int cnt[512], off[512];
    for (int x = threadIdx.x; x < n_expert; x += blockDim.x) cnt[x] = 0;
    __syncthreads();
    for (int e = threadIdx.x; e < n_ent; e += blockDim.x) atomicAdd(&cnt[ids[e]], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int g = 0, pos = 0;
        for (int x = 0; x < n_expert; ++x) {
            off[x] = pos;
            if (cnt[x] == 0) continue;
            grp_ptr[g] = (unsigned long long) (blobs + (size_t) (x % S) * blob);
            grp_start[g] = pos;
            pos += cnt[x];
            ++g;
        }
        grp_start[g] = pos;
        *n_groups = g;
        for (int e = 0; e < n_ent; ++e) {
            const int p = off[ids[e]]++;
            ent_dst[p] = e;
            ent_tok[p] = e / K_;
        }
    }
}
/// x[t] += shared[t] + sum_j wts[t * K + j] * parts[inv[t * K + j]] (the prompt's MoE block: the experts' rows are in sorted order).
__global__ void k_combine_sorted(float* x, const float* shared, const float* parts, const int32_t* inv, const float* wts,
                                 const float* rsc, int K_, int n) {
    const int t = blockIdx.x;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = shared[(size_t) t * n + i];
        for (int j = 0; j < K_; ++j) {
            const int r = inv[t * K_ + j];
            v += wts[t * K_ + j] * parts[(size_t) r * n + i] * rsc[r];
        }
        x[(size_t) t * n + i] += v;
    }
}
/// The window's keys and values of a sliding-window layer out of its ring: positions klo .. klo + nk - 1 into compact
/// [kvh][nk][128] halves.  grid1(HKV * nk * 128).
__global__ void k_ring_gather(const __half* kc, const __half* vc, __half* ko, __half* vo, int klo, int nk, int ring, int ncell, int HKV) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) HKV * nk * kHD) return;
    const int64_t per = (int64_t) nk * kHD;
    const int kvh = (int) (i / per);
    const int64_t rest = i % per;
    const int j = (int) (rest / kHD), d = (int) (rest % kHD);
    const int cell = (klo + j) % ring;
    const size_t src = ((size_t) kvh * ncell + cell) * kHD + d;
    ko[i] = kc[src];
    vo[i] = vc[src];
}
/// The prompt attention's softmax with the sink: row (h, t) of S (ld floats, nk cells starting at position klo) ->
/// exp(s * scale - m) / (sum + exp(sink - m)) over the cells the query at position qpos0 + t sees (the window, causal), 0
/// elsewhere, as FP16 (same row offsets).  grid (ns, H), block 256.
__global__ void k_att_softmax_sink(const float* S, __half* S16, int ld, int ns, int nk, int klo, int qpos0, int window,
                                   const float* sinks, float scale) {
    __shared__ float sh[kThreads];
    const int t = blockIdx.x, h = blockIdx.y;
    const float* row = S + ((size_t) h * ns + t) * ld;
    __half* out = S16 + ((size_t) h * ns + t) * ld;
    const int pos = qpos0 + t;
    const int hi = min(nk - 1, pos - klo);
    const int lo = window > 0 ? max(0, pos - window + 1 - klo) : 0;
    float mx = sinks[h];
    for (int p = lo + threadIdx.x; p <= hi; p += blockDim.x) mx = fmaxf(mx, row[p] * scale);
    mx = block_max(mx, sh);
    float sum = 0.f;
    for (int p = lo + threadIdx.x; p <= hi; p += blockDim.x) sum += expf(row[p] * scale - mx);
    sum = block_sum(sum, sh) + expf(sinks[h] - mx);
    const float inv = 1.0f / sum;
    for (int p = threadIdx.x; p < ld; p += blockDim.x)
        out[p] = __float2half(p >= lo && p <= hi ? expf(row[p] * scale - mx) * inv : 0.f);
}

// ------------------------------------------------------------------------------------------------ the model
struct Mat {
    const uint8_t* d = nullptr;   // device, GGUF blocks
    int type = -1;
    int n_in = 0, n_out = 0;
};

struct Layer {
    bool swa = false, rope = false, dense = false;
    const float *attn_norm = nullptr, *ffn_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr, *sinks = nullptr;
    Mat wq, wk, wv, wo;
    Mat ffn_gate, ffn_up, ffn_down;      // the dense layers
    const float* router = nullptr;       // [n_expert][N] f32
    Mat sh_gate, sh_up, sh_down;
    K::NativeExpertLayout L;
    size_t stride = 0;                   // bytes between expert blobs
    const uint8_t *src_g = nullptr, *src_u = nullptr, *src_d = nullptr;   // the mapped GGUF's expert tensors
    size_t src_gsz = 0, src_dsz = 0;     // bytes of one expert's gate (= up) and down rows
    uint64_t off_g = 0, off_u = 0, off_d = 0;   // file offsets of the three expert tensors (for the direct reads)
    uint8_t* blobs = nullptr;            // resident: n_expert blobs, `stride` apart
    // the VRAM expert cache, when not every expert of every layer fits: S slots of `stride` bytes, LRU
    uint8_t* cblob = nullptr;
    int S = 0;
    mutable std::vector<int> exp_slot, slot_exp;   // expert -> slot (-1: not cached), slot -> expert (-1: empty)
    mutable std::vector<long> slot_use;            // the engine clock at the slot's last use
    // the host-RAM tier below the VRAM cache: experts the VRAM cache gave up, kept as native blobs (an expert is in at
    // most one tier)
    uint8_t* hblob = nullptr;
    uint8_t* hdev = nullptr;   // the same memory as the GPU addresses it (mapped host memory): the experts are read in place
    int Sh = 0;
    mutable std::vector<int> hexp_slot, hslot_exp;
    mutable std::vector<long> hslot_use;
    mutable std::vector<uint8_t> hot;                 // uses of each expert outside the VRAM cache since it was last promoted (saturating)
    mutable std::vector<std::pair<int, int>> pfree;   // (expert, host slot) promoted to VRAM: the host copy is released at the next call of the layer
};

struct Model {
    std::shared_ptr<strata::GgufModel> file;
    std::string file_path;                     // the one file the expert reads go to
    std::vector<int32_t> eog;                  // tokenizer.ggml.eog_token_ids: the ids that end a turn
    int64_t N = 0, FF = 0, H = 0, HKV = 0, L = 0, NE = 0, TK = 0, FFE = 0, FFS = 0, VOC = 0, n_dense = 0, window = 0;
    float eps = 1e-6f, rope_base = 10000.f, w_scale = 1.f;
    std::vector<Layer> layers;
    const float* out_norm = nullptr;
    Mat head;
    const uint8_t* embd = nullptr;
    int embd_type = -1;
    size_t embd_row = 0;
};

float bf16_to_f32(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000) << 16, exponent = (h >> 10) & 0x1f, mantissa = h & 0x3ff;
    uint32_t u;
    if (exponent == 0) {
        if (mantissa == 0) {
            u = sign;
        } else {
            uint32_t m = mantissa, shift = 0;
            while ((m & 0x400) == 0) {
                m <<= 1;
                ++shift;
            }
            u = sign | ((uint32_t) (127 - 14 - shift) << 23) | ((m & 0x3ff) << 13);
        }
    } else if (exponent == 0x1f) {
        u = sign | 0x7f800000u | (mantissa << 13);
    } else {
        u = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
    }
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

size_t row_bytes(int type, int64_t n) {
    int be = 0, bb = 0;
    if (!strata::block_geometry((uint32_t) type, be, bb)) return 0;
    return (size_t) (n / be) * (size_t) bb;
}
size_t block_bytes(int type) {
    int be = 0, bb = 0;
    return strata::block_geometry((uint32_t) type, be, bb) ? (size_t) bb : 1;
}
size_t gcd_sz(size_t a, size_t b) { return b ? gcd_sz(b, a % b) : a; }
size_t lcm_sz(size_t a, size_t b) { return a / gcd_sz(a, b) * b; }

/// A tensor as f32 on the device (F32 as it is, BF16 / F16 / Q8_0 widened).
float* upload_f32(const strata::TensorInfo& t, const uint8_t* src) {
    std::vector<float> h((size_t) t.elements());
    if (t.type == 0) {
        std::memcpy(h.data(), src, h.size() * 4);
    } else if (t.type == 30) {
        for (size_t i = 0; i < h.size(); ++i) h[i] = bf16_to_f32((uint16_t) (src[2 * i] | (src[2 * i + 1] << 8)));
    } else if (t.type == 1) {
        for (size_t i = 0; i < h.size(); ++i) h[i] = f16_to_f32((uint16_t) (src[2 * i] | (src[2 * i + 1] << 8)));
    } else if (t.type == 8 && h.size() % 32 == 0) {
        for (size_t b = 0; b < h.size() / 32; ++b) {
            const uint8_t* q = src + b * 34;
            const float d = f16_to_f32((uint16_t) (q[0] | (q[1] << 8)));
            for (int j = 0; j < 32; ++j) h[b * 32 + j] = d * (float) (int8_t) q[2 + j];
        }
    } else {
        std::fprintf(stderr, "strata-flint: %s is %s, expected F32, BF16, F16 or Q8_0\n", t.name.c_str(), t.type_name());
        std::exit(1);
    }
    float* d = nullptr;
    CK(cudaMalloc(&d, h.size() * 4));
    CK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    return d;
}

bool load(const std::string& path, Model& m, std::string& err) {
    const auto t0 = Clock::now();
    m.file = std::make_shared<strata::GgufModel>(strata::GgufModel::open(path));
    const strata::GgufModel& gm = *m.file;
    err = strata::check_architecture(gm.meta());
    if (!err.empty()) return false;
    if (strata::model_family(gm.meta()) != strata::ModelFamily::Flint) {
        err = "strata-flint runs Flint (architecture 'flint') files only";
        return false;
    }
    if (gm.size() != 1) {
        err = "strata-flint reads the experts from one GGUF file; a split model is not supported yet";
        return false;
    }
    m.file_path = gm.shard(0).path();
    const std::string p = "flint.";
    auto num = [&](const char* k) { return (int64_t) gm.meta().get(p + k)->u; };
    m.N = num("embedding_length");
    m.FF = num("feed_forward_length");
    m.H = num("attention.head_count");
    m.HKV = num("attention.head_count_kv");
    m.L = num("block_count");
    m.NE = num("expert_count");
    m.TK = num("expert_used_count");
    m.FFE = num("expert_feed_forward_length");
    m.FFS = num("expert_shared_feed_forward_length");
    m.n_dense = num("leading_dense_block_count");
    m.window = num("attention.sliding_window");
    m.eps = (float) gm.meta().get(p + "attention.layer_norm_rms_epsilon")->num();
    m.rope_base = (float) gm.meta().get(p + "rope.freq_base")->num();
    if (const strata::MetaValue* v = gm.meta().get(p + "expert_weights_scale")) m.w_scale = (float) v->num();
    if (!gm.meta().get(p + "expert_weights_norm") || gm.meta().get(p + "expert_weights_norm")->u == 0) {
        err = "flint.expert_weights_norm is false; the engine is written for the normalised router";
        return false;
    }
    if (m.TK > kMaxK || m.NE > 512 || m.N % 128 != 0) {
        err = "a geometry this driver's kernels do not take";
        return false;
    }
    if (const strata::MetaValue* v = gm.meta().get("tokenizer.ggml.eog_token_ids"))
        for (const strata::MetaValue& e : v->items) m.eog.push_back((int32_t) e.u);
    if (m.eog.empty())
        if (const strata::MetaValue* v = gm.meta().get("tokenizer.ggml.eos_token_id")) m.eog.push_back((int32_t) v->u);
    const strata::MetaValue* swa = gm.meta().get(p + "attention.sliding_window_pattern");
    const strata::MetaValue* rop = gm.meta().get(p + "attention.rope_pattern");
    if (const strata::MetaValue* v = gm.meta().get("tokenizer.ggml.tokens")) m.VOC = (int64_t) v->count;
    uint64_t bytes = 0;
    auto find = [&](const std::string& name, size_t& s) -> const strata::TensorInfo* {
        const strata::TensorInfo* t = gm.find(name, &s);
        if (!t) {
            std::fprintf(stderr, "strata-flint: missing %s\n", name.c_str());
            std::exit(1);
        }
        return t;
    };
    auto f32 = [&](const std::string& name) -> const float* {
        size_t s = 0;
        const strata::TensorInfo* t = find(name, s);
        bytes += t->elements() * 4;
        return upload_f32(*t, gm.shard(s).tensor_data(*t));
    };
    auto mat = [&](const std::string& name) -> Mat {
        size_t s = 0;
        const strata::TensorInfo* t = find(name, s);
        Mat w;
        w.type = (int) t->type;
        w.n_in = (int) t->shape[0];
        w.n_out = (int) t->shape[1];
        if (!K::native_mmvq_supported(w.type)) {
            std::fprintf(stderr, "strata-flint: %s is %s, which the native MMVQ does not take\n", name.c_str(), t->type_name());
            std::exit(1);
        }
        const uint64_t nb = strata::tensor_payload_bytes(*t);
        uint8_t* d = nullptr;
        CK(cudaMalloc(&d, nb));
        CK(cudaMemcpy(d, gm.shard(s).tensor_data(*t), nb, cudaMemcpyHostToDevice));
        bytes += nb;
        w.d = d;
        return w;
    };
    auto Ln = [](int64_t l, const char* s) { return "blk." + std::to_string(l) + "." + s; };
    m.layers.resize((size_t) m.L);
    for (int64_t l = 0; l < m.L; ++l) {
        Layer& y = m.layers[(size_t) l];
        y.swa = swa->items[(size_t) l].u != 0;
        y.rope = rop->items[(size_t) l].u != 0;
        y.dense = l < m.n_dense;
        y.attn_norm = f32(Ln(l, "attn_norm.weight"));
        y.ffn_norm = f32(Ln(l, "ffn_norm.weight"));
        y.q_norm = f32(Ln(l, "attn_q_norm.weight"));
        y.k_norm = f32(Ln(l, "attn_k_norm.weight"));
        y.sinks = f32(Ln(l, "attn_sinks.weight"));
        y.wq = mat(Ln(l, "attn_q.weight"));
        y.wk = mat(Ln(l, "attn_k.weight"));
        y.wv = mat(Ln(l, "attn_v.weight"));
        y.wo = mat(Ln(l, "attn_output.weight"));
        if (y.dense) {
            y.ffn_gate = mat(Ln(l, "ffn_gate.weight"));
            y.ffn_up = mat(Ln(l, "ffn_up.weight"));
            y.ffn_down = mat(Ln(l, "ffn_down.weight"));
            continue;
        }
        y.router = f32(Ln(l, "ffn_gate_inp.weight"));
        y.sh_gate = mat(Ln(l, "ffn_gate_shexp.weight"));
        y.sh_up = mat(Ln(l, "ffn_up_shexp.weight"));
        y.sh_down = mat(Ln(l, "ffn_down_shexp.weight"));
        size_t sg = 0, su = 0, sd = 0;
        const strata::TensorInfo* tg = find(Ln(l, "ffn_gate_exps.weight"), sg);
        const strata::TensorInfo* tu = find(Ln(l, "ffn_up_exps.weight"), su);
        const strata::TensorInfo* td = find(Ln(l, "ffn_down_exps.weight"), sd);
        if (tg->type != tu->type) {
            err = Ln(l, "ffn_gate_exps / ffn_up_exps differ in type");
            return false;
        }
        if (!K::native_expert_supported((int) tg->type, (int) td->type, m.N, m.FFE)) {
            err = Ln(l, "experts are ") + tg->type_name() + "/" + td->type_name() +
                  ", which the native expert kernels do not take at this size";
            return false;
        }
        y.L = K::native_expert_layout((int) tg->type, (int) td->type, m.N, m.FFE);
        y.src_gsz = (size_t) m.FFE * y.L.gu_row;
        y.src_dsz = (size_t) m.N * y.L.d_row;
        const size_t al = lcm_sz(lcm_sz(block_bytes(y.L.gu_type), block_bytes(y.L.d_type)), 16);
        y.stride = (y.L.bytes + al - 1) / al * al;
        y.off_g = gm.shard(sg).data_start() + tg->offset;
        y.off_u = gm.shard(su).data_start() + tu->offset;
        y.off_d = gm.shard(sd).data_start() + td->offset;
        y.src_g = gm.shard(sg).tensor_data(*tg);
        y.src_u = gm.shard(su).tensor_data(*tu);
        y.src_d = gm.shard(sd).tensor_data(*td);
    }
    m.out_norm = f32("output_norm.weight");
    m.head = mat("output.weight");
    {
        size_t s = 0;
        const strata::TensorInfo* t = find("token_embd.weight", s);
        m.embd_type = (int) t->type;
        if (m.embd_type != 8 && m.embd_type != 30 && m.embd_type != 1 && m.embd_type != 0) {
            err = std::string("token_embd.weight is ") + t->type_name() + ", expected Q8_0, BF16, F16 or F32";
            return false;
        }
        m.embd_row = row_bytes(m.embd_type, (int64_t) t->shape[0]);
        const uint64_t nb = strata::tensor_payload_bytes(*t);
        // one row of the table is read per token: it stays in page-locked host memory (the kernel reads the row over the bus),
        // and the VRAM it would take goes to the expert cache (0.3 GiB is about 3 tokens/s of decode).  STRATA_FLINT_EMBD_VRAM=1
        // puts it in the VRAM.
        uint8_t* d = nullptr;
        void* hp = nullptr;
        if (!std::getenv("STRATA_FLINT_EMBD_VRAM") && cudaHostAlloc(&hp, nb, cudaHostAllocMapped) == cudaSuccess) {
            std::memcpy(hp, gm.shard(s).tensor_data(*t), nb);
            void* dp = nullptr;
            if (cudaHostGetDevicePointer(&dp, hp, 0) == cudaSuccess && dp) {
                m.embd = (const uint8_t*) dp;
            } else {
                (void) cudaGetLastError();
                cudaFreeHost(hp);
                hp = nullptr;
            }
        } else {
            (void) cudaGetLastError();
        }
        if (!hp) {
            CK(cudaMalloc(&d, nb));
            CK(cudaMemcpy(d, gm.shard(s).tensor_data(*t), nb, cudaMemcpyHostToDevice));
            m.embd = d;
            bytes += nb;
        }
    }
    std::fprintf(stderr, "strata-flint: %s: %.2f GiB on the GPU in %.1f s (the routed experts are placed after the "
                 "engine's buffers)\n", path.c_str(), (double) bytes / (1 << 30), ms_since(t0) / 1000.0);
    return true;
}

// ------------------------------------------------------------------------------------------------ the engine
/// A small fixed thread pool: run(n, fn) calls fn(0 .. n-1) on the workers and the caller, and returns when all are done.
class Pool {
public:
    explicit Pool(int n) {
        for (int i = 0; i < n; ++i) threads_.emplace_back([this] { worker(); });
    }
    ~Pool() {
        {
            std::lock_guard<std::mutex> g(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& t : threads_) t.join();
    }
    void run(int n, const std::function<void(int)>& fn) {
        std::unique_lock<std::mutex> lk(mu_);
        job_ = &fn;
        total_ = n;
        next_ = 0;
        done_ = 0;
        cv_.notify_all();
        while (next_ < total_) {
            const int i = next_++;
            lk.unlock();
            fn(i);
            lk.lock();
            ++done_;
        }
        dcv_.wait(lk, [&] { return done_ == total_; });
        job_ = nullptr;
    }

private:
    void worker() {
        std::unique_lock<std::mutex> lk(mu_);
        for (;;) {
            cv_.wait(lk, [&] { return stop_ || (job_ && next_ < total_); });
            if (stop_) return;
            while (job_ && next_ < total_) {
                const int i = next_++;
                lk.unlock();
                (*job_)(i);
                lk.lock();
                if (++done_ == total_) dcv_.notify_all();
            }
        }
    }
    std::mutex mu_;
    std::condition_variable cv_, dcv_;
    const std::function<void(int)>* job_ = nullptr;
    int total_ = 0, next_ = 0, done_ = 0;
    bool stop_ = false;
    std::vector<std::thread> threads_;
};

size_t round_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

struct Engine {
    const Model* m = nullptr;
    cudaStream_t st = nullptr;
    int ctx = 0;
    int tmax = 512;   // rows of every activation buffer: the longest prompt chunk
    bool trace = false;
    std::vector<float>* nll = nullptr;          // --nll: per position, -log p(next prompt token); filled by forward()
    const int32_t* nll_next = nullptr;          // the prompt (position i + 1 is the target of position i)
    int nll_len = 0;
    float* lgb = nullptr;                       // [16][vocab] logits of a block of rows
    std::vector<float>* dump = nullptr;   // --dump-layers: [position][layer][n_embd] f32, the residual after each layer
    // activations
    float *x = nullptr, *h = nullptr, *qb = nullptr, *kb = nullptr, *vb = nullptr, *qr = nullptr, *att = nullptr, *y = nullptr;
    float* h2 = nullptr;
    float *g1 = nullptr, *u1 = nullptr, *sg = nullptr, *su = nullptr, *sy = nullptr, *logits_r = nullptr, *ewts = nullptr;
    float *eparts = nullptr, *pacc = nullptr, *pm = nullptr, *pl = nullptr, *lg = nullptr;
    void *xq = nullptr, *q8 = nullptr, *escr = nullptr;
    int32_t *tok = nullptr, *ids = nullptr, *ent_dst = nullptr, *ent_tok = nullptr, *grp_start = nullptr, *n_groups = nullptr;
    unsigned long long* grp_ptr = nullptr;
    int max_chunks = 0;
    // KV caches, per layer
    std::vector<__half*> kc, vc;
    std::vector<int> ncell;
    // the quantised KV cache of the full-attention layers (--kv-quant): kv_mode 0 = FP16 (the default), 1 = trellis, 2 = HyperQuant
    int kv_mode = 0, kv_bits = 3;
    bool kv_check = false;   // STRATA_FLINT_KV_CHECK=1: the error of the codec against the vectors it was given, printed at the end
    void kv_report() {
        if (!kv_check) return;
        double h[4] = {0, 0, 0, 0};
        CK(cudaStreamSynchronize(st));
        CK(cudaMemcpyFromSymbol(h, g_kv_chk, sizeof(h)));
        auto db = [](double a, double e) { return a > 0 && e > 0 ? 10.0 * std::log10(a / e) : 0.0; };
        std::fprintf(stderr, "strata-flint: KV codec error (long-prompt path): K %.2f dB, V %.2f dB (signal / error)\n", db(h[0], h[1]), db(h[2], h[3]));
    }
    std::vector<uint8_t*> kq_code, vq_code, kq_mt, vq_mt;
    std::vector<__half*> kq_sc, vq_sc;
    __half *kv_kst = nullptr, *kv_vst = nullptr, *kv_k16 = nullptr, *kv_v16 = nullptr;   // staging, and the FP16 copy a long prompt reads
    static constexpr int kKvStageT = 512;
    bool kvq(size_t li) const { return kv_mode != 0 && kq_code[li] != nullptr; }
    // streaming
    // ---- the prompt path for long chunks (T > kPf2Min): GEMMs on the matrix cores, the experts through llama.cpp's MMQ, a
    // loader thread that has the next groups of experts waiting in a ring of buffers while the GPU computes
    static constexpr int kPf2Min = 64;
    static constexpr int kPfNB = 8;       // group buffers in the ring
    static constexpr int kPfGroup = 16;   // experts per MMQ launch
    bool pf2 = false;
    strata::prefill::Gemm gemm;
    cublasHandle_t blas = nullptr;
    __half *x16 = nullptr, *q16 = nullptr, *k16w = nullptr, *v16w = nullptr, *att_p16 = nullptr;
    float* att_s = nullptr;
    size_t att_s_floats = 0;
    std::unique_ptr<MMQ::Context> mctx;
    int32_t *rcnt = nullptr, *rcur = nullptr, *rbounds = nullptr, *ident = nullptr, *rinv = nullptr;
    float *rsc_e = nullptr, *rsc_s = nullptr;   // the factors that put back the rows k_row_fit scaled down (experts, per sorted row; dense / shared, per token)
    static constexpr float kFitTarget = 1024.0f;
    void *mq_x = nullptr, *mq_hq = nullptr;
    float *mq_gu = nullptr, *mq_h = nullptr;
    size_t pf_S1 = 0, pf_S2 = 0;          // bytes between experts in a group buffer: gate+up, down
    uint8_t* pf_gbuf[kPfNB] = {};
    cudaEvent_t pf_ready[kPfNB] = {}, pf_free[kPfNB] = {};
    cudaStream_t pf_sio = nullptr;
    uint8_t* pf_pin[2] = {};
    cudaEvent_t pf_pin_ev[2] = {};
    struct PfTask { int li, pass, group; };
    std::vector<PfTask> pf_tasks;
    std::mutex pf_mu;
    std::condition_variable pf_cv;
    size_t pf_produced = 0, pf_consumed = 0, pf_next = 0;
    std::thread pf_th;
    bool finite_check = std::getenv("STRATA_FLINT_CHECK_FINITE") != nullptr;
    int* d_cnt = nullptr;
    // the decode step's host work: ids come back in a pinned array, the groups go out in one pinned block (one copy, no wait)
    bool pro_pending = false;
    bool host_inplace = false;      // the decode reads host-tier experts where they are and promotes the repeated ones in the background
    int promote_after = 12;
    cudaStream_t st_pro = nullptr;  // the promotions' copies
    cudaEvent_t ev_pro = nullptr, ev_kd = nullptr;   // ev_kd: the expert kernel of the call is done
    FILE* trace_f = nullptr;
    int32_t* h_ids = nullptr;
    uint8_t *h_meta = nullptr, *d_meta = nullptr;
    // (for up to kFastEnt = 64 tokens x 6 experts: groups, group starts, their count, the entries' slots and tokens)
    static constexpr int kFastEnt = 384;
    static constexpr size_t kMetaGP = 0, kMetaGS = 3072, kMetaNG = 4624, kMetaED = 4640, kMetaET = 6176, kMetaBytes = 7712;
    bool fake_hit = false, skip_tok_copy = false;   // --bench-floor
    // --bench-floor: where the GPU time of a decode step goes, by section (events between the sections)
    bool secprof = false;
    std::vector<std::pair<const char*, cudaEvent_t>> marks;
    std::map<std::string, double> sec_ms;
    std::vector<cudaEvent_t> mark_pool;
    void mark(const char* name) {
        if (!secprof) return;
        if (marks.size() >= mark_pool.size()) {
            cudaEvent_t e;
            CK(cudaEventCreate(&e));
            mark_pool.push_back(e);
        }
        CK(cudaEventRecord(mark_pool[marks.size()], st));
        marks.emplace_back(name, mark_pool[marks.size()]);
    }
    void marks_collect() {
        if (!secprof || marks.empty()) return;
        CK(cudaStreamSynchronize(st));
        for (size_t i = 1; i < marks.size(); ++i) {
            float ms = 0;
            CK(cudaEventElapsedTime(&ms, marks[i - 1].second, marks[i].second));
            sec_ms[marks[i].first] += ms;
        }
        marks.clear();
    }
    bool pf_dry = std::getenv("STRATA_FLINT_PF_DRY") != nullptr;   // timing only: no expert is loaded, the products run on whatever the buffers hold                  // the warmup: the kernels run on whatever the buffers hold, no expert is loaded
    long clock_ = 0;
    bool pin_host = true;           // --no-pin-host: do not page-lock the host tier (then the swaps are staged and synchronous)
    double host_gib = -1;           // --host-cache: GiB of RAM for the second tier (< 0: what is free less a margin)
    uint8_t* swap_tmp = nullptr;    // pinned: a VRAM victim on its way to the host tier
    long host_hits_ = 0, ssd_reads_ = 0, demoted_ = 0, pf_loaded_ = 0;
    cudaStream_t st2 = nullptr;     // copies of the lookahead, beside the compute on st
    cudaStream_t cs = nullptr;      // the stream the expert copies of the moment go on (st, or st2 during the lookahead)
    cudaEvent_t ev = nullptr;
    bool host_direct = false;       // the GPU reads the experts of the host tier in place
    bool host_pinned = false;       // the host tier is page-locked: swaps are asynchronous DMA, no staging, no memcpy
    int prefetch_k = 0;             // experts of the next layer, from the guess, to bring into the VRAM cache early
    // --prof-predict: how well the next layer's experts can be guessed from this layer's residual (decode only)
    bool prof_predict = false;
    float* plog = nullptr;                    // [NE] logits of the guess
    int32_t* pids = nullptr;                  // device scratch for the guess' top-K
    float* pw = nullptr;
    std::vector<int32_t> guess_next;          // the guess for layer guess_layer (top 16 by router logit), empty if none
    size_t guess_layer = (size_t) -1;
    long pr_total = 0, pr_in6 = 0, pr_in12 = 0, pr_in16 = 0, pr_miss = 0, pr_miss_in12 = 0, pr_miss_in16 = 0;
    uint8_t* tmp_blobs = nullptr;   // before the caches exist (the warmup): up to 48 experts of any layer, read each time
    size_t tmp_stride = 0;
    std::vector<int> resident;   // per layer: 0 streamed, 1 resident
    // direct reads of the experts: O_DIRECT preads, 8 at a time, into a pinned buffer (slots x 3 parts x pin_cap bytes)
    int fd = -1;
    uint8_t* pin = nullptr;
    size_t pin_cap = 0;
    int pin_slots = 0;
    std::unique_ptr<Pool> pool;
    std::mutex pool_mu;   // Pool::run is for one caller at a time
    bool buffered = false;
    // --prof: where the decode time goes, and how well an LRU of experts per layer would do (simulated)
    bool prof = false;
    double t_sync = 0, t_read = 0, t_up = 0, t_fwd = 0, t_ssdread = 0;
    long n_calls = 0, n_fwd = 0, hits_ = 0, total_ = 0;
    std::vector<std::vector<int>> recency;   // per layer, most recent first
    long lru_hits[6] = {0, 0, 0, 0, 0, 0}, lru_total = 0;
    static constexpr int kLruCaps[6] = {6, 12, 24, 48, 96, 160};
    std::vector<long> layer_use;             // [layer][expert] choices in decode, for a static top-N estimate

    void init(const Model& model, int ctx_) {
        m = &model;
        ctx = ctx_;
        CK(cudaStreamCreate(&st));
        CK(cudaStreamCreateWithFlags(&st2, cudaStreamNonBlocking));
        CK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
        CK(cudaStreamCreateWithFlags(&st_pro, cudaStreamNonBlocking));
        CK(cudaEventCreateWithFlags(&ev_pro, cudaEventDisableTiming));
        CK(cudaEventCreateWithFlags(&ev_kd, cudaEventDisableTiming));
        cs = st;
        const size_t N = (size_t) m->N, T = (size_t) tmax, K_ = (size_t) m->TK;
        auto dal = [](size_t n) {
            float* p = nullptr;
            CK(cudaMalloc(&p, n * 4));
            CK(cudaMemset(p, 0, n * 4));
            return p;
        };
        x = dal(T * N);
        h = dal(T * N);
        y = dal(T * N);
        sy = dal(T * N);
        qb = dal(T * m->H * kHD);
        qr = dal(T * m->H * kHD);
        att = dal(T * m->H * kHD);
        kb = dal(T * m->HKV * kHD);
        vb = dal(T * m->HKV * kHD);
        g1 = dal(T * (size_t) std::max(m->FF, m->FFS));
        u1 = dal(T * (size_t) std::max(m->FF, m->FFS));
        sg = dal(T * (size_t) m->FFS);
        su = dal(T * (size_t) m->FFS);
        logits_r = dal(T * (size_t) m->NE);
        ewts = dal(T * K_);
        eparts = dal(T * K_ * N);
        lg = dal((size_t) m->VOC);
        h2 = dal((size_t) m->N);
        plog = dal((size_t) m->NE);
        const size_t qbytes = K::native_q8_1_bytes((int) std::max<int64_t>({m->FF, m->N, m->H * kHD}), 1) * (size_t) tmax + 4096;
        CK(cudaMalloc(&xq, qbytes));
        CK(cudaMalloc(&q8, qbytes));
        CK(cudaMalloc(&escr, K::native_expert_scratch_bytes((int64_t) (T * K_), m->FFE) + 4096));
        CK(cudaMalloc(&tok, T * 4));
        CK(cudaMalloc(&ids, T * K_ * 4));
        CK(cudaMalloc(&ent_dst, T * K_ * 4));
        CK(cudaMalloc(&ent_tok, T * K_ * 4));
        CK(cudaMalloc(&grp_start, (T * K_ + 2) * 4));
        CK(cudaMalloc(&n_groups, 4));
        CK(cudaMalloc(&grp_ptr, (T * K_ + 2) * 8));
        // attention partials for the longest context (full layers): chunks x (kQB queries * H) x 128
        max_chunks = (ctx + kAttChunk - 1) / kAttChunk + 1;
        pacc = dal((size_t) max_chunks * kQB * m->H * kHD);
        pm = dal((size_t) max_chunks * kQB * m->H);
        pl = dal((size_t) max_chunks * kQB * m->H);
        // KV caches
        size_t kv_bytes = 0;
        if (kv_mode && (m->H % m->HKV != 0 || m->H / m->HKV != kKvG)) {
            std::fprintf(stderr, "strata-flint: --kv-quant needs %d query heads per KV head (this model has %lld / %lld)\n", kKvG,
                         (long long) m->H, (long long) m->HKV);
            std::exit(2);
        }
        for (const Layer& L : m->layers) {
            const int nc = L.swa ? kRing : ctx;
            ncell.push_back(nc);
            const size_t n = (size_t) m->HKV * nc * kHD;
            __half *k_ = nullptr, *v_ = nullptr;
            uint8_t *kcq = nullptr, *vcq = nullptr, *kmt = nullptr, *vmt = nullptr;
            __half *ksc = nullptr, *vsc = nullptr;
            if (kv_mode && !L.swa) {
                const size_t nvec = (size_t) m->HKV * nc, pbytes = nvec * 16 * kv_bits + 64;
                CK(cudaMalloc(&kcq, pbytes));
                CK(cudaMalloc(&vcq, pbytes));
                CK(cudaMemset(kcq, 0, pbytes));
                CK(cudaMemset(vcq, 0, pbytes));
                CK(cudaMalloc(&ksc, nvec * 2));
                CK(cudaMalloc(&vsc, nvec * 2));
                CK(cudaMemset(ksc, 0, nvec * 2));
                CK(cudaMemset(vsc, 0, nvec * 2));
                kv_bytes += 2 * (pbytes + nvec * 2);
                if (kv_mode == 2) {
                    CK(cudaMalloc(&kmt, nvec));
                    CK(cudaMalloc(&vmt, nvec));
                    CK(cudaMemset(kmt, 0, nvec));
                    CK(cudaMemset(vmt, 0, nvec));
                    kv_bytes += 2 * nvec;
                }
            } else {
                CK(cudaMalloc(&k_, n * 2));
                CK(cudaMalloc(&v_, n * 2));
                CK(cudaMemset(k_, 0, n * 2));
                CK(cudaMemset(v_, 0, n * 2));
                kv_bytes += 2 * n * 2;
            }
            kc.push_back(k_);
            vc.push_back(v_);
            kq_code.push_back(kcq);
            vq_code.push_back(vcq);
            kq_mt.push_back(kmt);
            vq_mt.push_back(vmt);
            kq_sc.push_back(ksc);
            vq_sc.push_back(vsc);
        }
        size_t scratch = 0;
        if (kv_mode) {   // the encoder's staging, and the FP16 copy of one layer's keys that a long prompt reads
            const size_t stg = (size_t) kKvStageT * m->HKV * kHD * 2, cp = (size_t) m->HKV * ctx * kHD * 2;
            CK(cudaMalloc(&kv_kst, stg));
            CK(cudaMalloc(&kv_vst, stg));
            CK(cudaMalloc(&kv_k16, cp));
            CK(cudaMalloc(&kv_v16, cp));
            scratch = 2 * stg + 2 * cp;
        }
        if (kv_mode)
            std::fprintf(stderr, "strata-flint: context %d cells, KV cache %.2f GiB (%s, %d bits per sample, in the full layers; the sliding-window layers stay FP16) + %.2f GiB scratch\n",
                         ctx, (double) kv_bytes / (1 << 30), kv_mode == 1 ? "TurboQuant Trellis" : "HyperQuant", kv_bits,
                         (double) scratch / (1 << 30));
        else
            std::fprintf(stderr, "strata-flint: context %d cells, KV cache %.2f GiB (FP16)\n", ctx, (double) kv_bytes / (1 << 30));
    }

    /// Staging for the streamed layers (the largest expert blob x the most entries one forward can have), the direct
    /// reader of the experts and its pinned buffer.  Call first.
    void init_staging() {
        const Model& M = *m;
        resident.assign(M.layers.size(), 0);
        size_t max_part = 0;
        for (const Layer& L : M.layers)
            if (!L.dense) max_part = std::max({max_part, L.src_gsz, L.src_dsz});
        pin_slots = (int) (kMaxT * M.TK);   // 48: experts read at once
        for (const Layer& L : M.layers)
            if (!L.dense) tmp_stride = std::max(tmp_stride, L.stride);
        CK(cudaMalloc(&tmp_blobs, tmp_stride * (size_t) pin_slots + 4096));
        pin_cap = round_up(max_part + 8192, 4096);
        CK(cudaHostAlloc((void**) &swap_tmp, tmp_stride + 4096, cudaHostAllocDefault));
        CK(cudaHostAlloc((void**) &h_ids, kFastEnt * 4, cudaHostAllocDefault));
        if (const char* tf = std::getenv("STRATA_FLINT_TRACE")) trace_f = std::fopen(tf, "wb");
        CK(cudaHostAlloc((void**) &h_meta, kMetaBytes, cudaHostAllocDefault));
        CK(cudaMalloc(&d_meta, kMetaBytes));
        CK(cudaHostAlloc((void**) &pin, pin_cap * 3 * (size_t) pin_slots, cudaHostAllocDefault));
        fd = buffered ? -1 : open(M.file_path.c_str(), O_RDONLY | O_DIRECT);
        if (fd < 0) fd = open(M.file_path.c_str(), O_RDONLY);
        if (fd < 0) {
            std::fprintf(stderr, "strata-flint: cannot open %s\n", M.file_path.c_str());
            std::exit(1);
        }
        pool = std::make_unique<Pool>(15);   // plus the calling thread
    }

    /// Reads the parts of experts[0 .. n) of layer L named by `mask` (bit 0 gate, 1 up, 2 down) into the pinned buffer
    /// `pinbase` (n <= pin_slots); part p of expert k is then at part_ptr(L, experts[k], k, p, pinbase).
    /// Part p of expert e of layer L, read with one aligned O_DIRECT read into dst (page-aligned, round_up(len + 4096, 4096)
    /// bytes of room): the part starts at dst + part_shift(L, e, p).
    void read_one_part(const Layer& L, uint64_t e, int p, uint8_t* dst) {
        const size_t len = p == 2 ? L.src_dsz : L.src_gsz;
        const uint64_t off = (p == 0 ? L.off_g : p == 1 ? L.off_u : L.off_d) + e * len;
        const uint64_t aoff = off & ~(uint64_t) 4095;
        const size_t want = (size_t) (off - aoff) + len;
        const size_t alen = round_up(want, 4096);
        size_t got = 0;
        while (got < want) {
            const ssize_t r = pread(fd, dst + got, alen - got, (off_t) (aoff + got));
            if (r <= 0) {
                std::fprintf(stderr, "strata-flint: read of expert %llu failed (%zd, got %zu of %zu)\n", (unsigned long long) e, r,
                             got, want);
                std::exit(1);
            }
            got += (size_t) r;
        }
    }
    static size_t part_shift(const Layer& L, uint64_t e, int p) {
        const size_t len = p == 2 ? L.src_dsz : L.src_gsz;
        return (size_t) (((p == 0 ? L.off_g : p == 1 ? L.off_u : L.off_d) + e * len) & 4095);
    }
    void read_parts(const Layer& L, const int32_t* experts, int n, int mask = 7, uint8_t* pinbase = nullptr) {
        uint8_t* pb = pinbase ? pinbase : pin;
        std::lock_guard<std::mutex> lk(pool_mu);
        pool->run(n * 3, [&](int j) {
            const int k = j / 3, p = j % 3;
            if (!((mask >> p) & 1)) return;
            read_one_part(L, (uint64_t) experts[k], p, pb + (size_t) j * pin_cap);
        });
    }
    const uint8_t* part_ptr(const Layer& L, int32_t e, int k, int p, const uint8_t* pinbase = nullptr) const {
        const size_t len = p == 2 ? L.src_dsz : L.src_gsz;
        const uint64_t off = (p == 0 ? L.off_g : p == 1 ? L.off_u : L.off_d) + (uint64_t) e * len;
        return (pinbase ? pinbase : pin) + (size_t) (k * 3 + p) * pin_cap + (size_t) (off & 4095);
    }
    /// Copies expert k (read by read_parts) into the blob at dst on the device.
    void upload_expert(const Layer& L, int32_t e, int k, uint8_t* dst) {
        CK(cudaMemcpyAsync(dst, part_ptr(L, e, k, 0), L.src_gsz, cudaMemcpyHostToDevice, cs));
        CK(cudaMemcpyAsync(dst + L.L.up_off, part_ptr(L, e, k, 1), L.src_gsz, cudaMemcpyHostToDevice, cs));
        CK(cudaMemcpyAsync(dst + L.L.down_off, part_ptr(L, e, k, 2), L.src_dsz, cudaMemcpyHostToDevice, cs));
    }

    /// Places the routed experts in `budget` bytes of VRAM.  If every expert of every layer fits, they are all resident.
    /// Otherwise each MoE layer gets a cache of the same share of its experts (an LRU of slots, filled on demand from the
    /// file): measured on the code prompt, 52% of the experts per layer in every layer hits ~87% of the choices, where
    /// whole layers resident (and the rest streamed) hit 53%.
    void place_experts(size_t budget) {
        const Model& M = *m;
        const auto t0 = Clock::now();
        size_t all = 0;
        int n_moe = 0;
        for (const Layer& L : M.layers)
            if (!L.dense) {
                all += L.stride * (size_t) M.NE + 4096;
                ++n_moe;
            }
        if (all <= budget) {
            std::vector<int32_t> ex((size_t) pin_slots);
            for (size_t l = 0; l < M.layers.size(); ++l) {
                const Layer& L = M.layers[l];
                if (L.dense) continue;
                Layer& W = const_cast<Layer&>(L);
                const size_t need = L.stride * (size_t) M.NE + 4096;
                CK(cudaMalloc(&W.blobs, need));
                CK(cudaMemset(W.blobs + L.stride * (size_t) M.NE, 0, 4096));
                for (int64_t e0 = 0; e0 < M.NE; e0 += pin_slots) {
                    const int n = (int) std::min<int64_t>(pin_slots, M.NE - e0);
                    for (int k = 0; k < n; ++k) ex[(size_t) k] = (int32_t) (e0 + k);
                    read_parts(L, ex.data(), n);
                    for (int k = 0; k < n; ++k)
                        upload_expert(L, ex[(size_t) k], k, W.blobs + (size_t) ex[(size_t) k] * L.stride);
                    CK(cudaStreamSynchronize(st));
                }
                resident[l] = 1;
            }
            std::fprintf(stderr, "strata-flint: experts: all %d MoE layers resident in VRAM (%.2f GiB, %.1f s)\n", n_moe,
                         (double) all / (1 << 30), ms_since(t0) / 1000.0);
            return;
        }
        const size_t per = budget / (size_t) n_moe;
        size_t used = 0;
        int min_s = 1 << 30, max_s = 0;
        for (size_t l = 0; l < M.layers.size(); ++l) {
            const Layer& L = M.layers[l];
            if (L.dense) continue;
            Layer& W = const_cast<Layer&>(L);
            int S = per > 4096 ? (int) std::min<size_t>((size_t) M.NE, (per - 4096) / L.stride) : 0;
            if (S < pin_slots) S = pin_slots;   // a chunk of 8 tokens can choose 48 different experts of one layer
            W.S = S;
            CK(cudaMalloc(&W.cblob, L.stride * (size_t) S + 4096));
            CK(cudaMemset(W.cblob + L.stride * (size_t) S, 0, 4096));
            W.exp_slot.assign((size_t) M.NE, -1);
            W.slot_exp.assign((size_t) S, -1);
            W.slot_use.assign((size_t) S, 0);
            used += L.stride * (size_t) S + 4096;
            min_s = std::min(min_s, S);
            max_s = std::max(max_s, S);
        }
        // the host tier: the RAM the system can spare (MemAvailable less a margin), up to what the VRAM caches leave out
        size_t host_bytes = 0;
        if (host_gib >= 0) {
            host_bytes = (size_t) (host_gib * (double) (1ull << 30));
        } else if (FILE* f = std::fopen("/proc/meminfo", "r")) {
            char line[256];
            unsigned long long kb = 0, tot = 0;
            while (std::fgets(line, sizeof line, f)) {
                (void) std::sscanf(line, "MemTotal: %llu kB", &tot);
                if (std::sscanf(line, "MemAvailable: %llu kB", &kb) == 1) break;
            }
            std::fclose(f);
            // what the system can spare less 2.5 GiB (the rest of the engine's pinned buffers, the server, the system), and at most
            // 70% of the RAM: MemAvailable counts a page cache that the pinned tier will push out
            const size_t avail = (size_t) kb * 1024, margin = (size_t) (2.5 * (double) (1ull << 30));
            host_bytes = avail > margin ? avail - margin : 0;
            if (tot > 0) host_bytes = std::min<size_t>(host_bytes, (size_t) ((double) tot * 1024.0 * 0.70));
        }
        size_t host_used = 0;
        int min_h = 1 << 30, max_h = 0;
        if (host_bytes > 0) {
            // the same share of the budget for every layer, but a layer never takes more than what its VRAM cache leaves out
            // (NE - S): what the capped layers leave over goes to the others, so the whole budget is used when it is needed
            size_t total = 0;
            std::vector<int> sh(M.layers.size(), 0);
            std::vector<size_t> open_layers;
            for (size_t l = 0; l < M.layers.size(); ++l)
                if (!M.layers[l].dense) open_layers.push_back(l);
            size_t left = host_bytes;
            while (!open_layers.empty()) {
                const size_t share = left / open_layers.size();
                std::vector<size_t> still;
                bool capped_any = false;
                for (size_t l : open_layers) {
                    const Layer& L = M.layers[l];
                    const size_t room = (size_t) M.NE - (size_t) L.S;
                    if (room * L.stride <= share) {   // this layer takes all it can use
                        sh[l] = (int) room;
                        left -= room * L.stride;
                        capped_any = true;
                    } else {
                        still.push_back(l);
                    }
                }
                if (!capped_any) {   // nobody is capped by an equal share: split what is left
                    for (size_t l : still) sh[l] = (int) (share / M.layers[l].stride);
                    break;
                }
                open_layers = still;
            }
            for (size_t l = 0; l < M.layers.size(); ++l)
                if (!M.layers[l].dense) total += (size_t) sh[l] * M.layers[l].stride;
            if (total > 0) {
                uint8_t* base = (uint8_t*) mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
                if (base == MAP_FAILED) {
                    std::fprintf(stderr, "strata-flint: cannot reserve %.2f GiB of host memory for the second tier\n", (double) total / (1 << 30));
                } else {
                    size_t off = 0;
                    for (size_t l = 0; l < M.layers.size(); ++l) {
                        const Layer& L = M.layers[l];
                        if (L.dense || sh[l] == 0) continue;
                        Layer& W = const_cast<Layer&>(L);
                        W.hblob = base + off;
                        W.Sh = sh[l];
                        W.hexp_slot.assign((size_t) M.NE, -1);
                        W.hot.assign((size_t) M.NE, 0);
                        W.hslot_exp.assign((size_t) W.Sh, -1);
                        W.hslot_use.assign((size_t) W.Sh, 0);
                        off += (size_t) W.Sh * L.stride;
                        min_h = std::min(min_h, W.Sh);
                        max_h = std::max(max_h, W.Sh);
                    }
                    host_used = total;
                    if (pin_host) {   // page-locked: the copies to the GPU skip the driver's staging buffer
                        const auto tr = Clock::now();
                        const cudaError_t e = cudaHostRegister(base, total, cudaHostRegisterPortable | cudaHostRegisterMapped);
                        if (e == cudaSuccess) {
                            host_pinned = true;
                            void* dptr = nullptr;
                            if (cudaHostGetDevicePointer(&dptr, base, 0) == cudaSuccess && dptr) {
                                for (size_t l2 = 0; l2 < M.layers.size(); ++l2) {
                                    const Layer& L2 = M.layers[l2];
                                    if (!L2.dense && L2.hblob) const_cast<Layer&>(L2).hdev = (uint8_t*) dptr + (L2.hblob - base);
                                }
                                host_inplace = !std::getenv("STRATA_FLINT_NO_INPLACE");
                                if (const char* pa = std::getenv("STRATA_FLINT_PROMOTE_AFTER")) promote_after = std::max(1, std::atoi(pa));
                                host_direct = std::getenv("STRATA_FLINT_DIRECT") != nullptr;   // an experiment: measured slower than the LRU with swaps (40 against 35 ms per token)
                            } else {
                                (void) cudaGetLastError();
                            }
                            std::fprintf(stderr, "strata-flint: host tier page-locked (%.1f s)\n", ms_since(tr) / 1000.0);
                        } else {
                            (void) cudaGetLastError();
                            std::fprintf(stderr, "strata-flint: host tier not page-locked (%s); copies go through the driver's staging\n",
                                         cudaGetErrorString(e));
                        }
                    }
                }
            }
        }
        std::fprintf(stderr, "strata-flint: experts: a VRAM cache of %d-%d experts per MoE layer (of %lld; %.2f GiB)", min_s, max_s,
                     (long long) M.NE, (double) used / (1 << 30));
        if (host_used > 0)
            std::fprintf(stderr, " and a host-RAM tier of %d-%d (%.2f GiB)", min_h, max_h, (double) host_used / (1 << 30));
        std::fprintf(stderr, ", filled on demand from the file (%.1f s)\n", ms_since(t0) / 1000.0);
        warm_fill();
    }

    /// Reads bytes [off, off + len) of the file with big parallel O_DIRECT reads into `buf` (pinned, page-aligned, with room for
    /// the alignment: len + 2 * 4096 + one piece); the bytes start at buf + (off & 4095).
    void read_range(uint64_t off, size_t len, uint8_t* buf) {
        constexpr size_t kPiece = 8u << 20;
        const uint64_t aoff = off & ~(uint64_t) 4095;
        const size_t need = (size_t) (off - aoff) + len, total = round_up(need, 4096);
        const int njobs = (int) ((total + kPiece - 1) / kPiece);
        std::lock_guard<std::mutex> lk(pool_mu);
        pool->run(njobs, [&](int j) {
            const size_t o = (size_t) j * kPiece, n = std::min(kPiece, total - o);
            size_t got = 0;
            while (got < n && o + got < need) {
                const ssize_t r = pread(fd, buf + o + got, n - got, (off_t) (aoff + o + got));
                if (r <= 0) {
                    std::fprintf(stderr, "strata-flint: read at %llu failed (%zd)\n", (unsigned long long) (aoff + o + got), r);
                    std::exit(1);
                }
                got += (size_t) r;
            }
        });
    }

    /// The caches at the start: the first S experts of every layer in the VRAM cache, the next Sh in the host tier, read from
    /// the file in big sequential pieces.  Which experts does not matter to a prompt (it uses them all); the decode's LRU
    /// replaces what it does not use.  A prompt then finds the tiers full from its first chunk, and a decode starts with
    /// something to hit.  STRATA_FLINT_NO_WARM=1 skips it (the caches then fill from the first prompt, and from decode misses).
    void warm_fill() {
        const Model& M = *m;
        if (std::getenv("STRATA_FLINT_NO_WARM")) return;
        const auto t0 = Clock::now();
        // in pieces of 64 experts: the staging buffers are 3 x 64 experts (about 0.1 GiB), not 3 whole layers
        constexpr int kPiece = 64;
        size_t gmax = 0, dmax = 0;
        for (const Layer& L : M.layers)
            if (!L.dense && L.S > 0) {
                gmax = std::max(gmax, (size_t) kPiece * L.src_gsz);
                dmax = std::max(dmax, (size_t) kPiece * L.src_dsz);
            }
        constexpr size_t kSlack = (8u << 20) + 3 * 4096;
        uint8_t *bg = nullptr, *bu = nullptr, *bd = nullptr;
        if (cudaHostAlloc((void**) &bg, gmax + kSlack, cudaHostAllocDefault) != cudaSuccess ||
            cudaHostAlloc((void**) &bu, gmax + kSlack, cudaHostAllocDefault) != cudaSuccess ||
            cudaHostAlloc((void**) &bd, dmax + kSlack, cudaHostAllocDefault) != cudaSuccess) {
            (void) cudaGetLastError();
            if (bg) cudaFreeHost(bg);
            if (bu) cudaFreeHost(bu);
            std::fprintf(stderr, "strata-flint: no memory for the start-up fill (the host tier is large for this machine: --host-cache GiB takes less); the caches fill from use\n");
            return;
        }
        size_t bytes = 0;
        for (size_t l = 0; l < M.layers.size(); ++l) {
            const Layer& L = M.layers[l];
            if (L.dense || L.S <= 0 || resident[l]) continue;
            const int n = (int) std::min<size_t>((size_t) M.NE, (size_t) (L.S + L.Sh));
            for (int e0 = 0; e0 < n; e0 += kPiece) {
                const int cnt = std::min(kPiece, n - e0);
                const uint64_t og = L.off_g + (uint64_t) e0 * L.src_gsz, ou = L.off_u + (uint64_t) e0 * L.src_gsz,
                               od = L.off_d + (uint64_t) e0 * L.src_dsz;
                read_range(og, (size_t) cnt * L.src_gsz, bg);
                read_range(ou, (size_t) cnt * L.src_gsz, bu);
                read_range(od, (size_t) cnt * L.src_dsz, bd);
                bytes += (size_t) cnt * (2 * L.src_gsz + L.src_dsz);
                const uint8_t* g0 = bg + (og & 4095);
                const uint8_t* u0 = bu + (ou & 4095);
                const uint8_t* d0 = bd + (od & 4095);
                for (int e = e0; e < e0 + cnt; ++e) {
                    const size_t k = (size_t) (e - e0);
                    if (e < L.S) {   // the VRAM cache: slot e holds expert e
                        uint8_t* dst = L.cblob + (size_t) e * L.stride;
                        CK(cudaMemcpyAsync(dst, g0 + k * L.src_gsz, L.src_gsz, cudaMemcpyHostToDevice, st));
                        CK(cudaMemcpyAsync(dst + L.L.up_off, u0 + k * L.src_gsz, L.src_gsz, cudaMemcpyHostToDevice, st));
                        CK(cudaMemcpyAsync(dst + L.L.down_off, d0 + k * L.src_dsz, L.src_dsz, cudaMemcpyHostToDevice, st));
                        L.slot_exp[(size_t) e] = e;
                        L.exp_slot[(size_t) e] = e;
                        L.slot_use[(size_t) e] = 0;
                    } else {         // the host tier: slot e - S
                        const int hs = e - L.S;
                        uint8_t* dst = L.hblob + (size_t) hs * L.stride;
                        std::memcpy(dst, g0 + k * L.src_gsz, L.src_gsz);
                        std::memcpy(dst + L.L.up_off, u0 + k * L.src_gsz, L.src_gsz);
                        std::memcpy(dst + L.L.down_off, d0 + k * L.src_dsz, L.src_dsz);
                        L.hslot_exp[(size_t) hs] = e;
                        L.hexp_slot[(size_t) e] = hs;
                        L.hslot_use[(size_t) hs] = 0;
                    }
                }
                CK(cudaStreamSynchronize(st));   // the staging buffers are read again by the next piece
            }
        }
        CK(cudaFreeHost(bg));
        CK(cudaFreeHost(bu));
        CK(cudaFreeHost(bd));
        std::fprintf(stderr, "strata-flint: caches filled with %.1f GiB in %.1f s (%.1f GB/s)\n", (double) bytes / (1 << 30),
                     ms_since(t0) / 1000.0, (double) bytes / 1e9 / (ms_since(t0) / 1000.0));
    }

    /// out [T][n_out] = W x for T columns of q8_1 activations, 8 columns per launch.
    void mm(const Mat& w, const void* q, float* out, int T) {
        if (T <= kCols) {
            K::native_mmvq(w.type, w.d, q, out, w.n_in, w.n_out, T, st);
            return;
        }
        const size_t qb = K::native_q8_1_bytes(w.n_in, 1);
        for (int c0 = 0; c0 < T; c0 += kCols)
            K::native_mmvq(w.type, w.d, (const char*) q + (size_t) c0 * qb, out + (size_t) c0 * w.n_out, w.n_in, w.n_out,
                           std::min(kCols, T - c0), st);
    }
    /// SwiGLU of g and u [T][n] and its q8_1 image, 8 columns per launch.
    void swiglu_q(const float* g, const float* u, void* q, int n, int T) {
        const size_t qb = K::native_q8_1_bytes(n, 1);
        for (int c0 = 0; c0 < T; c0 += kCols)
            K::native_swiglu_quantize_q8_1(g + (size_t) c0 * n, u + (size_t) c0 * n, (char*) q + (size_t) c0 * qb, n,
                                           std::min(kCols, T - c0), st);
    }
    void quant(const float* in, int n, int T, void* q) { K::native_quantize_q8_1(in, q, n * T, 1, st); }

    /// The dense SwiGLU block (layers 0 and 1) or the MoE block, on h (already normed, quantised in xq); adds to x.
    void ffn(const Layer& L, size_t li, int T) {
        const Model& M = *m;
        const int N = (int) M.N, Kp = (int) M.TK;
        if (L.dense) {
            mm(L.ffn_gate, xq, g1, T);
            mm(L.ffn_up, xq, u1, T);
            swiglu_q(g1, u1, q8, (int) M.FF, T);
            mm(L.ffn_down, q8, y, T);
            k_add<<<grid1((int64_t) T * N), kThreads, 0, st>>>(x, y, (int64_t) T * N);
            return;
        }
        k_gemv_f32<<<dim3(rows8((int) M.NE), T), kThreads, 0, st>>>(L.router, h, logits_r, N, (int) M.NE);
        k_route<<<T, 256, 0, st>>>(logits_r, (int) M.NE, Kp, M.w_scale, ids, ewts);
        if (fake_hit && T == 1 && L.S > 0) {   // --bench-floor: every expert a hit, no host work
            mark("  router+route");
            k_group_mod<<<1, 256, 0, st>>>(ids, Kp, Kp, (int) M.NE, L.cblob, L.stride, L.S, grp_ptr, grp_start, n_groups, ent_dst, ent_tok);
            mark("  group");
            K::native_expert_grouped(L.L, grp_ptr, grp_start, n_groups, ent_dst, ent_tok, std::min<int64_t>(Kp, M.NE), Kp, xq, escr, eparts, st);
            mark("  experts (native grouped)");
            mm(L.sh_gate, xq, sg, T);
            mm(L.sh_up, xq, su, T);
            swiglu_q(sg, su, q8, (int) M.FFS, T);
            mm(L.sh_down, q8, sy, T);
            mark("  shared expert");
            k_combine<<<T, kThreads, 0, st>>>(x, sy, eparts, ewts, Kp, N);
            mark("  combine");
            return;
        }
        if ((prof_predict || prefetch_k > 0) && T == 1 && li + 1 < M.layers.size() && !M.layers[li + 1].dense) {
            // the guess for the next layer: its router on this layer's residual (after the attention, before the MoE)
            const Layer& Ln = M.layers[li + 1];
            k_rms<<<1, kThreads, 0, st>>>(x, Ln.ffn_norm, h2, N, M.eps);
            k_gemv_f32<<<dim3(rows8((int) M.NE), 1), kThreads, 0, st>>>(Ln.router, h2, plog, N, (int) M.NE);
        }
        const int n_ent = T * Kp;
        if (resident[li]) {
            k_group<<<1, 256, 0, st>>>(ids, n_ent, Kp, (int) M.NE, L.blobs, L.stride, grp_ptr, grp_start, n_groups, ent_dst,
                                       ent_tok);
            K::native_expert_grouped(L.L, grp_ptr, grp_start, n_groups, ent_dst, ent_tok, std::min<int64_t>(n_ent, M.NE),
                                     n_ent, xq, escr, eparts, st);
        } else if (T <= 64 && !host_direct && !fake_hit && !prof_predict && prefetch_k <= 0 && L.S >= 48) {
            if (!moe_fast(L, li, T)) moe_cached(L, li, T);
        } else {
            moe_cached(L, li, T);
        }
        // the shared expert
        mm(L.sh_gate, xq, sg, T);
        mm(L.sh_up, xq, su, T);
        swiglu_q(sg, su, q8, (int) M.FFS, T);
        mm(L.sh_down, q8, sy, T);
        k_combine<<<T, kThreads, 0, st>>>(x, sy, eparts, ewts, Kp, N);
    }

    /// A slot of the host tier to put an expert in: an empty one, else the one that has been there longest; never one
    /// marked in `protect` (slots of this batch's experts that are still to be promoted).
    int host_victim(const Layer& L, const std::vector<char>& protect) const {
        int best = -1;
        for (int sl = 0; sl < L.Sh; ++sl) {
            if (!protect.empty() && protect[(size_t) sl]) continue;
            if (L.hslot_exp[(size_t) sl] < 0) return sl;
            if (best < 0 || L.hslot_use[(size_t) sl] < L.hslot_use[(size_t) best]) best = sl;
        }
        return best;
    }

    /// Makes the experts in `list` (not in the VRAM cache) present in slots of it, copying on stream `cs`: from the host
    /// tier when they are there (a copy over the PCIe), else read from the file (up to 48 at a time).  The expert each slot
    /// gives up goes down to the host tier, unless the round is large (a prompt chunk: that churn is not worth the copies).
    /// With a page-locked host tier all of this is asynchronous DMA; otherwise the demoted expert is staged through swap_tmp.
    void load_experts(const Layer& L, const std::vector<int32_t>& list) {
        const auto tp1 = Clock::now();
        const bool demote = L.Sh > 0 && list.size() <= 16;
        const bool fast = demote && host_pinned;
        for (size_t b0 = 0; b0 < list.size(); b0 += (size_t) pin_slots) {
            const int n = (int) std::min<size_t>((size_t) pin_slots, list.size() - b0);
            std::vector<int32_t> ssd;
            std::vector<int> ssd_k((size_t) n, -1);
            std::vector<char> protect;
            if (fast) protect.assign((size_t) L.Sh, 0);
            for (int k = 0; k < n; ++k) {
                const int32_t xk = list[b0 + (size_t) k];
                if (L.Sh > 0 && L.hexp_slot[(size_t) xk] >= 0) {
                    ++host_hits_;
                    if (fast) protect[(size_t) L.hexp_slot[(size_t) xk]] = 1;
                } else {
                    ssd_k[(size_t) k] = (int) ssd.size();
                    ssd.push_back(xk);
                    ++ssd_reads_;
                }
            }
            if (!ssd.empty()) {
                const auto ts0 = Clock::now();
                read_parts(L, ssd.data(), (int) ssd.size());
                t_ssdread += ms_since(ts0);
            }
            // the experts that come from the host tier first: a demotion made for one read from the file could otherwise
            // give away the host slot of an expert of this batch that is still to be promoted
            std::vector<int> order;
            for (int k = 0; k < n; ++k)
                if (ssd_k[(size_t) k] < 0) order.push_back(k);
            for (int k = 0; k < n; ++k)
                if (ssd_k[(size_t) k] >= 0) order.push_back(k);
            for (int k : order) {
                int victim = -1;   // an empty slot, else the least recently used one that this round does not use
                long best = 0;
                for (int sl = 0; sl < L.S; ++sl) {
                    if (L.slot_use[(size_t) sl] == clock_) continue;
                    if (victim < 0 || L.slot_use[(size_t) sl] < best) {
                        victim = sl;
                        best = L.slot_use[(size_t) sl];
                        if (L.slot_exp[(size_t) sl] < 0) break;
                    }
                }
                if (victim < 0) {
                    std::fprintf(stderr, "strata-flint: no free expert slot (S=%d)\n", L.S);
                    std::exit(1);
                }
                const int32_t xk = list[b0 + (size_t) k];
                uint8_t* vdst = L.cblob + (size_t) victim * L.stride;
                const int V = L.slot_exp[(size_t) victim];
                const int hsA = ssd_k[(size_t) k] < 0 ? L.hexp_slot[(size_t) xk] : -1;   // where xk is in the host tier
                int hsV = -1;                                                            // where V goes
                if (demote && V >= 0) {
                    if (fast) {
                        if (hsA >= 0) protect[(size_t) hsA] = 0;   // (its copy to the GPU is queued before anything that overwrites it)
                        hsV = host_victim(L, protect);
                        if (hsA >= 0) protect[(size_t) hsA] = 1;
                        if (hsV == hsA) hsV = -1;   // keep it simple: the slot A is leaving is not reused in the same step
                        if (hsV >= 0) {
                            if (L.hslot_exp[(size_t) hsV] >= 0) L.hexp_slot[(size_t) L.hslot_exp[(size_t) hsV]] = -1;   // dropped
                            CK(cudaMemcpyAsync(L.hblob + (size_t) hsV * L.stride, vdst, L.L.bytes, cudaMemcpyDeviceToHost, cs));
                        }
                    } else {
                        hsV = hsA >= 0 ? hsA : host_victim(L, {});
                        CK(cudaMemcpyAsync(swap_tmp, vdst, L.L.bytes, cudaMemcpyDeviceToHost, cs));   // V, before it is overwritten
                    }
                }
                if (ssd_k[(size_t) k] >= 0) upload_expert(L, xk, ssd_k[(size_t) k], vdst);
                else CK(cudaMemcpyAsync(vdst, L.hblob + (size_t) hsA * L.stride, L.L.bytes, cudaMemcpyHostToDevice, cs));
                if (hsV >= 0) {
                    if (!fast) {
                        CK(cudaStreamSynchronize(cs));   // swap_tmp holds V, and xk's copy from the host slot is done
                        if (hsA < 0 && L.hslot_exp[(size_t) hsV] >= 0) L.hexp_slot[(size_t) L.hslot_exp[(size_t) hsV]] = -1;   // dropped
                        std::memcpy(L.hblob + (size_t) hsV * L.stride, swap_tmp, L.L.bytes);
                    }
                    L.hslot_exp[(size_t) hsV] = V;
                    L.hexp_slot[(size_t) V] = hsV;
                    L.hslot_use[(size_t) hsV] = clock_;
                    ++demoted_;
                    if (hsA >= 0 && hsA != hsV) {   // xk left the host tier: its slot is free
                        L.hexp_slot[(size_t) xk] = -1;
                        L.hslot_exp[(size_t) hsA] = -1;
                    } else if (hsA >= 0) {
                        L.hexp_slot[(size_t) xk] = -1;   // (slot reused by V)
                    }
                } else if (hsA >= 0) {
                    L.hexp_slot[(size_t) xk] = -1;                  // xk left the host tier, nothing took its slot
                    L.hslot_exp[(size_t) hsA] = -1;
                }
                if (V >= 0) L.exp_slot[(size_t) V] = -1;
                L.slot_exp[(size_t) victim] = xk;
                L.exp_slot[(size_t) xk] = victim;
                L.slot_use[(size_t) victim] = clock_;
            }
            if (!ssd.empty() || !host_pinned) CK(cudaStreamSynchronize(cs));   // the pinned buffer is free again
        }
        if (prof) t_read += ms_since(tp1);
    }

    /// The lookahead: the experts of layer li that the previous layer's guess puts first, brought into its VRAM cache on
    /// the copy stream while the GPU is still busy with the work queued before this point.  The compute stream waits for
    /// the copies (an event) before anything queued from here on.
    void prefetch(const Layer& L, size_t li) {
        if (prefetch_k <= 0 || guess_layer != li || guess_next.empty() || L.S == 0) return;
        ++clock_;
        std::vector<int32_t> want;
        for (int r = 0; r < prefetch_k && r < (int) guess_next.size(); ++r) {
            const int32_t xk = guess_next[(size_t) r];
            if (L.exp_slot[(size_t) xk] >= 0) L.slot_use[(size_t) L.exp_slot[(size_t) xk]] = clock_;   // cached, and wanted: keep it
            else want.push_back(xk);
        }
        if (want.empty()) return;
        pf_loaded_ += (long) want.size();
        cs = st2;
        load_experts(L, want);
        cs = st;
        CK(cudaEventRecord(ev, st2));
        CK(cudaStreamWaitEvent(st, ev, 0));
    }

    /// The routed experts of a layer with an expert cache.  The router's choices come to the host; the experts of the
    /// chunk are run in rounds of at most S (the cache's slots): the cached ones first, then the others, each round's
    /// missing experts read from the file into the least recently used slots.  A decode step or a chunk of 8 tokens is
    /// one round.  Writes the unweighted expert outputs to eparts (row = token * k + j).
    /// The routed experts of a layer for a decode step or a few tokens (T <= 8, at most 48 entries): the same choices as moe_cached
    /// makes (hits in the VRAM cache, the others brought in by load_experts), with the host work cut to what it needs: the ids come
    /// back in a pinned array, nothing is allocated, the groups go to the card in one pinned block with one copy, and the kernel
    /// is queued at once (the stream orders the copy before it).
    bool moe_fast(const Layer& L, size_t li, int T) {
        const Model& M = *m;
        const int Kp = (int) M.TK, n_ent = T * Kp;
        const auto tp0 = Clock::now();
        CK(cudaMemcpyAsync(h_ids, ids, (size_t) n_ent * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        if (prof) t_sync += ms_since(tp0);
        if (trace_f && T == 1) {   // STRATA_FLINT_TRACE=<file>: layer (u8), then the Kp expert ids (u16) of every decode step's layer
            const uint8_t lb = (uint8_t) li;
            std::fwrite(&lb, 1, 1, trace_f);
            for (int e = 0; e < n_ent; ++e) {
                const uint16_t v = (uint16_t) h_ids[e];
                std::fwrite(&v, 2, 1, trace_f);
            }
        }
        // the entries sorted by (expert, entry): ord[i] = expert * 512 + entry
        int ord[kFastEnt];
        for (int e = 0; e < n_ent; ++e) ord[e] = h_ids[e] * 512 + e;
        std::sort(ord, ord + n_ent);
        ++clock_;
        for (const auto& pf_ : L.pfree)   // experts promoted to the VRAM at an earlier call: their host copy is released now
            if (L.hexp_slot[(size_t) pf_.first] == pf_.second && L.hslot_exp[(size_t) pf_.second] == pf_.first) {
                L.hexp_slot[(size_t) pf_.first] = -1;
                L.hslot_exp[(size_t) pf_.second] = -1;
            }
        L.pfree.clear();
        std::vector<int32_t> misses;   // not in the VRAM cache
        for (int i = 0; i < n_ent; ++i) {
            const int xk = ord[i] >> 9;
            if (i > 0 && (ord[i - 1] >> 9) == xk) continue;
            if (L.exp_slot[(size_t) xk] >= 0) L.slot_use[(size_t) L.exp_slot[(size_t) xk]] = clock_;
            else misses.push_back(xk);
        }
        if (prof) {
            ++n_calls;
            total_ += n_ent;
            for (int i = 0; i < n_ent; ++i)
                if (L.exp_slot[(size_t) (ord[i] >> 9)] >= 0) ++hits_;
        }
        const bool inplace = host_inplace && L.hdev != nullptr && (size_t) L.Sh >= misses.size();   // (all of the call's host experts at once)
        if (T > 8 && !inplace) return false;   // (a prompt chunk that wants more than the host tier holds: the rounds of moe_cached)
        if (inplace) {
            std::vector<int32_t> ssd;
            for (int32_t xk : misses) {
                if (L.hexp_slot[(size_t) xk] >= 0) {
                    L.hslot_use[(size_t) L.hexp_slot[(size_t) xk]] = clock_;
                    if (prof) ++host_hits_;
                } else {
                    ssd.push_back(xk);
                }
            }
            if (!ssd.empty()) {   // on neither level: from the file into a host slot, read in place from there
                const auto tp1 = Clock::now();
                for (size_t b0 = 0; b0 < ssd.size(); b0 += (size_t) pin_slots) {
                    const int n = (int) std::min<size_t>((size_t) pin_slots, ssd.size() - b0);
                    const auto ts0 = Clock::now();
                    read_parts(L, ssd.data() + b0, n);
                    t_ssdread += ms_since(ts0);
                    for (int k = 0; k < n; ++k) {
                        const int hs = host_lru(L, -1);
                        if (hs < 0) {
                            std::fprintf(stderr, "strata-flint: no free host slot (Sh=%d)\n", L.Sh);
                            std::exit(1);
                        }
                        const int32_t xk = ssd[b0 + (size_t) k];
                        if (L.hslot_exp[(size_t) hs] >= 0) L.hexp_slot[(size_t) L.hslot_exp[(size_t) hs]] = -1;
                        uint8_t* dst = L.hblob + (size_t) hs * L.stride;
                        std::memcpy(dst, part_ptr(L, xk, k, 0), L.src_gsz);
                        std::memcpy(dst + L.L.up_off, part_ptr(L, xk, k, 1), L.src_gsz);
                        std::memcpy(dst + L.L.down_off, part_ptr(L, xk, k, 2), L.src_dsz);
                        L.hslot_exp[(size_t) hs] = xk;
                        L.hexp_slot[(size_t) xk] = hs;
                        L.hslot_use[(size_t) hs] = clock_;
                        if (prof) ++ssd_reads_;
                    }
                }
                if (prof) t_read += ms_since(tp1);
            }
        } else if (!misses.empty()) {
            load_experts(L, misses);
        }
        unsigned long long* gp = (unsigned long long*) (h_meta + kMetaGP);
        int32_t* gs = (int32_t*) (h_meta + kMetaGS);
        int32_t* ed = (int32_t*) (h_meta + kMetaED);
        int32_t* et = (int32_t*) (h_meta + kMetaET);
        int ng = 0, ne_ = 0;
        for (int i = 0; i < n_ent;) {
            const int xk = ord[i] >> 9;
            int j = i;
            while (j < n_ent && (ord[j] >> 9) == xk) ++j;
            const unsigned long long blob = L.exp_slot[(size_t) xk] >= 0
                                                ? (unsigned long long) (L.cblob + (size_t) L.exp_slot[(size_t) xk] * L.stride)
                                                : (unsigned long long) (L.hdev + (size_t) L.hexp_slot[(size_t) xk] * L.stride);
            for (int a = i; a < j; a += (int) kGroupEntries) {   // (a group takes at most kGroupEntries activations)
                gp[ng] = blob;
                gs[ng] = ne_;
                ++ng;
                for (int b = a; b < std::min(j, a + (int) kGroupEntries); ++b) {
                    ed[ne_] = ord[b] & 511;
                    et[ne_] = (ord[b] & 511) / Kp;
                    ++ne_;
                }
            }
            i = j;
        }
        gs[ng] = ne_;
        *(int32_t*) (h_meta + kMetaNG) = ng;
        CK(cudaMemcpyAsync(d_meta, h_meta, kMetaBytes, cudaMemcpyHostToDevice, st));
        K::native_expert_grouped(L.L, (const unsigned long long*) (d_meta + kMetaGP), (const int32_t*) (d_meta + kMetaGS),
                                 (const int32_t*) (d_meta + kMetaNG), (const int32_t*) (d_meta + kMetaED),
                                 (const int32_t*) (d_meta + kMetaET), (int64_t) ng, (int64_t) ne_, xq, escr, eparts, st);
        if (inplace) {   // the experts used again and again outside the VRAM cache move in, in the background
            CK(cudaEventRecord(ev_kd, st));
            for (int32_t xk : misses) {
                if (L.exp_slot[(size_t) xk] >= 0) continue;
                if (L.hot[(size_t) xk] < 255) ++L.hot[(size_t) xk];
                if (L.hot[(size_t) xk] >= promote_after) promote(L, xk);
            }
        }
        (void) li;
        return true;
    }

    /// A slot of the host tier to give away: an empty one, else the least recently used that is not in use (this call's clock
    /// nor the slot `keep`); -1 when none.
    int host_lru(const Layer& L, int keep) const {
        int best = -1;
        for (int c = 0; c < L.Sh; ++c) {
            if (c == keep || L.hslot_use[(size_t) c] == clock_) continue;
            if (L.hslot_exp[(size_t) c] < 0) return c;
            if (best < 0 || L.hslot_use[(size_t) c] < L.hslot_use[(size_t) best]) best = c;
        }
        return best;
    }

    /// Expert xk (now in the host tier) moves to the VRAM cache on the promotion stream: the least recently used slot of the
    /// cache that this call does not use gives its expert down to the host tier (a copy on the card to the host) and takes
    /// xk's bytes (a copy from the host).  The kernel of this call, queued before, reads xk where it is; its host copy is
    /// released at the next call of the layer.  The next step waits for the copies (ev_pro).
    void promote(const Layer& L, int xk) {
        const int hs = L.hexp_slot[(size_t) xk];
        if (hs < 0) return;
        int v = -1;
        long best = 0;
        for (int c = 0; c < L.S; ++c) {
            if (L.slot_use[(size_t) c] == clock_) continue;
            if (v < 0 || L.slot_use[(size_t) c] < best) {
                v = c;
                best = L.slot_use[(size_t) c];
                if (L.slot_exp[(size_t) c] < 0) break;
            }
        }
        if (v < 0) return;
        const int V = L.slot_exp[(size_t) v];
        uint8_t* vdst = L.cblob + (size_t) v * L.stride;
        if (V >= 0) {
            // x (host slot hs) and V (VRAM slot v) swap places, so the host tier keeps as many experts as it had: V goes through a
            // VRAM scratch, x goes into the slot, V takes x's host slot once the expert kernel of this call has finished reading
            // x there (ev_kd).  All on the promotion stream; the next step waits for it.
            CK(cudaStreamWaitEvent(st_pro, ev_kd, 0));
            CK(cudaMemcpyAsync(tmp_blobs, vdst, L.L.bytes, cudaMemcpyDeviceToDevice, st_pro));
            CK(cudaMemcpyAsync(vdst, L.hblob + (size_t) hs * L.stride, L.L.bytes, cudaMemcpyHostToDevice, st_pro));
            CK(cudaMemcpyAsync(L.hblob + (size_t) hs * L.stride, tmp_blobs, L.L.bytes, cudaMemcpyDeviceToHost, st_pro));
            L.exp_slot[(size_t) V] = -1;
            L.hexp_slot[(size_t) xk] = -1;
            L.hslot_exp[(size_t) hs] = V;
            L.hexp_slot[(size_t) V] = hs;
            L.hslot_use[(size_t) hs] = clock_;
            L.slot_exp[(size_t) v] = xk;
            L.exp_slot[(size_t) xk] = v;
            L.slot_use[(size_t) v] = clock_;
            L.hot[(size_t) xk] = 0;
            pro_pending = true;
            ++demoted_;
            return;
        }
        CK(cudaMemcpyAsync(vdst, L.hblob + (size_t) hs * L.stride, L.L.bytes, cudaMemcpyHostToDevice, st_pro));
        L.slot_exp[(size_t) v] = xk;
        L.exp_slot[(size_t) xk] = v;
        L.slot_use[(size_t) v] = clock_;
        L.hot[(size_t) xk] = 0;
        L.pfree.emplace_back(xk, hs);
        pro_pending = true;
        ++demoted_;
    }

    /// A decode step (or a few tokens) with the host tier read in place: an expert is read by the GPU from the VRAM cache or from
    /// the mapped host memory, wherever it is; one that is on neither is read from the file into a host slot first.  Nothing is
    /// copied to the VRAM and nothing is given back: the experts stay where they were put.
    void moe_direct(const Layer& L, size_t li, int T, const std::vector<std::vector<int32_t>>& entries,
                    const std::vector<int32_t>& uniq) {
        const Model& M = *m;
        const int Kp = (int) M.TK;
        ++clock_;
        std::vector<int32_t> ssd;
        for (int32_t xk : uniq) {
            if (L.exp_slot[(size_t) xk] >= 0) L.slot_use[(size_t) L.exp_slot[(size_t) xk]] = clock_;
            else if (L.hexp_slot[(size_t) xk] >= 0) L.hslot_use[(size_t) L.hexp_slot[(size_t) xk]] = clock_;
            else ssd.push_back(xk);
        }
        if (!ssd.empty()) {
            const auto tp1 = Clock::now();
            for (size_t b0 = 0; b0 < ssd.size(); b0 += (size_t) pin_slots) {
                const int n = (int) std::min<size_t>((size_t) pin_slots, ssd.size() - b0);
                read_parts(L, ssd.data() + b0, n);
                for (int k = 0; k < n; ++k) {
                    int hs = -1;   // an empty host slot, else the least recently used one that this call does not use
                    long best = 0;
                    for (int c = 0; c < L.Sh; ++c) {
                        if (L.hslot_use[(size_t) c] == clock_) continue;
                        if (hs < 0 || L.hslot_use[(size_t) c] < best) {
                            hs = c;
                            best = L.hslot_use[(size_t) c];
                            if (L.hslot_exp[(size_t) c] < 0) break;
                        }
                    }
                    if (hs < 0) {
                        std::fprintf(stderr, "strata-flint: no free host slot (Sh=%d)\n", L.Sh);
                        std::exit(1);
                    }
                    const int32_t xk = ssd[b0 + (size_t) k];
                    if (L.hslot_exp[(size_t) hs] >= 0) L.hexp_slot[(size_t) L.hslot_exp[(size_t) hs]] = -1;
                    uint8_t* dst = L.hblob + (size_t) hs * L.stride;
                    std::memcpy(dst, part_ptr(L, xk, k, 0), L.src_gsz);
                    std::memcpy(dst + L.L.up_off, part_ptr(L, xk, k, 1), L.src_gsz);
                    std::memcpy(dst + L.L.down_off, part_ptr(L, xk, k, 2), L.src_dsz);
                    L.hslot_exp[(size_t) hs] = xk;
                    L.hexp_slot[(size_t) xk] = hs;
                    L.hslot_use[(size_t) hs] = clock_;
                }
            }
            if (prof) t_read += ms_since(tp1);
        }
        std::vector<unsigned long long> gp;
        std::vector<int32_t> gs, ed, et;
        for (int32_t xk : uniq) {
            const unsigned long long blob = L.exp_slot[(size_t) xk] >= 0
                                                ? (unsigned long long) (L.cblob + (size_t) L.exp_slot[(size_t) xk] * L.stride)
                                                : (unsigned long long) (L.hdev + (size_t) L.hexp_slot[(size_t) xk] * L.stride);
            const std::vector<int32_t>& en = entries[(size_t) xk];
            for (size_t e0 = 0; e0 < en.size(); e0 += kGroupEntries) {
                gp.push_back(blob);
                gs.push_back((int32_t) ed.size());
                for (size_t e1 = e0; e1 < std::min(en.size(), e0 + kGroupEntries); ++e1) {
                    ed.push_back(en[e1]);
                    et.push_back(en[e1] / Kp);
                }
            }
        }
        gs.push_back((int32_t) ed.size());
        const int32_t ng = (int32_t) gp.size();
        CK(cudaMemcpyAsync(grp_ptr, gp.data(), gp.size() * 8, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(grp_start, gs.data(), gs.size() * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(n_groups, &ng, 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(ent_dst, ed.data(), ed.size() * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(ent_tok, et.data(), et.size() * 4, cudaMemcpyHostToDevice, st));
        CK(cudaStreamSynchronize(st));   // the host vectors are free again
        K::native_expert_grouped(L.L, grp_ptr, grp_start, n_groups, ent_dst, ent_tok, (int64_t) gp.size(), (int64_t) ed.size(), xq, escr,
                                 eparts, st);
        (void) li;
        (void) T;
    }

    void moe_cached(const Layer& L, size_t li, int T) {
        const Model& M = *m;
        const int Kp = (int) M.TK, n_ent = T * Kp;
        std::vector<int32_t> hid((size_t) n_ent);
        if (T == 1) prefetch(L, li);   // (the guess for this layer was made by the previous one)
        const auto tp0 = Clock::now();
        CK(cudaMemcpyAsync(hid.data(), ids, (size_t) n_ent * 4, cudaMemcpyDeviceToHost, st));
        std::vector<float> hplog;
        const bool guessing = (prof_predict || prefetch_k > 0) && T == 1 && li + 1 < M.layers.size() && !M.layers[li + 1].dense;
        if (guessing) {
            hplog.resize((size_t) M.NE);
            CK(cudaMemcpyAsync(hplog.data(), plog, (size_t) M.NE * 4, cudaMemcpyDeviceToHost, st));
        }
        CK(cudaStreamSynchronize(st));
        if (prof) t_sync += ms_since(tp0);
        std::vector<std::vector<int32_t>> entries((size_t) M.NE);   // expert -> its entries (token * k + j), ascending
        for (int e = 0; e < n_ent; ++e) entries[(size_t) hid[(size_t) e]].push_back(e);
        std::vector<int32_t> uniq;
        for (int xk = 0; xk < (int) M.NE; ++xk)
            if (!entries[(size_t) xk].empty()) uniq.push_back(xk);
        const bool tmp = L.S == 0;   // no cache yet (the warmup): every chosen expert is read into a temporary buffer
        std::vector<int32_t> hits, misses;
        for (int32_t xk : uniq) (!tmp && L.exp_slot[(size_t) xk] >= 0 ? hits : misses).push_back(xk);
        if (prof) {
            ++n_calls;
            for (int32_t xk : hits) hits_ += (long) entries[(size_t) xk].size();
            total_ += n_ent;
        }
        if (host_direct && !tmp && L.Sh >= (int) uniq.size() && T <= 64) {
            if (prof) {   // (hits are the VRAM cache's; the rest are read in place from the host tier or from the file)
                for (int32_t xk : misses) {
                    if (L.hexp_slot[(size_t) xk] >= 0) ++host_hits_;
                    else ++ssd_reads_;
                }
            }
            moe_direct(L, li, T, entries, uniq);
            return;
        }
        if ((prof_predict || prefetch_k > 0) && T == 1) {
            if (prof_predict && guess_layer == li && !guess_next.empty()) {
                auto rank = [&](int32_t xk) { return (int) (std::find(guess_next.begin(), guess_next.end(), xk) - guess_next.begin()); };
                for (int32_t xk : uniq) {
                    const int r = rank(xk);
                    ++pr_total;
                    pr_in6 += r < 6;
                    pr_in12 += r < 12;
                    pr_in16 += r < 16;
                }
                for (int32_t xk : misses) {
                    const int r = rank(xk);
                    ++pr_miss;
                    pr_miss_in12 += r < 12;
                    pr_miss_in16 += r < 16;
                }
            }
            guess_next.clear();
            guess_layer = (size_t) -1;
            if (guessing) {
                std::vector<int32_t> order((size_t) M.NE);
                for (int i = 0; i < (int) M.NE; ++i) order[(size_t) i] = i;
                std::partial_sort(order.begin(), order.begin() + 16, order.end(), [&](int32_t a, int32_t b) { return hplog[(size_t) a] > hplog[(size_t) b]; });
                guess_next.assign(order.begin(), order.begin() + 16);
                guess_layer = li + 1;
            }
        }
        size_t next_miss = 0;
        bool first = true;
        while (first || next_miss < misses.size()) {
            ++clock_;   // a new round: the slots of the previous round may be given away
            std::vector<int32_t> round;
            if (first) round = hits;
            for (int32_t xk : round) L.slot_use[(size_t) L.exp_slot[(size_t) xk]] = clock_;
            std::vector<int32_t> load;
            const size_t room = tmp ? (size_t) pin_slots : (size_t) L.S - round.size();
            while (next_miss < misses.size() && load.size() < room) load.push_back(misses[next_miss++]);
            first = false;
            if (tmp) {
                const auto tp1 = Clock::now();
                read_parts(L, load.data(), (int) load.size());
                for (size_t k = 0; k < load.size(); ++k) upload_expert(L, load[k], (int) k, tmp_blobs + k * tmp_stride);
                CK(cudaStreamSynchronize(st));
                if (prof) t_read += ms_since(tp1);
            } else {
                load_experts(L, load);
            }
            for (int32_t xk : load) round.push_back(xk);
            std::sort(round.begin(), round.end());
            if (round.empty()) continue;
            std::vector<unsigned long long> gp;
            std::vector<int32_t> gs, ed, et;
            for (size_t r = 0; r < round.size(); ++r) {
                const int32_t xk = round[r];
                const unsigned long long blob =
                    tmp ? (unsigned long long) (tmp_blobs + (size_t) (std::find(load.begin(), load.end(), xk) - load.begin()) * tmp_stride)
                        : (unsigned long long) (L.cblob + (size_t) L.exp_slot[(size_t) xk] * L.stride);
                const std::vector<int32_t>& en = entries[(size_t) xk];
                // a group takes at most kGroupEntries activations: an expert with more gets several groups, one blob
                for (size_t e0 = 0; e0 < en.size(); e0 += kGroupEntries) {
                    gp.push_back(blob);
                    gs.push_back((int32_t) ed.size());
                    for (size_t e1 = e0; e1 < std::min(en.size(), e0 + kGroupEntries); ++e1) {
                        ed.push_back(en[e1]);
                        et.push_back(en[e1] / Kp);
                    }
                }
            }
            gs.push_back((int32_t) ed.size());
            const int32_t ng = (int32_t) gp.size();
            CK(cudaMemcpyAsync(grp_ptr, gp.data(), gp.size() * 8, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(grp_start, gs.data(), gs.size() * 4, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(n_groups, &ng, 4, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(ent_dst, ed.data(), ed.size() * 4, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(ent_tok, et.data(), et.size() * 4, cudaMemcpyHostToDevice, st));
            CK(cudaStreamSynchronize(st));   // the host vectors are free again
            K::native_expert_grouped(L.L, grp_ptr, grp_start, n_groups, ent_dst, ent_tok, (int64_t) gp.size(),
                                     (int64_t) ed.size(), xq, escr, eparts, st);
        }
        (void) li;
    }

    // ================================================================================================ the long-prompt path
    template <class T_> T_* dmalloc(size_t n) {
        T_* p_ = nullptr;
        CK(cudaMalloc((void**) &p_, n * sizeof(T_) + 64));
        CK(cudaMemset(p_, 0, n * sizeof(T_) + 64));
        return p_;
    }

    /// Bytes between two experts in a group buffer for layer L: pass 0 holds gate+up, pass 1 the down.  MMQ steps over the
    /// experts by whole blocks of the weights' type, so the step is a multiple of the block size (and of 16).
    static size_t pf_stride(const Layer& L, int pass) {
        const int ty = pass == 0 ? L.L.gu_type : L.L.d_type;
        const size_t bytes = pass == 0 ? L.L.down_off : L.L.bytes - L.L.down_off;
        return round_up(bytes, lcm_sz(block_bytes(ty), 16));
    }

    /// Sets the long-prompt path up (after init_staging): the matrix-core GEMMs, MMQ for the experts, the buffers, the loader.
    /// Left off (pf2 false) when this build has no MMQ or a type of the file is not covered: the prompt then goes through the
    /// 64-token path, in chunks of at most 512 tokens.
    void pf_init() {
        const Model& M = *m;
        const int64_t N = M.N, FF = M.FFE;
        const size_t T = (size_t) tmax, Kp = (size_t) M.TK;
        if (std::getenv("STRATA_FLINT_NO_PF2") || tmax <= kPf2Min || M.NE % kPfGroup != 0) return;
        bool ok = MMQ::built();
        if (!ok) std::fprintf(stderr, "strata-flint: no MMQ in this build: long prompts go through the 64-token path\n");
        for (const Layer& L : M.layers)
            if (ok && !L.dense && !(MMQ::supported(L.L.gu_type) && MMQ::supported(L.L.d_type) && MMQ::fits(L.L.gu_type, 2 * FF) &&
                                    MMQ::fits(L.L.d_type, N))) {
                std::fprintf(stderr, "strata-flint: MMQ does not cover an expert type of this file (gate/up %s: supported %d fits %d; "
                             "down %s: supported %d fits %d): long prompts go through the 64-token path\n",
                             strata::ggml_type_name(L.L.gu_type), (int) MMQ::supported(L.L.gu_type), (int) MMQ::fits(L.L.gu_type, 2 * FF),
                             strata::ggml_type_name(L.L.d_type), (int) MMQ::supported(L.L.d_type), (int) MMQ::fits(L.L.d_type, N));
                ok = false;
            }
        if (!ok) return;
        int64_t big = 0;
        for (const Layer& L : M.layers)
            for (const Mat* w : {&L.wq, &L.wk, &L.wv, &L.wo, &L.ffn_gate, &L.ffn_up, &L.ffn_down, &L.sh_gate, &L.sh_up, &L.sh_down})
                big = std::max<int64_t>(big, (int64_t) w->n_in * w->n_out);
        std::string gerr;
        if (!gemm.init(st, big, gerr)) {
            std::fprintf(stderr, "strata-flint: prompt GEMM: %s\n", gerr.c_str());
            return;
        }
        if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS || cublasSetStream(blas, st) != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "strata-flint: cuBLAS handle failed\n");
            return;
        }
        const int64_t ne = (int64_t) (T * Kp);
        x16 = dmalloc<__half>(T * (size_t) std::max<int64_t>({M.FF, M.H * kHD, N}));
        q16 = dmalloc<__half>((size_t) 256 * M.H * kHD);
        const size_t kvw = (size_t) M.HKV * (256 + M.window + 16) * kHD;
        k16w = dmalloc<__half>(kvw);
        v16w = dmalloc<__half>(kvw);
        att_s_floats = (size_t) 24 << 20;   // 96 MiB of scores, as many again of probabilities
        att_s = dmalloc<float>(att_s_floats);
        att_p16 = dmalloc<__half>(att_s_floats);
        mctx = std::make_unique<MMQ::Context>();
        rcnt = dmalloc<int32_t>((size_t) M.NE);
        rcur = dmalloc<int32_t>((size_t) M.NE);
        rbounds = dmalloc<int32_t>((size_t) M.NE + 1);
        ident = dmalloc<int32_t>((size_t) ne);
        rinv = dmalloc<int32_t>((size_t) ne);
        rsc_e = dmalloc<float>((size_t) ne);
        rsc_s = dmalloc<float>(T);
        CK(cudaDeviceSynchronize());
        MMQ::iota(ident, ne, st);
        CK(cudaMalloc(&mq_x, MMQ::q8_bytes(ne, N)));
        CK(cudaMalloc(&mq_hq, MMQ::q8_bytes(ne, FF)));
        mq_gu = dmalloc<float>((size_t) ne * 2 * FF);
        mq_h = dmalloc<float>((size_t) ne * FF);
        // the loader: group buffers (gate+up of 16 experts, or their down), the stream the copies go on, the events
        size_t s1 = 0, s2 = 0;
        for (const Layer& L : M.layers)
            if (!L.dense) {
                s1 = std::max(s1, L.L.down_off);
                s2 = std::max(s2, L.L.bytes - L.L.down_off);
            }
        size_t gmax = 0;
        for (const Layer& L : M.layers)
            if (!L.dense) gmax = std::max({gmax, pf_stride(L, 0), pf_stride(L, 1)});
        (void) s1;
        (void) s2;
        const size_t gbytes = (size_t) kPfGroup * gmax + 8192;
        for (int b = 0; b < kPfNB; ++b) {
            pf_gbuf[b] = dmalloc<uint8_t>(gbytes);
            CK(cudaEventCreateWithFlags(&pf_ready[b], cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&pf_free[b], cudaEventDisableTiming));
        }
        CK(cudaStreamCreateWithFlags(&pf_sio, cudaStreamNonBlocking));
        {
            size_t gs = 0, ds = 0;
            for (const Layer& L : M.layers)
                if (!L.dense) {
                    gs = std::max(gs, L.src_gsz);
                    ds = std::max(ds, L.src_dsz);
                }
            pf_capg = round_up(gs + 4096, 4096);
            pf_capd = round_up(ds + 4096, 4096);
            pf_perexp = 2 * pf_capg + pf_capd;
            for (int i = 0; i < kPfNL; ++i) CK(cudaEventCreateWithFlags(&pf_lfree[i], cudaEventDisableTiming));
        }
        pf_pin[0] = pin;
        CK(cudaHostAlloc((void**) &pf_pin[1], pin_cap * 3 * (size_t) pin_slots, cudaHostAllocDefault));
        for (int i = 0; i < 2; ++i) CK(cudaEventCreateWithFlags(&pf_pin_ev[i], cudaEventDisableTiming));
        if (finite_check) {
            const Layer& L2 = M.layers[2];
            std::fprintf(stderr, "layout of layer 2: gu_type %d d_type %d gu_row %zu d_row %zu up_off %zu down_off %zu bytes %zu | src_gsz %zu src_dsz %zu stride %zu | S1 %zu S2 %zu\n",
                         L2.L.gu_type, L2.L.d_type, L2.L.gu_row, L2.L.d_row, L2.L.up_off, L2.L.down_off, L2.L.bytes, L2.src_gsz,
                         L2.src_dsz, L2.stride, pf_stride(L2, 0), pf_stride(L2, 1));
        }
        pf2 = true;
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        std::fprintf(stderr, "strata-flint: long-prompt path on: chunks of %d tokens, GEMMs on the matrix cores, experts on MMQ "
                     "(int8); %.2f GiB of VRAM free\n", tmax, (double) fr / (1 << 30));
    }

    // the file reader of a chunk: layer by layer, a few layers ahead of the loader, the experts of a layer that neither cache holds
    static constexpr int kPfNL = 3;           // layer buffers in the ring
    bool pf_use_reader = false;
    size_t pf_capg = 0, pf_capd = 0, pf_perexp = 0;
    size_t pf_lcap = 0;                       // experts one layer buffer holds
    uint8_t* pf_lbuf[kPfNL] = {};
    cudaEvent_t pf_lfree[kPfNL] = {};
    std::vector<std::vector<int>> pf_idx;     // [layer][expert]: its index in the layer's buffer, -1 if it is not read by the reader
    std::vector<std::vector<int32_t>> pf_need;
    std::vector<int> pf_ord;                  // [layer]: its ordinal among the MoE layers
    size_t pf_lr_ready = 0, pf_layers_done = 0;
    std::thread pf_rd;
    long pf_filled_v = 0, pf_filled_h = 0;
    double pf_t_rd = 0, pf_t_prod = 0, pf_t_ssd = 0, pf_t_waitfree = 0, pf_t_mainwait = 0;   // --prof: the loader's wall time, its file reads, its waits; the compute's waits on it
    long pf_n_ssd = 0, pf_n_vram = 0, pf_n_host = 0;

    void pf_wait_produced(size_t seq) {
        const auto t0 = Clock::now();
        std::unique_lock<std::mutex> lk(pf_mu);
        pf_cv.wait(lk, [&] { return pf_produced > seq; });
        pf_t_mainwait += ms_since(t0);
    }
    void pf_mark_consumed(size_t n) {
        {
            std::lock_guard<std::mutex> lk(pf_mu);
            pf_consumed = n;
        }
        pf_cv.notify_all();
    }

    /// The loader thread of a chunk: for each group of experts of each MoE layer, in the order the compute will ask for them
    /// (gate+up of the layer's groups, then their down), the group's experts into a free buffer of the ring: from the
    /// VRAM cache (a copy on the card), from the host tier (DMA), or read from the file.
    void pf_produce() {
        const Model& M = *m;
        const int ngroups = (int) M.NE / kPfGroup;
        const auto tprod0 = Clock::now();
        for (size_t seq = 0; seq < pf_tasks.size(); ++seq) {
            const PfTask tk = pf_tasks[seq];
            const Layer& L = M.layers[(size_t) tk.li];
            const int b = (int) (seq % kPfNB);
            if (seq >= (size_t) kPfNB) {   // the buffer's last user: queued by the compute, and finished
                const auto tw0 = Clock::now();
                std::unique_lock<std::mutex> lk(pf_mu);
                pf_cv.wait(lk, [&] { return pf_consumed + kPfNB > seq; });
                lk.unlock();
                CK(cudaEventSynchronize(pf_free[b]));
                pf_t_waitfree += ms_since(tw0);
            }
            const size_t S = pf_stride(L, tk.pass);
            const size_t off = tk.pass == 0 ? 0 : L.L.down_off;
            const size_t len = tk.pass == 0 ? L.L.down_off : L.L.bytes - L.L.down_off;
            const size_t ord = (size_t) pf_ord[(size_t) tk.li];
            uint8_t* base = pf_gbuf[b];
            struct Ssd { int e, k; const uint8_t* p[3]; };
            std::vector<Ssd> ssd;
            for (int k = 0; k < kPfGroup && !pf_dry; ++k) {
                const int e = tk.group * kPfGroup + k;
                uint8_t* dst = base + (size_t) k * S;
                if (resident[(size_t) tk.li]) {
                    ++pf_n_vram;
                    CK(cudaMemcpyAsync(dst, L.blobs + (size_t) e * L.stride + off, len, cudaMemcpyDeviceToDevice, pf_sio));
                } else if (L.S > 0 && L.exp_slot[(size_t) e] >= 0) {
                    ++pf_n_vram;
                    CK(cudaMemcpyAsync(dst, L.cblob + (size_t) L.exp_slot[(size_t) e] * L.stride + off, len,
                                       cudaMemcpyDeviceToDevice, pf_sio));
                } else if (L.Sh > 0 && L.hexp_slot[(size_t) e] >= 0) {
                    ++pf_n_host;
                    CK(cudaMemcpyAsync(dst, L.hblob + (size_t) L.hexp_slot[(size_t) e] * L.stride + off, len,
                                       cudaMemcpyHostToDevice, pf_sio));
                } else {
                    Ssd x;
                    x.e = e;
                    x.k = k;
                    x.p[0] = x.p[1] = x.p[2] = nullptr;
                    ssd.push_back(x);
                }
            }
            if (!ssd.empty()) {
                int set = -1;
                if (pf_use_reader) {   // the reader has this layer's experts in a pinned buffer by now
                    {
                        const auto tw0 = Clock::now();
                        std::unique_lock<std::mutex> lk(pf_mu);
                        pf_cv.wait(lk, [&] { return pf_lr_ready > ord; });
                        pf_t_ssd += ms_since(tw0);   // (here: the time the loader waited for the reader)
                    }
                    const uint8_t* lb = pf_lbuf[ord % kPfNL];
                    for (Ssd& x : ssd) {
                        const int j = pf_idx[(size_t) tk.li][(size_t) x.e];
                        for (int p = 0; p < 3; ++p)
                            x.p[p] = lb + (size_t) j * pf_perexp + (size_t) (p == 2 ? 2 * pf_capg : p * pf_capg) + part_shift(L, (uint64_t) x.e, p);
                    }
                } else {               // no reader (too many experts to hold): this task reads its own
                    set = (int) (seq & 1);
                    CK(cudaEventSynchronize(pf_pin_ev[set]));
                    std::vector<int32_t> ex;
                    for (const Ssd& x : ssd) ex.push_back(x.e);
                    const auto ts0 = Clock::now();
                    read_parts(L, ex.data(), (int) ex.size(), tk.pass == 0 ? 3 : 4, pf_pin[set]);
                    pf_t_ssd += ms_since(ts0);
                    for (size_t q = 0; q < ssd.size(); ++q)
                        for (int p = 0; p < 3; ++p) ssd[q].p[p] = part_ptr(L, ssd[q].e, (int) q, p, pf_pin[set]);
                }
                pf_n_ssd += (long) ssd.size();
                for (const Ssd& x : ssd) {
                    uint8_t* dst = base + (size_t) x.k * S;
                    if (tk.pass == 0) {
                        CK(cudaMemcpyAsync(dst, x.p[0], L.src_gsz, cudaMemcpyHostToDevice, pf_sio));
                        CK(cudaMemcpyAsync(dst + L.L.up_off, x.p[1], L.src_gsz, cudaMemcpyHostToDevice, pf_sio));
                    } else {
                        CK(cudaMemcpyAsync(dst, x.p[2], L.src_dsz, cudaMemcpyHostToDevice, pf_sio));
                    }
                }
                if (set >= 0) CK(cudaEventRecord(pf_pin_ev[set], pf_sio));
            }
            if (pf_use_reader && tk.pass == 1 && tk.group == ngroups - 1) {   // the layer is done: its buffer may be read over
                CK(cudaEventRecord(pf_lfree[ord % kPfNL], pf_sio));
                {
                    std::lock_guard<std::mutex> lk(pf_mu);
                    pf_layers_done = ord + 1;
                }
                pf_cv.notify_all();
            }
            CK(cudaEventRecord(pf_ready[b], pf_sio));
            {
                std::lock_guard<std::mutex> lk(pf_mu);
                pf_produced = seq + 1;
            }
            pf_cv.notify_all();
        }
        pf_t_prod += ms_since(tprod0);
    }

    /// The reader of a chunk: layer by layer, up to kPfNL layers ahead of the loader, the experts of the layer that neither
    /// cache holds, read from the file (15 reads at once) into the layer's pinned buffer.
    void pf_read_layers() {
        const Model& M = *m;
        size_t ord = 0;
        for (size_t li = 0; li < M.layers.size(); ++li) {
            const Layer& L = M.layers[li];
            if (L.dense) continue;
            const int slot = (int) (ord % kPfNL);
            if (ord >= (size_t) kPfNL) {
                std::unique_lock<std::mutex> lk(pf_mu);
                pf_cv.wait(lk, [&] { return pf_layers_done + kPfNL > ord; });
                lk.unlock();
                CK(cudaEventSynchronize(pf_lfree[slot]));   // the loader's copies out of this buffer are done
            }
            const std::vector<int32_t>& need = pf_need[li];
            uint8_t* buf = pf_lbuf[slot];
            if (!pf_dry && !need.empty()) {
                const auto ts0 = Clock::now();
                std::lock_guard<std::mutex> lk(pool_mu);
                pool->run((int) need.size() * 3, [&](int j) {
                    const int k = j / 3, p = j % 3;
                    read_one_part(L, (uint64_t) need[(size_t) k], p,
                                  buf + (size_t) k * pf_perexp + (size_t) (p == 2 ? 2 * pf_capg : p * pf_capg));
                });
                pf_t_rd += ms_since(ts0);
            }
            {
                std::lock_guard<std::mutex> lk(pf_mu);
                pf_lr_ready = ord + 1;
            }
            pf_cv.notify_all();
            ++ord;
        }
    }
    void pf_begin() {
        pf_tasks.clear();
        for (size_t li = 0; li < m->layers.size(); ++li) {
            if (m->layers[li].dense) continue;
            for (int pass = 0; pass < 2; ++pass)
                for (int g = 0; g < (int) m->NE / kPfGroup; ++g) pf_tasks.push_back({(int) li, pass, g});
        }
        pf_produced = pf_consumed = pf_next = 0;
        pf_lr_ready = pf_layers_done = 0;
        // which experts of each layer neither cache holds, and whether the reader can hold them
        const Model& M = *m;
        pf_need.assign(M.layers.size(), {});
        pf_idx.assign(M.layers.size(), {});
        pf_ord.assign(M.layers.size(), -1);
        size_t max_need = 0;
        int ord = 0;
        for (size_t li = 0; li < M.layers.size(); ++li) {
            const Layer& L = M.layers[li];
            if (L.dense) continue;
            pf_ord[li] = ord++;
            pf_idx[li].assign((size_t) M.NE, -1);
            if (resident[li]) continue;
            for (int e = 0; e < (int) M.NE; ++e)
                if (!((L.S > 0 && L.exp_slot[(size_t) e] >= 0) || (L.Sh > 0 && L.hexp_slot[(size_t) e] >= 0))) {
                    pf_idx[li][(size_t) e] = (int) pf_need[li].size();
                    pf_need[li].push_back(e);
                }
            max_need = std::max(max_need, pf_need[li].size());
        }
        pf_use_reader = max_need <= 112;
        if (pf_use_reader && pf_lcap < max_need) {   // (grown when a chunk needs more; pinned memory kept for the next ones)
            for (int i = 0; i < kPfNL; ++i)
                if (pf_lbuf[i]) CK(cudaFreeHost(pf_lbuf[i]));
            pf_lcap = std::max<size_t>(max_need, 8);
            for (int i = 0; i < kPfNL; ++i) CK(cudaHostAlloc((void**) &pf_lbuf[i], pf_lcap * pf_perexp + 4096, cudaHostAllocDefault));
        }
        pf_th = std::thread([this] { pf_produce(); });
        if (pf_use_reader) pf_rd = std::thread([this] { pf_read_layers(); });
    }
    void pf_end() {
        if (pf_th.joinable()) pf_th.join();
        if (pf_rd.joinable()) pf_rd.join();
        CK(cudaStreamSynchronize(pf_sio));   // the copies the loader queued are done
    }

    /// STRATA_FLINT_CHECK_FINITE: says how many of the n floats at p are not finite.
    void chk(const char* what, const float* p, int64_t n) {
        if (!finite_check) return;
        if (!d_cnt) CK(cudaMalloc(&d_cnt, 4));
        CK(cudaMemsetAsync(d_cnt, 0, 4, st));
        k_nonfinite<<<grid1(n), kThreads, 0, st>>>(p, n, d_cnt);
        int hc = 0;
        CK(cudaMemcpyAsync(&hc, d_cnt, 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        if (hc) std::fprintf(stderr, "finite check: %s: %d of %lld values are not finite\n", what, hc, (long long) n);
    }

    /// Dense projection through the matrix cores: out [T][w.n_out] = x16 [T][w.n_in] . W^T.
    void gm(const Mat& w, float* out, int T) {
        gemm.native((const uint16_t*) x16, w.type, w.d, out, T, w.n_out, w.n_in);
    }

#define KV_DISPATCH(CALL)                       \
    if (kv_mode == 1) {                         \
        constexpr int MODE = 1;                 \
        if (kv_bits == 2) {                     \
            constexpr int KB = 2;               \
            CALL;                               \
        } else if (kv_bits == 3) {              \
            constexpr int KB = 3;               \
            CALL;                               \
        } else {                                \
            constexpr int KB = 4;               \
            CALL;                               \
        }                                       \
    } else {                                    \
        constexpr int MODE = 2, KB = 4;         \
        CALL;                                   \
    }

    /// The K and V that k_prep left in the staging (T tokens at p0 ..) of a quantised layer, encoded into its cache.
    void kv_encode(size_t li, int T, int p0) {
        const Model& M = *m;
        if (T > kKvStageT) {
            std::fprintf(stderr, "strata-flint: %d tokens in one attention call exceed the %d of the KV staging\n", T, kKvStageT);
            std::exit(1);
        }
        const dim3 g((unsigned) M.HKV, (unsigned) T, 2);
        if (kv_mode == 1) {
            if (kv_bits == 2)
                k_kvq_enc_tqt<2><<<g, 256, 0, st>>>(kv_kst, kv_vst, kq_code[li], vq_code[li], kq_sc[li], vq_sc[li], (int) M.HKV, (int) li, p0, ncell[li]);
            else if (kv_bits == 3)
                k_kvq_enc_tqt<3><<<g, 256, 0, st>>>(kv_kst, kv_vst, kq_code[li], vq_code[li], kq_sc[li], vq_sc[li], (int) M.HKV, (int) li, p0, ncell[li]);
            else
                k_kvq_enc_tqt<4><<<g, 256, 0, st>>>(kv_kst, kv_vst, kq_code[li], vq_code[li], kq_sc[li], vq_sc[li], (int) M.HKV, (int) li, p0, ncell[li]);
        } else {
            k_kvq_enc_hq<<<g, 128, 0, st>>>(kv_kst, kv_vst, kq_code[li], vq_code[li], kq_sc[li], vq_sc[li], kq_mt[li], vq_mt[li], (int) M.HKV,
                                            (int) li, p0, ncell[li], kv_bits);
        }
    }

    /// The FP16 copy (kv_k16 / kv_v16, the cache's own layout) of keys [k0, k0 + n) of a quantised layer.
    void kv_dequant(size_t li, int k0, int n) {
        if (n <= 0) return;
        const dim3 g((unsigned) ((n + 3) / 4), (unsigned) m->HKV, 2);
        KV_DISPATCH((k_kvq_dequant<MODE, KB><<<g, 128, 0, st>>>(kq_code[li], kq_sc[li], kq_mt[li], vq_code[li], vq_sc[li], vq_mt[li], kv_k16,
                                                                 kv_v16, ncell[li], k0, n, (int) li, kv_bits)));
    }

    /// Attention part 1 of a quantised layer: the partials k_att_merge takes (the 8 query heads of a KV head in one block).
    void kv_att_part(size_t li, const float* qp, int nt, int pb, int nchunk) {
        const Model& M = *m;
        const dim3 g((unsigned) nchunk, (unsigned) (nt * M.HKV));
        KV_DISPATCH((k_att_part_q<MODE, KB><<<g, 128, 0, st>>>(qp, kq_code[li], kq_sc[li], kq_mt[li], vq_code[li], vq_sc[li], vq_mt[li], (int) M.H,
                                                                (int) M.HKV, pb, ncell[li], 1.0f / sqrtf((float) kHD), pacc, pm, pl, nchunk,
                                                                (int) li, kv_bits)));
    }

    /// The attention block of layer li for a long chunk: the projections on the matrix cores, then blocks of 256 queries: Q / K
    /// norm and rotary embedding, the cache written, scores = K^T Q by batched GEMM per KV head, the softmax with the sink,
    /// output = V P.  A sliding-window layer reads its window out of the ring into a compact buffer.
    void attention_pf(const Layer& L, size_t li, int T, int p0) {
        const Model& M = *m;
        const int H = (int) M.H, HKV = (int) M.HKV, N = (int) M.N, G = H / HKV;
        const float one = 1.0f, zero = 0.0f;
        k_to_f16<<<grid1((int64_t) T * N), kThreads, 0, st>>>(h, x16, (int64_t) T * N);
        gm(L.wq, qb, T);
        gm(L.wk, kb, T);
        gm(L.wv, vb, T);
        const int ring = L.swa ? kRing : 0, window = L.swa ? (int) M.window : 0;
        const float scale = 1.0f / sqrtf((float) kHD);
        const bool qz = kvq(li);
        if (qz) kv_dequant(li, 0, p0);   // (the keys of the chunks before: this chunk's are added as they are encoded)
        for (int t0 = 0; t0 < T; t0 += 256) {
            const int nt = std::min(256, T - t0), pb = p0 + t0;
            k_prep<<<dim3(H + HKV, nt), kHD, 0, st>>>(qb + (size_t) t0 * H * kHD, kb + (size_t) t0 * HKV * kHD,
                                                      vb + (size_t) t0 * HKV * kHD, L.q_norm, L.k_norm,
                                                      qr + (size_t) t0 * H * kHD, kc[li], vc[li], H, HKV, pb, L.rope ? 1 : 0,
                                                      M.rope_base, M.eps, ring, ncell[li], qz ? kv_kst : nullptr, qz ? kv_vst : nullptr);
            if (qz) {
                kv_encode(li, nt, pb);
                kv_dequant(li, pb, nt);
                if (kv_check) k_kv_err<<<dim3((unsigned) HKV, (unsigned) nt, 2), kHD, 0, st>>>(kv_kst, kv_vst, kv_k16, kv_v16, HKV, ncell[li], pb);
            }
            k_to_f16<<<grid1((int64_t) nt * H * kHD), kThreads, 0, st>>>(qr + (size_t) t0 * H * kHD, q16, (int64_t) nt * H * kHD);
            const int khi = pb + nt - 1, klo = window > 0 ? std::max(0, pb - window + 1) : 0, nk = khi - klo + 1;
            const __half *Kb, *Vb;
            size_t kvs;   // halves between two KV heads
            if (ring) {
                k_ring_gather<<<grid1((int64_t) HKV * nk * kHD), kThreads, 0, st>>>(kc[li], vc[li], k16w, v16w, klo, nk, ring,
                                                                                    ncell[li], HKV);
                Kb = k16w;
                Vb = v16w;
                kvs = (size_t) nk * kHD;
            } else {
                Kb = qz ? kv_k16 : kc[li];
                Vb = qz ? kv_v16 : vc[li];
                kvs = (size_t) ncell[li] * kHD;
            }
            const int ldp = (nk + 7) / 8 * 8;
            const int tq = (int) std::max<size_t>(1, std::min<size_t>((size_t) nt, att_s_floats / ((size_t) H * ldp)));
            for (int s0 = 0; s0 < nt; s0 += tq) {
                const int ns = std::min(tq, nt - s0);
                for (int kvh = 0; kvh < HKV; ++kvh) {
                    const __half* Kh = Kb + (size_t) kvh * kvs;
                    const __half* Qh = q16 + ((size_t) s0 * H + (size_t) kvh * G) * kHD;
                    float* Sh = att_s + (size_t) kvh * G * ns * ldp;
                    if (cublasGemmStridedBatchedEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, nk, ns, kHD, &one, Kh, CUDA_R_16F, kHD, 0, Qh,
                                                   CUDA_R_16F, H * kHD, kHD, &zero, Sh, CUDA_R_32F, ldp, (long long) ns * ldp, G,
                                                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS) {
                        std::fprintf(stderr, "strata-flint: attention scores GEMM failed\n");
                        std::exit(1);
                    }
                }
                k_att_softmax_sink<<<dim3(ns, H), kThreads, 0, st>>>(att_s, att_p16, ldp, ns, nk, klo, pb + s0, window, L.sinks, scale);
                for (int kvh = 0; kvh < HKV; ++kvh) {
                    const __half* Vh = Vb + (size_t) kvh * kvs;
                    const __half* Ph = att_p16 + (size_t) kvh * G * ns * ldp;
                    float* Oh = att + ((size_t) (t0 + s0) * H + (size_t) kvh * G) * kHD;
                    if (cublasGemmStridedBatchedEx(blas, CUBLAS_OP_N, CUBLAS_OP_N, kHD, ns, nk, &one, Vh, CUDA_R_16F, kHD, 0, Ph,
                                                   CUDA_R_16F, ldp, (long long) ns * ldp, &zero, Oh, CUDA_R_32F, H * kHD, kHD, G,
                                                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS) {
                        std::fprintf(stderr, "strata-flint: attention values GEMM failed\n");
                        std::exit(1);
                    }
                }
            }
        }
        k_to_f16<<<grid1((int64_t) T * H * kHD), kThreads, 0, st>>>(att, x16, (int64_t) T * H * kHD);
        gm(L.wo, y, T);
        k_add<<<grid1((int64_t) T * N), kThreads, 0, st>>>(x, y, (int64_t) T * N);
    }

    /// The routed experts of a layer for a long chunk: the router, the entries sorted by expert, then two passes over the
    /// layer's groups of 16 experts through MMQ (gate+up, the SwiGLU of every row, down), each group's weights waiting in the
    /// ring.  No wait for the host: every expert is loaded, whether or not a token chose it.
    void moe_prefill(const Layer& L, size_t li, int T) {
        const Model& M = *m;
        const int ne = (int) M.NE, Kp = (int) M.TK, n_ent = T * Kp, FF = (int) M.FFE, N = (int) M.N;
        const float one = 1.0f, zero = 0.0f;
        if (cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, ne, T, N, &one, L.router, N, h, N, &zero, logits_r, ne) != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "strata-flint: router GEMM failed\n");
            std::exit(1);
        }
        k_route<<<T, 256, 0, st>>>(logits_r, ne, Kp, M.w_scale, ids, ewts);
        chk("router logits", logits_r, (int64_t) T * ne);
        chk("normed input h", h, (int64_t) T * N);
        CK(cudaMemsetAsync(rcnt, 0, (size_t) ne * 4, st));
        k_route_count<<<grid1(n_ent), kThreads, 0, st>>>(ids, n_ent, rcnt);
        k_route_scan<<<1, 32, 0, st>>>(rcnt, ne, rbounds, rcur);
        k_route_scatter<<<grid1(n_ent), kThreads, 0, st>>>(ids, n_ent, Kp, rcur, ent_dst, ent_tok, rinv);
        MMQ::quantize(h, ent_tok, mq_x, L.L.gu_type, N, N, n_ent, st);
        const int ng = ne / kPfGroup;
        for (int pass = 0; pass < 2; ++pass) {
            if (pass == 1) {
                chk("MMQ gate+up rows", mq_gu, (int64_t) n_ent * 2 * FF);
                MMQ::swiglu(mq_gu, mq_h, n_ent, FF, false, st);
                k_row_fit<<<n_ent, kThreads, 0, st>>>(mq_h, FF, rsc_e, kFitTarget);
                chk("swiglu rows", mq_h, (int64_t) n_ent * FF);
                MMQ::quantize(mq_h, nullptr, mq_hq, L.L.d_type, FF, FF, n_ent, st);
            }
            for (int g = 0; g < ng; ++g) {
                const size_t seq = pf_next++;
                const int b = (int) (seq % kPfNB);
                pf_wait_produced(seq);
                CK(cudaStreamWaitEvent(st, pf_ready[b], 0));
                MMQ::Product pr;
                pr.w = pf_gbuf[b];
                pr.n = kPfGroup;
                pr.bounds = rbounds + (size_t) g * kPfGroup;
                pr.ids = ident;
                pr.total_rows = n_ent;
                pr.max_rows = T;
                if (pass == 0) {
                    pr.type = L.L.gu_type; pr.w_rows = 2 * FF; pr.w_cols = N; pr.expert_bytes = pf_stride(L, 0);
                    pr.xq = mq_x; pr.dst = mq_gu; pr.ld_dst = 2 * FF;
                } else {
                    pr.type = L.L.d_type; pr.w_rows = N; pr.w_cols = FF; pr.expert_bytes = pf_stride(L, 1);
                    pr.xq = mq_hq; pr.dst = eparts; pr.ld_dst = N;
                }
                mctx->run(pr, st);
                if (finite_check && li == 2) {   // the first MoE layer, group by group
                    std::vector<int32_t> hb((size_t) ne + 1);
                    CK(cudaMemcpyAsync(hb.data(), rbounds, hb.size() * 4, cudaMemcpyDeviceToHost, st));
                    CK(cudaStreamSynchronize(st));
                    const int r0 = hb[(size_t) g * kPfGroup], r1 = hb[(size_t) (g + 1) * kPfGroup];
                    char nm[96];
                    std::snprintf(nm, sizeof nm, "pass %d group %d (experts %d-%d, rows %d-%d, type %d)", pass, g, g * kPfGroup,
                                  g * kPfGroup + kPfGroup - 1, r0, r1, pass == 0 ? L.L.gu_type : L.L.d_type);
                    if (pass == 0 && g == 0 && r1 > r0) {   // which rows and columns of group 0 are not finite
                        std::vector<float> hr((size_t) (r1 - r0) * 2 * FF);
                        CK(cudaMemcpy(hr.data(), mq_gu + (size_t) r0 * 2 * FF, hr.size() * 4, cudaMemcpyDeviceToHost));
                        std::vector<int32_t> hid2((size_t) n_ent);
                        CK(cudaMemcpy(hid2.data(), ids, (size_t) n_ent * 4, cudaMemcpyDeviceToHost));
                        std::fprintf(stderr, "  group 0: rows with non-finite values (row: count, first column, last column):\n");
                        int shown = 0;
                        for (int rr = 0; rr < r1 - r0 && shown < 12; ++rr) {
                            int cnt = 0, c0 = -1, c1 = -1;
                            for (int c = 0; c < 2 * FF; ++c)
                                if (!std::isfinite(hr[(size_t) rr * 2 * FF + (size_t) c])) {
                                    ++cnt;
                                    if (c0 < 0) c0 = c;
                                    c1 = c;
                                }
                            if (cnt) {
                                std::fprintf(stderr, "    row %d: %d, cols %d..%d\n", r0 + rr, cnt, c0, c1);
                                ++shown;
                            }
                        }
                    }
                    if (r1 > r0) chk(nm, (pass == 0 ? mq_gu : eparts) + (size_t) r0 * (pass == 0 ? 2 * FF : N), (int64_t) (r1 - r0) * (pass == 0 ? 2 * FF : N));
                }
                CK(cudaEventRecord(pf_free[b], st));
                pf_mark_consumed(seq + 1);
            }
        }
        chk("MMQ down rows", eparts, (int64_t) n_ent * N);
        (void) li;
    }

    /// The feed-forward block of layer li for a long chunk (h holds the normed input).
    void ffn_pf(const Layer& L, size_t li, int T) {
        const Model& M = *m;
        const int N = (int) M.N, Kp = (int) M.TK;
        k_to_f16<<<grid1((int64_t) T * N), kThreads, 0, st>>>(h, x16, (int64_t) T * N);
        if (L.dense) {
            gm(L.ffn_gate, g1, T);
            gm(L.ffn_up, u1, T);
            k_swiglu<<<grid1((int64_t) T * M.FF), kThreads, 0, st>>>(g1, u1, g1, (int64_t) T * M.FF);
            k_row_fit<<<T, kThreads, 0, st>>>(g1, (int) M.FF, rsc_s, kFitTarget);
            k_to_f16<<<grid1((int64_t) T * M.FF), kThreads, 0, st>>>(g1, x16, (int64_t) T * M.FF);
            gemm.native((const uint16_t*) x16, L.ffn_down.type, L.ffn_down.d, y, T, L.ffn_down.n_out, L.ffn_down.n_in);
            k_row_mul<<<T, kThreads, 0, st>>>(y, rsc_s, N);
            k_add<<<grid1((int64_t) T * N), kThreads, 0, st>>>(x, y, (int64_t) T * N);
            return;
        }
        moe_prefill(L, li, T);
        gm(L.sh_gate, sg, T);
        gm(L.sh_up, su, T);
        k_swiglu<<<grid1((int64_t) T * M.FFS), kThreads, 0, st>>>(sg, su, sg, (int64_t) T * M.FFS);
        k_row_fit<<<T, kThreads, 0, st>>>(sg, (int) M.FFS, rsc_s, kFitTarget);
        k_to_f16<<<grid1((int64_t) T * M.FFS), kThreads, 0, st>>>(sg, x16, (int64_t) T * M.FFS);
        gemm.native((const uint16_t*) x16, L.sh_down.type, L.sh_down.d, sy, T, L.sh_down.n_out, L.sh_down.n_in);
        k_row_mul<<<T, kThreads, 0, st>>>(sy, rsc_s, N);
        chk("shared expert rows", sy, (int64_t) T * N);
        k_combine_sorted<<<T, kThreads, 0, st>>>(x, sy, eparts, rinv, ewts, rsc_e, Kp, N);
    }

    /// The attention block of layer li for T tokens at positions p0 ..: x += wo(attn(norm(x))).  h holds norm(x).
    void attention(const Layer& L, size_t li, int T, int p0, bool fuse_add = false) {
        const Model& M = *m;
        const int H = (int) M.H, HKV = (int) M.HKV, N = (int) M.N;
        quant(h, N, T, xq);
        mm(L.wq, xq, qb, T);
        mm(L.wk, xq, kb, T);
        mm(L.wv, xq, vb, T);
        const int ring = L.swa ? kRing : 0;
        const bool qz = kvq(li);
        k_prep<<<dim3(H + HKV, T), kHD, 0, st>>>(qb, kb, vb, L.q_norm, L.k_norm, qr, kc[li], vc[li], H, HKV, p0, L.rope ? 1 : 0,
                                                 M.rope_base, M.eps, ring, ncell[li], qz ? kv_kst : nullptr, qz ? kv_vst : nullptr);
        if (qz) kv_encode(li, T, p0);
        const int window = L.swa ? (int) M.window : 0;
        for (int t0 = 0; t0 < T; t0 += kQB) {   // query blocks: the partials are sized for kQB queries
            const int nt = std::min(kQB, T - t0);
            const int pb = p0 + t0, last = pb + nt - 1;
            const int n_max = window > 0 ? std::min(last + 1, window) : last + 1;
            const int nchunk = (n_max + kAttChunk - 1) / kAttChunk;
            if (qz)
                kv_att_part(li, qr + (size_t) t0 * H * kHD, nt, pb, nchunk);
            else
                k_att_part<<<dim3(nchunk, nt * H), kHD, 0, st>>>(qr + (size_t) t0 * H * kHD, kc[li], vc[li], H, HKV, pb, window,
                                                                 ring, ncell[li], 1.0f / sqrtf((float) kHD), pacc, pm, pl, nchunk);
            k_att_merge<<<nt * H, kHD, 0, st>>>(pacc, pm, pl, L.sinks, H, nchunk, att + (size_t) t0 * H * kHD);
        }
        quant(att, H * kHD, T, q8);
        mm(L.wo, q8, y, T);
        if (!fuse_add) k_add<<<grid1((int64_t) T * N), kThreads, 0, st>>>(x, y, (int64_t) T * N);   // (fuse_add: the next kernel adds it)
    }

    /// T tokens (host ids) at positions p0 ..; with `want_logits` the last row's logits are left in lg.
    void forward(const int32_t* toks, int T, int p0, bool want_logits) {
        const Model& M = *m;
        const auto tf0 = Clock::now();
        const int N = (int) M.N;
        if (pro_pending) {   // the promotions of the last step have landed before this one reads the cache
            CK(cudaStreamWaitEvent(st, ev_pro, 0));
            if (T > kPf2Min) CK(cudaStreamSynchronize(st_pro));   // (and before a prompt copies out of the cache)
        }
        if (!skip_tok_copy) CK(cudaMemcpyAsync(tok, toks, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        mark("start");
        k_embed<<<T, kThreads, 0, st>>>(tok, M.embd, M.embd_row, M.embd_type, x, N);
        mark("embed");
        const bool pf = pf2 && T > kPf2Min;
        if (pf) pf_begin();
        for (size_t l = 0; l < M.layers.size(); ++l) {
            const Layer& L = M.layers[l];
            k_rms<<<T, kThreads, 0, st>>>(x, L.attn_norm, h, N, M.eps);
            mark("rms_attn");
            // Outside the prompt path the attention's residual add, the FFN's norm and the q8_1 image of its input are one kernel
            // (whose arithmetic is k_add + k_rms + the quantizer: same bits).
            const bool fuse = !pf && !secprof && !std::getenv("STRATA_FLINT_NO_FUSE");
            if (pf) attention_pf(L, l, T, p0);
            else attention(L, l, T, p0, fuse);
            mark("attention");
            if (pf && finite_check) {
                if (!d_cnt) CK(cudaMalloc(&d_cnt, 4));
                CK(cudaMemsetAsync(d_cnt, 0, 4, st));
                k_nonfinite<<<grid1((int64_t) T * N), kThreads, 0, st>>>(x, (int64_t) T * N, d_cnt);
                int hc = 0;
                CK(cudaMemcpyAsync(&hc, d_cnt, 4, cudaMemcpyDeviceToHost, st));
                CK(cudaStreamSynchronize(st));
                if (hc) std::fprintf(stderr, "finite check: layer %zu after the attention (%s): %d non-finite values in x\n", l,
                                     L.swa ? "window" : "full", hc);
            }
            if (fuse) {
                k_add_rms_q8<<<T, 256, 0, st>>>(x, y, L.ffn_norm, h, (Q8Blk*) xq, N, M.eps);
                mark("add_rms_quant");
            } else {
                k_rms<<<T, kThreads, 0, st>>>(x, L.ffn_norm, h, N, M.eps);
                mark("rms_ffn");
            }
            if (pf) {
                ffn_pf(L, l, T);
            } else {
                if (!fuse) {
                    quant(h, N, T, xq);
                    mark("quantize");
                }
                ffn(L, l, T);
            }
            mark(L.dense ? "ffn_dense" : "ffn_moe_total");
            if (pf && finite_check) {   // STRATA_FLINT_CHECK_FINITE=1: where do non-finite values first appear?
                if (!d_cnt) CK(cudaMalloc(&d_cnt, 4));
                const char* names[] = {"after the layer"};
                (void) names;
                CK(cudaMemsetAsync(d_cnt, 0, 4, st));
                k_nonfinite<<<grid1((int64_t) T * N), kThreads, 0, st>>>(x, (int64_t) T * N, d_cnt);
                int hc = 0;
                CK(cudaMemcpyAsync(&hc, d_cnt, 4, cudaMemcpyDeviceToHost, st));
                CK(cudaStreamSynchronize(st));
                if (hc) std::fprintf(stderr, "finite check: layer %zu (%s%s): %d non-finite values in x\n", l, L.dense ? "dense" : "MoE",
                                     L.swa ? ", window" : ", full", hc);
            }
            if (dump) {
                std::vector<float> hx((size_t) T * N);
                CK(cudaMemcpyAsync(hx.data(), x, (size_t) T * N * 4, cudaMemcpyDeviceToHost, st));
                CK(cudaStreamSynchronize(st));
                for (int t = 0; t < T; ++t)
                    std::memcpy(dump->data() + ((size_t) (p0 + t) * M.layers.size() + l) * N, hx.data() + (size_t) t * N,
                                (size_t) N * 4);
            }
            if (trace) {
                std::vector<float> hx((size_t) N);
                CK(cudaMemcpyAsync(hx.data(), x + (size_t) (T - 1) * N, (size_t) N * 4, cudaMemcpyDeviceToHost, st));
                CK(cudaStreamSynchronize(st));
                double s = 0;
                for (float v : hx) s += std::fabs(v);
                std::fprintf(stderr, "trace layer %2zu: mean|x| %.6f  x[0..3] %.5f %.5f %.5f %.5f\n", l, s / N, hx[0], hx[1],
                             hx[2], hx[3]);
            }
        }
        if (pro_pending) CK(cudaEventRecord(ev_pro, st_pro));   // (the next step waits for this step's promotions)
        if (pf) {   // every group of experts has been consumed: the loader thread is done
            CK(cudaStreamSynchronize(st));
            pf_end();
        }
        if (nll) {   // the loss of every row of the chunk: final norm, then the head 16 rows at a time
            k_rms<<<T, kThreads, 0, st>>>(x, M.out_norm, h, N, M.eps);
            quant(h, N, T, xq);
            if (!lgb) CK(cudaMalloc(&lgb, (size_t) 16 * M.VOC * 4));
            std::vector<float> row((size_t) 16 * M.VOC);
            const size_t qb = K::native_q8_1_bytes(N, 1);
            for (int r0 = 0; r0 < T; r0 += 16) {
                const int nr = std::min(16, T - r0);
                for (int c0 = 0; c0 < nr; c0 += kCols)
                    K::native_mmvq(M.head.type, M.head.d, (const char*) xq + (size_t) (r0 + c0) * qb,
                                   lgb + (size_t) c0 * M.VOC, M.head.n_in, M.head.n_out, std::min(kCols, nr - c0), st);
                CK(cudaMemcpyAsync(row.data(), lgb, (size_t) nr * M.VOC * 4, cudaMemcpyDeviceToHost, st));
                CK(cudaStreamSynchronize(st));
                for (int r = 0; r < nr; ++r) {
                    const int pos = p0 + r0 + r;
                    if (pos + 1 >= nll_len) continue;
                    const float* l = row.data() + (size_t) r * M.VOC;
                    double mx = l[0];
                    for (int64_t i = 1; i < M.VOC; ++i) mx = std::max(mx, (double) l[i]);
                    double z = 0;
                    for (int64_t i = 0; i < M.VOC; ++i) z += std::exp((double) l[i] - mx);
                    (*nll)[(size_t) pos] = (float) (std::log(z) + mx - (double) l[nll_next[pos + 1]]);
                }
            }
        }
        if (!want_logits) return;
        k_rms<<<1, kThreads, 0, st>>>(x + (size_t) (T - 1) * N, M.out_norm, h, N, M.eps);
        quant(h, N, 1, xq);
        mm(M.head, xq, lg, 1);
        mark("head");
        if (prof && T == 1) {
            CK(cudaStreamSynchronize(st));
            t_fwd += ms_since(tf0);
            ++n_fwd;
        }
    }

    void host_logits(std::vector<float>& out) {
        out.resize((size_t) m->VOC);
        CK(cudaMemcpyAsync(out.data(), lg, (size_t) m->VOC * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
    }
};

/// The sizes of the chunks a prompt of n tokens is read in: as few as chunks of at most `tmax` allow, of near-equal size (every
/// chunk costs a pass over all the experts, so a short tail is a waste).
std::vector<int> chunk_sizes(int n, int tmax) {
    std::vector<int> v;
    if (n <= 0) return v;
    const int k = (n + tmax - 1) / tmax;
    for (int i = 0; i < k; ++i) v.push_back(n / k + (i < n % k ? 1 : 0));
    return v;
}

std::vector<int32_t> parse_ids(const char* s) {
    std::vector<int32_t> v;
    const char* p = s;
    while (*p) {
        char* e = nullptr;
        const long x = std::strtol(p, &e, 10);
        if (e == p) break;
        v.push_back((int32_t) x);
        p = *e == ',' ? e + 1 : e;
    }
    return v;
}

// ------------------------------------------------------------------------------------------------ the engine protocol
/// The host sampler (the one of the Qwen3.6 path): penalties, then greedy, or top-k / top-p / min-p at the temperature.
int sample(std::vector<float>& lg, const strata::qwen35::Sampling& s, const std::vector<int32_t>& hist, std::mt19937_64& rng) {
    strata::qwen35::apply_penalties(lg, s, hist);
    if (s.temperature <= 0.f) return (int) (std::max_element(lg.begin(), lg.end()) - lg.begin());
    const int kk = std::max(1, std::min(s.top_k > 0 ? s.top_k : 64, (int) lg.size()));
    std::vector<int> idx(lg.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int) i;
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(), [&](int a, int b) { return lg[(size_t) a] > lg[(size_t) b]; });
    idx.resize((size_t) kk);
    std::vector<double> pr(idx.size());
    const double mx = lg[(size_t) idx[0]];
    double sum = 0;
    for (size_t i = 0; i < idx.size(); ++i) sum += (pr[i] = std::exp((lg[(size_t) idx[i]] - mx) / s.temperature));
    size_t keep = idx.size();
    double cum = 0;
    for (size_t i = 0; i < idx.size(); ++i) {
        if (pr[i] / sum < s.min_p * (pr[0] / sum)) { keep = i; break; }
        cum += pr[i] / sum;
        if (cum >= s.top_p) { keep = i + 1; break; }
    }
    keep = std::max<size_t>(1, keep);
    double tot = 0;
    for (size_t i = 0; i < keep; ++i) tot += pr[i];
    std::uniform_real_distribution<double> u(0.0, tot);
    double r = u(rng);
    for (size_t i = 0; i < keep; ++i)
        if ((r -= pr[i]) <= 0) return idx[i];
    return idx[keep - 1];
}

/// The protocol serve/server.py speaks to an engine (the same lines as the Qwen3.6 path, one request at a time):
///   READY <context> stop                     once, when the model is placed
///   GEN <max_new> [key=value ...] <id,id,...>   a request; keys: temperature top_p top_k min_p penalty_last_n penalty_repeat
///                                            penalty_freq penalty_present seed (others are ignored)
///   PP <done> <total> <ms> <tok/s>           while the prompt is read (a chunk each)
///   T <id>                                   every generated token (the ending one too)
///   DONE <n> <prompt> <prompt ms> <decode ms> <finish> <accepted> <offered> <reused> 0 0 0 0 0.0 <read> 0
///   STOP                                     (any time) ends the request at the next token or chunk
///   ERR <text>                               a request that cannot run
/// The tokens the cache holds (`live`) are kept between requests: a prompt that starts with them reads only its new end.
/// The sliding-window layers keep a ring of kRing cells, so a state can be continued only from at most 512 tokens before
/// its end (the window it needs is still in the ring); a prompt that diverges earlier is read from the start.
struct Server {
    Engine& E;
    const Model& M;
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> lines;
    bool eof = false;
    std::atomic<bool> stop{false};
    std::vector<int32_t> live;   // positions 0 .. live.size() - 1 are in the KV caches

    Server(Engine& e, const Model& m) : E(e), M(m) {}

    bool is_eog(int32_t t) const { return std::find(M.eog.begin(), M.eog.end(), t) != M.eog.end(); }

    void request(const std::string& line) {
        if (line.rfind("GEN ", 0) != 0) {
            std::printf("ERR expected: GEN <max_new> [key=value ...] <id,id,...>\n");
            std::fflush(stdout);
            return;
        }
        const char* p = line.c_str() + 4;
        char* e1 = nullptr;
        const long long max_new = std::strtoll(p, &e1, 10);
        p = e1;
        strata::qwen35::Sampling smp;
        for (;;) {
            while (*p == ' ') ++p;
            const char* st0 = p;
            while (*p && *p != ' ') ++p;
            const std::string tk(st0, (size_t) (p - st0));
            const size_t eq = tk.find('=');
            if (tk.empty() || eq == std::string::npos) { p = st0; break; }
            const std::string key = tk.substr(0, eq);
            const char* v = tk.c_str() + eq + 1;
            if (key == "temperature") smp.temperature = std::strtof(v, nullptr);
            else if (key == "top_p") smp.top_p = std::strtof(v, nullptr);
            else if (key == "top_k") smp.top_k = std::atoi(v);
            else if (key == "min_p") smp.min_p = std::strtof(v, nullptr);
            else if (key == "penalty_last_n") smp.last_n = std::atoi(v);
            else if (key == "penalty_repeat") smp.repeat = std::strtof(v, nullptr);
            else if (key == "penalty_freq") smp.freq = std::strtof(v, nullptr);
            else if (key == "penalty_present") smp.present = std::strtof(v, nullptr);
            else if (key == "seed") smp.seed = std::strtoull(v, nullptr, 10);
        }
        const std::vector<int32_t> ids = parse_ids(p);
        if (max_new < 1 || ids.empty()) {
            std::printf("ERR bad request: max_new or the ids\n");
            std::fflush(stdout);
            return;
        }
        if ((long long) ids.size() + max_new > E.ctx) {
            std::printf("ERR prompt (%zu tokens) + max_new (%lld) exceeds the context (%d)\n", ids.size(), max_new, E.ctx);
            std::fflush(stdout);
            return;
        }
        for (int32_t t : ids)
            if (t < 0 || t >= (int32_t) M.VOC) {
                std::printf("ERR a token id is outside the vocabulary\n");
                std::fflush(stdout);
                return;
            }
        stop.store(false);
        // what the cache already holds of this prompt
        size_t reuse = 0;
        while (reuse < live.size() && reuse < ids.size() && live[reuse] == ids[reuse]) ++reuse;
        if (reuse == ids.size()) reuse = ids.size() - 1;                 // the last token is read again: its logits start the answer
        if (reuse > 0 && reuse + (size_t) (kRing / 2) < live.size()) reuse = 0;   // the window before `reuse` is gone from the ring
        live.resize(reuse);
        const auto t0 = Clock::now();
        const long long fresh = (long long) (ids.size() - reuse);
        bool cancelled = false;
        size_t done = reuse;
        const std::vector<int> sizes = chunk_sizes((int) (ids.size() - reuse), E.tmax);
        size_t chunk_i = 0;
        while (done < ids.size()) {
            const int T = sizes[chunk_i++];
            const bool last = done + (size_t) T == ids.size();
            E.forward(ids.data() + done, T, (int) done, last);
            CK(cudaStreamSynchronize(E.st));
            live.insert(live.end(), ids.begin() + (long) done, ids.begin() + (long) done + T);
            done += (size_t) T;
            std::printf("PP %lld %lld %.0f %.1f\n", (long long) (done - reuse), fresh, ms_since(t0),
                        (double) (done - reuse) / std::max(1e-9, ms_since(t0) / 1000.0));
            std::fflush(stdout);
            if (stop.load() && done < ids.size()) { cancelled = true; break; }
        }
        const double prompt_ms = ms_since(t0);
        if (cancelled) {
            std::printf("DONE 0 %zu %.1f 0.0 cancel 0 0 %zu 0 0 0 0 0.0 %lld 0\n", ids.size(), prompt_ms, reuse,
                        (long long) (done - reuse));
            std::fflush(stdout);
            return;
        }
        std::mt19937_64 rng(smp.seed);
        std::vector<int32_t> hist = ids;
        std::vector<float> lg;
        const auto d0 = Clock::now();
        long long produced = 0;
        const char* finish = "length";
        int pos = (int) ids.size();
        for (;;) {
            E.host_logits(lg);
            const int32_t nx = (int32_t) sample(lg, smp, hist, rng);
            std::printf("T %d\n", nx);
            std::fflush(stdout);
            ++produced;
            hist.push_back(nx);
            if (is_eog(nx)) { finish = "stop"; break; }
            if (produced >= max_new) break;
            if (stop.load()) { finish = "cancel"; break; }
            if (pos + 1 > E.ctx) break;
            E.forward(&nx, 1, pos, true);
            live.push_back(nx);
            ++pos;
        }
        const double decode_ms = ms_since(d0);
        std::printf("DONE %lld %zu %.1f %.1f %s 0 0 %zu 0 0 0 0 0.0 %lld 0\n", produced, ids.size(), prompt_ms, decode_ms, finish,
                    reuse, fresh);
        std::fflush(stdout);
        std::fprintf(stderr, "strata serve: prompt %zu tokens = %zu reused + %lld read in %.0f ms (%.1f tok/s), %lld generated in "
                     "%.0f ms (%.1f tok/s) (%s)\n", ids.size(), reuse, fresh, prompt_ms,
                     prompt_ms > 0 ? 1000.0 * (double) fresh / prompt_ms : 0.0, produced, decode_ms,
                     decode_ms > 0 ? 1000.0 * (double) produced / decode_ms : 0.0, finish);
    }

    int run() {
        std::printf("READY %d stop\n", E.ctx);
        std::fflush(stdout);
        std::thread reader([&] {
            std::string l;
            while (std::getline(std::cin, l)) {
                if (!l.empty() && l.back() == '\r') l.pop_back();
                if (l == "STOP") { stop.store(true); continue; }
                std::lock_guard<std::mutex> lk(mu);
                lines.push_back(l);
                cv.notify_one();
            }
            std::lock_guard<std::mutex> lk(mu);
            eof = true;
            cv.notify_one();
        });
        reader.detach();
        for (;;) {
            std::string line;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return eof || !lines.empty(); });
                if (lines.empty()) return 0;
                line = lines.front();
                lines.pop_front();
            }
            if (line.empty()) continue;
            request(line);
        }
    }
};

}  // namespace

int strata::flint::run(int argc, char** argv) {
    std::string path, dump, dump_layers;
    std::vector<int32_t> prompt, forced;
    int n_new = 16, ctx = 4096, bench = 0, kv_mode = 0, kv_bits = 3;
    double expert_vram = -1;
    bool trace = false, buffered = false, prof = false;
    int chunk = 2048;
    double host_gib = -1;
    bool pin_host = true, prof_predict = false;
    int bench_floor = 0;
    int prefetch_k = 0;   // the lookahead (--prefetch N): measured no faster than the page-locked host tier alone
    std::string nll_file;
    bool serve_mode = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "strata-flint: %s needs a value\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--model" || a == "--native") path = next();
        else if (a == "--serve") serve_mode = true;
        else if (a == "--ctx") ctx = std::atoi(next());
        else if (a == "--prefill") { const std::string v = next(); chunk = v == "auto" ? 2048 : std::atoi(v.c_str()); }
        else if (a == "--gpu") (void) next();   // applied by strata before the switch (HIP_VISIBLE_DEVICES)
        else if (a == "--tokens") prompt = parse_ids(next());
        else if (a == "--n" || a == "--max-new") n_new = std::atoi(next());
        else if (a == "--dump-full-logits") dump = next();
        else if (a == "--max-context") ctx = std::atoi(next());
        else if (a == "--expert-vram") expert_vram = std::atof(next());
        else if (a == "--bench-tg") bench = std::atoi(next());
        else if (a == "--kv-quant") {
            const std::string v = next();
            if (v == "none" || v == "fp16") kv_mode = 0;
            else if (v == "tqt" || v == "turboquant-trellis") kv_mode = 1;
            else if (v == "hq" || v == "hyperquant") kv_mode = 2;
            else {
                std::fprintf(stderr, "strata-flint: --kv-quant takes none, tqt or hq (not %s)\n", v.c_str());
                return 2;
            }
        } else if (a == "--kv-bits") {
            kv_bits = std::atoi(next());
            if (kv_bits < 2 || kv_bits > 4) {
                std::fprintf(stderr, "strata-flint: --kv-bits takes 2, 3 or 4\n");
                return 2;
            }
        }
        else if (a == "--trace") trace = true;
        else if (a == "--buffered") buffered = true;
        else if (a == "--prof") prof = true;
        else if (a == "--chunk") chunk = std::atoi(next());
        else if (a == "--host-cache") host_gib = std::atof(next());
        else if (a == "--pin-host") pin_host = true;
        else if (a == "--no-pin-host") pin_host = false;
        else if (a == "--prefetch") prefetch_k = std::atoi(next());
        else if (a == "--prof-predict") prof_predict = true;
        else if (a == "--bench-floor") bench_floor = std::atoi(next());
        else if (a == "--nll") nll_file = next();
        else if (a == "--dump-layers") dump_layers = next();
        else if (a == "--force") forced = parse_ids(next());
        else if (a == "--help" || a == "-h") {
            std::printf("strata-flint --model FILE --tokens 1,2,3 [options]\n"
                        "  --n N                 tokens to generate (greedy), default 16\n"
                        "  --force IDS           teacher forcing: feed these ids instead of the greedy ones\n"
                        "  --max-context N       cells of the KV cache of the full-attention layers (default: prompt + new tokens)\n"
                        "  --kv-quant MODE       the KV cache of the full-attention layers: none (FP16, default), tqt (TurboQuant Trellis) or hq (HyperQuant)\n"
                        "  --kv-bits N           bits per sample of --kv-quant, 2, 3 (default) or 4: 3 bits is about 5x less cache than FP16\n"
                        "  --chunk N             prompt tokens per forward, 8..4096 (default 2048; 512 without the long-prompt path; every 1000 tokens of chunk takes ~0.3 GiB from the expert cache in VRAM)\n"
                        "  --expert-vram GiB     VRAM for the expert cache (default: what is free after a warmup, less 0.6 GiB)\n"
                        "  --host-cache GiB      host RAM for the expert tier below it (default: MemAvailable less 2.5 GiB, at most 70% of the RAM)\n"
                        "  --no-pin-host         do not page-lock the host tier (slower: its swaps are staged and synchronous)\n"
                        "  --prefetch N          lookahead: bring N guessed experts of the next layer in early (default 0, no gain measured)\n"
                        "  --buffered            read the file through the page cache instead of O_DIRECT (slower)\n"
                        "  --bench-tg N          generate N tokens and report the speed\n"
                        "  --dump-full-logits F  the logits of every step (f32 rows), for tools/flint/compare.py\n"
                        "  --dump-layers F       the residual after every layer of every position (f32)\n"
                        "  --nll F               -log p(next prompt token) of every prompt position (f32), for a perplexity check\n"
                        "  --prof, --prof-predict, --trace   timing, lookahead-guess and per-layer diagnostics\n");
            return 0;
        } else if (serve_mode && a.rfind("--", 0) == 0) {
            // serve passes the config's args plus a few of its own: one this engine does not have is said and skipped
            // (with its value), not fatal
            const bool val = i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0;
            std::fprintf(stderr, "strata-flint: ignoring %s%s%s (not an option of this engine)\n", a.c_str(), val ? " " : "",
                         val ? argv[i + 1] : "");
            if (val) ++i;
        } else {
            std::fprintf(stderr, "strata-flint: unknown argument %s\n", a.c_str());
            return 2;
        }
    }
    if (path.empty() || (prompt.empty() && !serve_mode)) {
        std::fprintf(stderr, "strata-flint: --model FILE and --tokens IDS are needed (see --help)\n");
        return 2;
    }
    setenv("STRATA_MMQ_KQ", "1", 0);   // the K-quant experts of the Unsloth files on MMQ (the long-prompt path needs them)
    Model m;
    std::string err;
    try {
        if (!load(path, m, err)) {
            std::fprintf(stderr, "strata-flint: %s\n", err.c_str());
            return 1;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-flint: %s\n", e.what());
        return 1;
    }
    if (serve_mode && ctx == 4096) ctx = 32768;   // (the default of a server; --max-context says otherwise)
    if (!serve_mode && (int) prompt.size() + n_new + 8 > ctx) ctx = (int) prompt.size() + n_new + 8;
    Engine E;
    E.trace = trace;
    E.buffered = buffered;
    E.prof = prof;
    E.host_gib = host_gib;
    E.pin_host = pin_host;
    E.prefetch_k = prefetch_k;
    E.prof_predict = prof_predict;
    std::vector<float> nll_buf;
    if (!nll_file.empty()) {
        nll_buf.assign(prompt.size(), 0.f);
        E.nll = &nll_buf;
        E.nll_next = prompt.data();
        E.nll_len = (int) prompt.size();
    }
    E.tmax = std::max(kMaxT, std::min(chunk, 4096));
    std::vector<float> layer_dump;
    if (!dump_layers.empty()) {
        const size_t steps = forced.empty() ? (size_t) std::max(bench, n_new) : forced.size() + 1;
        layer_dump.assign((prompt.size() + steps) * (size_t) m.L * (size_t) m.N, 0.f);
        E.dump = &layer_dump;
    }
    E.kv_mode = kv_mode;
    E.kv_bits = kv_bits;
    if (const char* kc_ = std::getenv("STRATA_FLINT_KV_CHECK")) E.kv_check = kc_[0] == '1';
    E.init(m, ctx);
    E.init_staging();
    E.pf_init();
    if (!E.pf2 && E.tmax > 512) E.tmax = 512;   // (without the long-prompt path a chunk is written into the 1024-cell ring whole: at most 512)
    {
        // The library's kernels load into VRAM on first use: run every shape once (one token, a chunk of 8) with all the
        // experts streamed, so the memory that is left is the memory that is really free
        std::vector<float>* nll_keep = E.nll;
        E.nll = nullptr;   // (the warmup is not the prompt)
        std::vector<int32_t> wt((size_t) 24, 0);
        E.forward(wt.data(), 1, 0, true);
        E.forward(wt.data(), std::min(kMaxT, E.tmax), 0, true);
        if (E.tmax > kMaxT) E.forward(wt.data(), std::min(24, E.tmax), 0, true);
        if (E.pf2) {   // the long-prompt kernels load on first use: run them now, with no expert read
            std::vector<int32_t> wl((size_t) 256, 0);
            const bool dry0 = E.pf_dry;
            E.pf_dry = true;
            E.forward(wl.data(), 256, 0, true);
            CK(cudaStreamSynchronize(E.st));
            E.pf_dry = dry0;
        }
        E.nll = nll_keep;
        CK(cudaStreamSynchronize(E.st));
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        const size_t reserve = (size_t) (0.6 * (1 << 30));
        const size_t budget = expert_vram >= 0 ? (size_t) (expert_vram * (double) (1 << 30)) : (fr > reserve ? fr - reserve : 0);
        E.place_experts(budget);
    }
    auto print_prof = [&]() {
    if (prof && E.n_fwd > 0) {
        const double f = (double) E.n_fwd;
        std::fprintf(stderr, "prof: %ld decode steps, %.1f ms each; per step streamed-layer waits: ids D2H+sync %.1f ms, "
                     "reads %.1f ms, upload+sync %.1f ms (%.1f streamed layers/step)\n", E.n_fwd, E.t_fwd / f, E.t_sync / f,
                     E.t_read / f, E.t_up / f, (double) E.n_calls / f);
        std::fprintf(stderr, "prof: per decode step: %.2f ms reading the file (of the %.2f ms in load_experts), %.2f ms waiting for the GPU at the router's sync\n",
                     E.t_ssdread / (double) E.n_fwd, E.t_read / (double) E.n_fwd, E.t_sync / (double) E.n_fwd);
        std::fprintf(stderr, "prof: VRAM misses: %ld from the host tier, %ld read from the SSD; %ld experts demoted to the host tier; %ld loaded ahead by the lookahead\n",
                     E.host_hits_, E.ssd_reads_, E.demoted_, E.pf_loaded_);
        if (E.pf_filled_v + E.pf_filled_h > 0)
            std::fprintf(stderr, "prof: the prompt filled %ld VRAM-cache and %ld host-tier slots\n", E.pf_filled_v, E.pf_filled_h);
        if (E.total_ > 0)
            std::fprintf(stderr, "prof: expert cache hit rate over the whole run (prompt included): %.1f%% of %ld choices\n",
                         100.0 * (double) E.hits_ / (double) E.total_, E.total_);
        if (E.lru_total > 0) {
            std::fprintf(stderr, "prof: simulated per-layer LRU hit rate on the streamed layers (decode):");
            for (int c = 0; c < 6; ++c)
                std::fprintf(stderr, " %d slots %.0f%%", Engine::kLruCaps[c], 100.0 * (double) E.lru_hits[c] / (double) E.lru_total);
            std::fprintf(stderr, "\n");
            // a static choice: the N most-used experts of each streamed layer (counted over this very run: an upper bound)
            for (int N_ : {24, 48, 96, 160}) {
                long hit = 0, tot = 0;
                for (size_t l = 0; l < m.layers.size(); ++l) {
                    if (m.layers[l].dense || E.resident[l]) continue;
                    std::vector<long> u(E.layer_use.begin() + (long) (l * (size_t) m.NE), E.layer_use.begin() + (long) ((l + 1) * (size_t) m.NE));
                    std::sort(u.begin(), u.end(), std::greater<long>());
                    for (int i = 0; i < (int) u.size(); ++i) (i < N_ ? hit : tot) += u[(size_t) i];
                    tot += 0;
                }
                long all = 0;
                for (long v : E.layer_use) all += v;
                std::fprintf(stderr, "prof: static top-%d experts per streamed layer would cover %.0f%% of the choices\n", N_,
                             100.0 * (double) hit / (double) std::max(all, 1L));
            }
        }
    }
    };
    if (serve_mode) {
        std::signal(SIGPIPE, SIG_IGN);
        Server sv(E, m);
        const int rc = sv.run();
        print_prof();
        return rc;
    }
    if (bench_floor > 0) {   // the time of a decode step with every expert a hit and no host work: eager, then as a graph
        const int n = bench_floor;
        E.fake_hit = true;
        int32_t tk0 = 1000;
        CK(cudaMemcpy(E.tok, &tk0, 4, cudaMemcpyHostToDevice));
        E.skip_tok_copy = true;
        for (int i = 0; i < 5; ++i) E.forward(&tk0, 1, 200, true);
        CK(cudaStreamSynchronize(E.st));
        auto t1 = Clock::now();
        for (int i = 0; i < n; ++i) E.forward(&tk0, 1, 200, true);
        CK(cudaStreamSynchronize(E.st));
        const double eager = ms_since(t1) / n;
        E.secprof = true;
        for (int i = 0; i < 20; ++i) {
            E.forward(&tk0, 1, 200, true);
            E.marks_collect();
        }
        std::fprintf(stderr, "strata-flint: GPU time of a decode step by section (ms per step, from events; the sections nest as printed):\n");
        for (const auto& kv : E.sec_ms) std::fprintf(stderr, "  %-30s %6.2f\n", kv.first.c_str(), kv.second / 20.0);
        E.secprof = false;
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t gx = nullptr;
        CK(cudaStreamBeginCapture(E.st, cudaStreamCaptureModeThreadLocal));
        E.forward(&tk0, 1, 200, true);
        CK(cudaStreamEndCapture(E.st, &graph));
        CK(cudaGraphInstantiate(&gx, graph, 0));
        for (int i = 0; i < 5; ++i) CK(cudaGraphLaunch(gx, E.st));
        CK(cudaStreamSynchronize(E.st));
        t1 = Clock::now();
        for (int i = 0; i < n; ++i) CK(cudaGraphLaunch(gx, E.st));
        CK(cudaStreamSynchronize(E.st));
        const double graphed = ms_since(t1) / n;
        std::fprintf(stderr, "strata-flint: floor of a decode step (every expert a hit, no host work): %.2f ms eager (%.1f tok/s), "
                     "%.2f ms as a graph (%.1f tok/s)\n", eager, 1000.0 / eager, graphed, 1000.0 / graphed);
        return 0;
    }
    // the prompt, in chunks
    const auto t0 = Clock::now();
    const int np = (int) prompt.size();
    int c0 = 0;
    for (const int T : chunk_sizes(np, E.tmax)) {
        const auto tc0 = Clock::now();
        E.forward(prompt.data() + c0, T, c0, c0 + T == np);
        if (prof) {
            CK(cudaStreamSynchronize(E.st));
            std::fprintf(stderr, "prof: prompt chunk %d..%d: %.0f ms (%.0f tok/s); loader %.0f ms (file reads %.0f ms for %ld experts, "
                         "%ld from the VRAM cache, %ld from the host tier, waits for a free buffer %.0f ms; the file reader %.0f ms); the "
                         "compute waited for it %.0f ms\n", c0, c0 + T, ms_since(tc0), T / (ms_since(tc0) / 1000.0), E.pf_t_prod, E.pf_t_ssd, E.pf_n_ssd,
                         E.pf_n_vram, E.pf_n_host, E.pf_t_waitfree, E.pf_t_rd, E.pf_t_mainwait);
            E.pf_t_prod = E.pf_t_ssd = E.pf_t_waitfree = E.pf_t_mainwait = E.pf_t_rd = 0;
            E.pf_n_ssd = E.pf_n_vram = E.pf_n_host = 0;
        }
        c0 += T;
    }
    CK(cudaStreamSynchronize(E.st));
    std::fprintf(stderr, "strata-flint: prompt of %d tokens in %.1f ms (%.1f tok/s)\n", np, ms_since(t0),
                 np / (ms_since(t0) / 1000.0));
    if (!nll_file.empty()) {
        FILE* f = std::fopen(nll_file.c_str(), "wb");
        std::fwrite(nll_buf.data(), 4, nll_buf.size() > 0 ? nll_buf.size() - 1 : 0, f);
        std::fclose(f);
    }
    std::vector<float> lg;
    E.host_logits(lg);
    FILE* fdump = nullptr;
    if (!dump.empty()) {   // one row of logits per step: row i is the distribution after the prompt and i generated / forced tokens
        fdump = std::fopen(dump.c_str(), "wb");
        if (!fdump) {
            std::fprintf(stderr, "strata-flint: cannot write %s\n", dump.c_str());
            return 1;
        }
        std::fwrite(lg.data(), 4, lg.size(), fdump);
    }
    auto argmax = [&](const std::vector<float>& v) {
        int b = 0;
        for (size_t i = 1; i < v.size(); ++i)
            if (v[i] > v[(size_t) b]) b = (int) i;
        return b;
    };
    if (!forced.empty()) n_new = (int) forced.size() + 1;   // as the llama.cpp dump tool: one row more than forced tokens
    const int gen = bench > 0 ? bench : n_new;
    std::vector<int32_t> out;
    const auto t1 = Clock::now();
    for (int i = 0; i < gen; ++i) {
        const int32_t am = argmax(lg);
        out.push_back(am);
        if (i + 1 == gen) break;
        const int32_t nxt = i < (int) forced.size() ? forced[(size_t) i] : am;
        E.forward(&nxt, 1, np + i, true);
        E.host_logits(lg);
        if (prof && (i + 1) % 100 == 0) {   // rolling view: the speed and where the VRAM misses came from, per 100 steps
            static double last_ms = 0;
            static long last_h = 0, last_s = 0;
            const double now = ms_since(t1);
            std::fprintf(stderr, "prof: steps %4d-%4d: %.1f ms/step, VRAM misses per step: %.1f from RAM, %.1f from SSD\n", i - 98, i + 1,
                         (now - last_ms) / 100.0, (double) (E.host_hits_ - last_h) / 100.0, (double) (E.ssd_reads_ - last_s) / 100.0);
            last_ms = now;
            last_h = E.host_hits_;
            last_s = E.ssd_reads_;
        }
        if (fdump) std::fwrite(lg.data(), 4, lg.size(), fdump);
    }
    E.kv_report();
    print_prof();
    if (prof_predict && E.pr_total > 0)
        std::fprintf(stderr, "predict: next layer's experts guessed from this layer's residual: of the experts really chosen, %.0f%% are in the guess's top 6, %.0f%% in its top 12, %.0f%% in its top 16; of the VRAM misses, %.0f%% in the top 12 and %.0f%% in the top 16 (%ld misses)\n",
                     100.0 * (double) E.pr_in6 / (double) E.pr_total, 100.0 * (double) E.pr_in12 / (double) E.pr_total,
                     100.0 * (double) E.pr_in16 / (double) E.pr_total, 100.0 * (double) E.pr_miss_in12 / (double) std::max(E.pr_miss, 1L),
                     100.0 * (double) E.pr_miss_in16 / (double) std::max(E.pr_miss, 1L), E.pr_miss);
    if (fdump) std::fclose(fdump);
    if (!dump_layers.empty()) {   // [position][layer][n_embd] for the prompt and every token fed back
        FILE* f = std::fopen(dump_layers.c_str(), "wb");
        if (!f) {
            std::fprintf(stderr, "strata-flint: cannot write %s\n", dump_layers.c_str());
            return 1;
        }
        std::fwrite(layer_dump.data(), 4, (prompt.size() + out.size() - 1) * (size_t) m.L * (size_t) m.N, f);
        std::fclose(f);
    }
    const double ms = ms_since(t1);
    std::printf("generated:");
    for (int32_t t : out) std::printf(" %d", t);
    std::printf("\n");
    std::fprintf(stderr, "strata-flint: %d tokens in %.1f ms (%.2f tok/s)\n", (int) out.size(), ms,
                 (double) (out.size() > 1 ? out.size() - 1 : 1) / (ms / 1000.0));
    return 0;
}
