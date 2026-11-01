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

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace K = strata::kernels;

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
unsigned rows8(int n) { return (unsigned) ((n + 7) / 8); }
unsigned grid1(int64_t n, int b = kThreads) { return (unsigned) ((n + b - 1) / b); }

/// Q / K / V of T tokens at positions p0 .. p0 + T - 1: the per-head RMS norm of Q and K, NEOX rotary embedding when
/// `rope`, Q out as f32, K and V into the FP16 cache [HKV][ncell][128] (cell = position % ring when ring > 0, else the
/// position).  Grid (H + HKV, T), 128 threads.
__global__ void k_prep(const float* qx, const float* kx, const float* vx, const float* qw, const float* kw, float* qout,
                       __half* kc, __half* vc, int H, int HKV, int p0, int rope, float base, float eps, int ring, int ncell) {
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

/// The router: p = sigmoid(logit) for each of the n_expert experts, the top-k by p (ties: the lower index), their
/// weights p / max(sum, 6.1035156e-5) * scale.  One block per token, one thread does the (tiny) selection.
__global__ void k_route(const float* logits, int n_expert, int k, float scale, int32_t* ids, float* wts) {
    __shared__ float p[512];
    const float* lr = logits + (size_t) blockIdx.x * n_expert;
    for (int i = threadIdx.x; i < n_expert; i += blockDim.x) p[i] = 1.0f / (1.0f + expf(-lr[i]));
    __syncthreads();
    if (threadIdx.x != 0) return;
    float sel_w[kMaxK];
    float sum = 0.f;
    for (int j = 0; j < k; ++j) {
        int best = -1;
        float bv = -INFINITY;
        for (int i = 0; i < n_expert; ++i) {
            bool taken = false;
            for (int q = 0; q < j; ++q) taken |= ids[(size_t) blockIdx.x * k + q] == i;
            if (!taken && p[i] > bv) {
                bv = p[i];
                best = i;
            }
        }
        ids[(size_t) blockIdx.x * k + j] = best;
        sel_w[j] = bv;
        sum += bv;
    }
    const float denom = fmaxf(sum, 6.103515625e-5f);
    for (int j = 0; j < k; ++j) wts[(size_t) blockIdx.x * k + j] = sel_w[j] / denom * scale;
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
    int Sh = 0;
    mutable std::vector<int> hexp_slot, hslot_exp;
    mutable std::vector<long> hslot_use;
};

struct Model {
    std::shared_ptr<strata::GgufModel> file;
    std::string file_path;                     // the one file the expert reads go to
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
        uint8_t* d = nullptr;
        CK(cudaMalloc(&d, nb));
        CK(cudaMemcpy(d, gm.shard(s).tensor_data(*t), nb, cudaMemcpyHostToDevice));
        m.embd = d;
        bytes += nb;
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
    // streaming
    long clock_ = 0;
    bool pin_host = true;           // --no-pin-host: do not page-lock the host tier (then the swaps are staged and synchronous)
    double host_gib = -1;           // --host-cache: GiB of RAM for the second tier (< 0: what is free less a margin)
    uint8_t* swap_tmp = nullptr;    // pinned: a VRAM victim on its way to the host tier
    long host_hits_ = 0, ssd_reads_ = 0, demoted_ = 0, pf_loaded_ = 0;
    cudaStream_t st2 = nullptr;     // copies of the lookahead, beside the compute on st
    cudaStream_t cs = nullptr;      // the stream the expert copies of the moment go on (st, or st2 during the lookahead)
    cudaEvent_t ev = nullptr;
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
    bool buffered = false;
    // --prof: where the decode time goes, and how well an LRU of experts per layer would do (simulated)
    bool prof = false;
    double t_sync = 0, t_read = 0, t_up = 0, t_fwd = 0;
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
        for (const Layer& L : m->layers) {
            const int nc = L.swa ? kRing : ctx;
            ncell.push_back(nc);
            const size_t n = (size_t) m->HKV * nc * kHD;
            __half *k_ = nullptr, *v_ = nullptr;
            CK(cudaMalloc(&k_, n * 2));
            CK(cudaMalloc(&v_, n * 2));
            CK(cudaMemset(k_, 0, n * 2));
            CK(cudaMemset(v_, 0, n * 2));
            kc.push_back(k_);
            vc.push_back(v_);
            kv_bytes += 2 * n * 2;
        }
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
        CK(cudaHostAlloc((void**) &pin, pin_cap * 3 * (size_t) pin_slots, cudaHostAllocDefault));
        fd = buffered ? -1 : open(M.file_path.c_str(), O_RDONLY | O_DIRECT);
        if (fd < 0) fd = open(M.file_path.c_str(), O_RDONLY);
        if (fd < 0) {
            std::fprintf(stderr, "strata-flint: cannot open %s\n", M.file_path.c_str());
            std::exit(1);
        }
        pool = std::make_unique<Pool>(7);   // plus the calling thread
    }

    /// Reads the three parts (gate, up, down) of experts[0 .. n) of layer L into the pinned buffer (n <= pin_slots);
    /// part p of expert k is then at part_ptr(L, experts[k], k, p).
    void read_parts(const Layer& L, const int32_t* experts, int n) {
        pool->run(n * 3, [&](int j) {
            const int k = j / 3, p = j % 3;
            const uint64_t e = (uint64_t) experts[k];
            const size_t len = p == 2 ? L.src_dsz : L.src_gsz;
            const uint64_t off = (p == 0 ? L.off_g : p == 1 ? L.off_u : L.off_d) + e * len;
            const uint64_t aoff = off & ~(uint64_t) 4095;
            const size_t want = (size_t) (off - aoff) + len;
            const size_t alen = round_up(want, 4096);
            uint8_t* dst = pin + (size_t) j * pin_cap;
            size_t got = 0;
            while (got < want) {
                const ssize_t r = pread(fd, dst + got, alen - got, (off_t) (aoff + got));
                if (r <= 0) {
                    std::fprintf(stderr, "strata-flint: read of expert %llu failed (%zd, got %zu of %zu)\n",
                                 (unsigned long long) e, r, got, want);
                    std::exit(1);
                }
                got += (size_t) r;
            }
        });
    }
    const uint8_t* part_ptr(const Layer& L, int32_t e, int k, int p) const {
        const size_t len = p == 2 ? L.src_dsz : L.src_gsz;
        const uint64_t off = (p == 0 ? L.off_g : p == 1 ? L.off_u : L.off_d) + (uint64_t) e * len;
        return pin + (size_t) (k * 3 + p) * pin_cap + (size_t) (off & 4095);
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
            unsigned long long kb = 0;
            while (std::fgets(line, sizeof line, f))
                if (std::sscanf(line, "MemAvailable: %llu kB", &kb) == 1) break;
            std::fclose(f);
            const size_t avail = (size_t) kb * 1024, margin = (size_t) (3.5 * (double) (1ull << 30));
            host_bytes = avail > margin ? avail - margin : 0;
        }
        size_t host_used = 0;
        int min_h = 1 << 30, max_h = 0;
        if (host_bytes > 0) {
            const size_t per_h = host_bytes / (size_t) n_moe;
            size_t total = 0;
            std::vector<int> sh(M.layers.size(), 0);
            for (size_t l = 0; l < M.layers.size(); ++l) {
                const Layer& L = M.layers[l];
                if (L.dense) continue;
                sh[l] = (int) std::min<size_t>((size_t) M.NE - (size_t) L.S, per_h / L.stride);
                total += (size_t) sh[l] * L.stride;
            }
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
                        W.hslot_exp.assign((size_t) W.Sh, -1);
                        W.hslot_use.assign((size_t) W.Sh, 0);
                        off += (size_t) W.Sh * L.stride;
                        min_h = std::min(min_h, W.Sh);
                        max_h = std::max(max_h, W.Sh);
                    }
                    host_used = total;
                    if (pin_host) {   // page-locked: the copies to the GPU skip the driver's staging buffer
                        const auto tr = Clock::now();
                        const cudaError_t e = cudaHostRegister(base, total, cudaHostRegisterPortable);
                        if (e == cudaSuccess) {
                            host_pinned = true;
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
            if (!ssd.empty()) read_parts(L, ssd.data(), (int) ssd.size());
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

    /// The attention block of layer li for T tokens at positions p0 ..: x += wo(attn(norm(x))).  h holds norm(x).
    void attention(const Layer& L, size_t li, int T, int p0) {
        const Model& M = *m;
        const int H = (int) M.H, HKV = (int) M.HKV, N = (int) M.N;
        quant(h, N, T, xq);
        mm(L.wq, xq, qb, T);
        mm(L.wk, xq, kb, T);
        mm(L.wv, xq, vb, T);
        const int ring = L.swa ? kRing : 0;
        k_prep<<<dim3(H + HKV, T), kHD, 0, st>>>(qb, kb, vb, L.q_norm, L.k_norm, qr, kc[li], vc[li], H, HKV, p0, L.rope ? 1 : 0,
                                                 M.rope_base, M.eps, ring, ncell[li]);
        const int window = L.swa ? (int) M.window : 0;
        for (int t0 = 0; t0 < T; t0 += kQB) {   // query blocks: the partials are sized for kQB queries
            const int nt = std::min(kQB, T - t0);
            const int pb = p0 + t0, last = pb + nt - 1;
            const int n_max = window > 0 ? std::min(last + 1, window) : last + 1;
            const int nchunk = (n_max + kAttChunk - 1) / kAttChunk;
            k_att_part<<<dim3(nchunk, nt * H), kHD, 0, st>>>(qr + (size_t) t0 * H * kHD, kc[li], vc[li], H, HKV, pb, window,
                                                             ring, ncell[li], 1.0f / sqrtf((float) kHD), pacc, pm, pl, nchunk);
            k_att_merge<<<nt * H, kHD, 0, st>>>(pacc, pm, pl, L.sinks, H, nchunk, att + (size_t) t0 * H * kHD);
        }
        quant(att, H * kHD, T, q8);
        mm(L.wo, q8, y, T);
        k_add<<<grid1((int64_t) T * N), kThreads, 0, st>>>(x, y, (int64_t) T * N);
    }

    /// T tokens (host ids) at positions p0 ..; with `want_logits` the last row's logits are left in lg.
    void forward(const int32_t* toks, int T, int p0, bool want_logits) {
        const Model& M = *m;
        const auto tf0 = Clock::now();
        const int N = (int) M.N;
        CK(cudaMemcpyAsync(tok, toks, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        k_embed<<<T, kThreads, 0, st>>>(tok, M.embd, M.embd_row, M.embd_type, x, N);
        for (size_t l = 0; l < M.layers.size(); ++l) {
            const Layer& L = M.layers[l];
            k_rms<<<T, kThreads, 0, st>>>(x, L.attn_norm, h, N, M.eps);
            attention(L, l, T, p0);
            k_rms<<<T, kThreads, 0, st>>>(x, L.ffn_norm, h, N, M.eps);
            quant(h, N, T, xq);
            ffn(L, l, T);
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

}  // namespace

int strata::flint::run(int argc, char** argv) {
    std::string path, dump, dump_layers;
    std::vector<int32_t> prompt, forced;
    int n_new = 16, ctx = 4096, bench = 0;
    double expert_vram = -1;
    bool trace = false, buffered = false, prof = false;
    int chunk = 512;
    double host_gib = -1;
    bool pin_host = true, prof_predict = false;
    int prefetch_k = 0;   // the lookahead (--prefetch N): measured no faster than the page-locked host tier alone
    std::string nll_file;
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
        else if (a == "--tokens") prompt = parse_ids(next());
        else if (a == "--n" || a == "--max-new") n_new = std::atoi(next());
        else if (a == "--dump-full-logits") dump = next();
        else if (a == "--max-context") ctx = std::atoi(next());
        else if (a == "--expert-vram") expert_vram = std::atof(next());
        else if (a == "--bench-tg") bench = std::atoi(next());
        else if (a == "--trace") trace = true;
        else if (a == "--buffered") buffered = true;
        else if (a == "--prof") prof = true;
        else if (a == "--chunk") chunk = std::atoi(next());
        else if (a == "--host-cache") host_gib = std::atof(next());
        else if (a == "--pin-host") pin_host = true;
        else if (a == "--no-pin-host") pin_host = false;
        else if (a == "--prefetch") prefetch_k = std::atoi(next());
        else if (a == "--prof-predict") prof_predict = true;
        else if (a == "--nll") nll_file = next();
        else if (a == "--dump-layers") dump_layers = next();
        else if (a == "--force") forced = parse_ids(next());
        else if (a == "--help" || a == "-h") {
            std::printf("strata-flint --model FILE --tokens 1,2,3 [options]\n"
                        "  --n N                 tokens to generate (greedy), default 16\n"
                        "  --force IDS           teacher forcing: feed these ids instead of the greedy ones\n"
                        "  --max-context N       cells of the KV cache of the full-attention layers (default: prompt + new tokens)\n"
                        "  --chunk N             prompt tokens per forward, 8..512 (default 512)\n"
                        "  --expert-vram GiB     VRAM for the expert cache (default: what is free after a warmup, less 0.6 GiB)\n"
                        "  --host-cache GiB      host RAM for the expert tier below it (default: MemAvailable less 3.5 GiB)\n"
                        "  --no-pin-host         do not page-lock the host tier (slower: its swaps are staged and synchronous)\n"
                        "  --prefetch N          lookahead: bring N guessed experts of the next layer in early (default 0, no gain measured)\n"
                        "  --buffered            read the file through the page cache instead of O_DIRECT (slower)\n"
                        "  --bench-tg N          generate N tokens and report the speed\n"
                        "  --dump-full-logits F  the logits of every step (f32 rows), for tools/flint/compare.py\n"
                        "  --dump-layers F       the residual after every layer of every position (f32)\n"
                        "  --nll F               -log p(next prompt token) of every prompt position (f32), for a perplexity check\n"
                        "  --prof, --prof-predict, --trace   timing, lookahead-guess and per-layer diagnostics\n");
            return 0;
        } else {
            std::fprintf(stderr, "strata-flint: unknown argument %s\n", a.c_str());
            return 2;
        }
    }
    if (path.empty() || prompt.empty()) {
        std::fprintf(stderr, "strata-flint: --model FILE and --tokens IDS are needed (see --help)\n");
        return 2;
    }
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
    if ((int) prompt.size() + n_new + 8 > ctx) ctx = (int) prompt.size() + n_new + 8;
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
    E.tmax = std::max(kMaxT, std::min(chunk, 512));   // (a SWA ring of 1024 cells holds a chunk of 512 plus the window)
    std::vector<float> layer_dump;
    if (!dump_layers.empty()) {
        const size_t steps = forced.empty() ? (size_t) std::max(bench, n_new) : forced.size() + 1;
        layer_dump.assign((prompt.size() + steps) * (size_t) m.L * (size_t) m.N, 0.f);
        E.dump = &layer_dump;
    }
    E.init(m, ctx);
    E.init_staging();
    {
        // The library's kernels load into VRAM on first use: run every shape once (one token, a chunk of 8) with all the
        // experts streamed, so the memory that is left is the memory that is really free
        std::vector<float>* nll_keep = E.nll;
        E.nll = nullptr;   // (the warmup is not the prompt)
        std::vector<int32_t> wt((size_t) 24, 0);
        E.forward(wt.data(), 1, 0, true);
        E.forward(wt.data(), std::min(kMaxT, E.tmax), 0, true);
        if (E.tmax > kMaxT) E.forward(wt.data(), std::min(24, E.tmax), 0, true);
        E.nll = nll_keep;
        CK(cudaStreamSynchronize(E.st));
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        const size_t reserve = (size_t) (0.6 * (1 << 30));
        const size_t budget = expert_vram >= 0 ? (size_t) (expert_vram * (double) (1 << 30)) : (fr > reserve ? fr - reserve : 0);
        E.place_experts(budget);
    }
    // the prompt, in chunks
    const auto t0 = Clock::now();
    const int np = (int) prompt.size();
    for (int c0 = 0; c0 < np; c0 += E.tmax) {
        const int T = std::min(E.tmax, np - c0);
        E.forward(prompt.data() + c0, T, c0, c0 + T == np);
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
    if (prof && E.n_fwd > 0) {
        const double f = (double) E.n_fwd;
        std::fprintf(stderr, "prof: %ld decode steps, %.1f ms each; per step streamed-layer waits: ids D2H+sync %.1f ms, "
                     "reads %.1f ms, upload+sync %.1f ms (%.1f streamed layers/step)\n", E.n_fwd, E.t_fwd / f, E.t_sync / f,
                     E.t_read / f, E.t_up / f, (double) E.n_calls / f);
        std::fprintf(stderr, "prof: VRAM misses: %ld from the host tier, %ld read from the SSD; %ld experts demoted to the host tier; %ld loaded ahead by the lookahead\n",
                     E.host_hits_, E.ssd_reads_, E.demoted_, E.pf_loaded_);
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
