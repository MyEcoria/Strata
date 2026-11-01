// src/flint/flint_kv.cuh - two optional codecs for the KV cache of the full-attention layers of Flint (included by flint_engine.cu).
//
//   --kv-quant tqt   "TurboQuant Trellis": TurboQuant's pipeline (a random rotation of each 128-wide K or V vector, its norm kept
//                    apart, one fixed codebook for the rotated samples, no calibration) with a trellis-coded quantiser in place of
//                    the scalar one: a bit-shift trellis of 256 states, k = 2..4 bits per sample, a trained table of 256 values (one
//                    per state, flint_kv_codebooks.h) and a Viterbi search per vector.  A sample costs k bits, as with a scalar
//                    code, and the error is 1.4-1.6 dB lower (tools/flint/kv_codebooks.py).
//   --kv-quant hq    "HyperQuant" (arXiv 2606.23406, without its CUDA-core/MMA fusion): the same rotation, then the Gosset lattice E8
//                    (8-wide blocks, nearest point in O(1)), the stripped lattice coordinates Rice-coded, and per vector the finest
//                    step among 32 whose code fits a fixed budget of 128 * bits bits - so a vector can still be found by its index.
//   --kv-bits 2|3|4  bits per sample (default 3) for either codec.
//
// What is stored per K or V vector: the payload (16 * bits bytes), the RMS of the vector (half) and, for hq, one byte (step index
// and Rice parameter).  5 bits per sample payload + 2-3 bytes: 3 bits = 51 bytes against 256 for FP16.  The rotation is
// R = H diag(s) / sqrt(128) (H the Walsh-Hadamard matrix, s random signs fixed by layer and K / V), so q . k = (R q) . (R k) and
// the attention reads the rotated keys directly: the query is rotated once, the values are summed rotated and un-rotated once.
//
// This file needs kHD (= 128) and block_sum from flint_engine.cu above it.
#pragma once
#include "flint_kv_codebooks.h"

constexpr int kKvG = 8;              // query heads per KV head the attention kernel handles (Flint: 32 / 4)
constexpr float kKvRs = 0.08838834764831845f;   // 1 / sqrt(128)

__device__ const float kHqDelta[32] = {
    0.150000f, 0.163072f, 0.177283f, 0.192733f, 0.209529f, 0.227789f, 0.247640f, 0.269221f, 0.292683f, 0.318190f, 0.345919f,
    0.376065f, 0.408838f, 0.444467f, 0.483201f, 0.525311f, 0.571090f, 0.620859f, 0.674965f, 0.733787f, 0.797734f, 0.867254f,
    0.942833f, 1.024998f, 1.114324f, 1.211434f, 1.317008f, 1.431781f, 1.556557f, 1.692206f, 1.839677f, 2.000000f};

/// The sign of lane d of the rotation of layer `layer` (kv 0: K, 1: V).
__device__ __forceinline__ float kv_sign(int layer, int kv, int d) {
    unsigned h = (unsigned) d * 0x9E3779B1u + (unsigned) (layer * 2 + kv + 1) * 0x85EBCA6Bu;
    h ^= h >> 15;
    h *= 0x2C1B3C6Du;
    h ^= h >> 12;
    h *= 0x297A2D39u;
    h ^= h >> 15;
    return (h & 1u) ? -1.f : 1.f;
}

/// In-place Walsh-Hadamard transform (unscaled) of `n` vectors of 128 in shared memory, by the block's `nthr` threads.
__device__ __forceinline__ void kv_wht(float (*v)[kHD], int n, int tid, int nthr) {
    for (int h = 1; h < kHD; h <<= 1) {
        for (int p = tid; p < 64 * n; p += nthr) {
            const int g = p >> 6, q = p & 63;
            const int i = ((q & ~(h - 1)) << 1) | (q & (h - 1));
            const float a = v[g][i], b = v[g][i + h];
            v[g][i] = a + b;
            v[g][i + h] = a - b;
        }
        __syncthreads();
    }
}
/// The same for one vector by one warp.
__device__ __forceinline__ void kv_wht_warp(float* v, int lane) {
    for (int h = 1; h < kHD; h <<= 1) {
        for (int q = lane; q < 64; q += 32) {
            const int i = ((q & ~(h - 1)) << 1) | (q & (h - 1));
            const float a = v[i], b = v[i + h];
            v[i] = a + b;
            v[i + h] = a - b;
        }
        __syncwarp();
    }
}

__device__ __forceinline__ unsigned kv_peek(const uint32_t* w, unsigned pos) {   // the 32 bits at bit `pos` (LSB first)
    const unsigned i = pos >> 5, sh = pos & 31;
    const unsigned lo = w[i];
    return sh ? (lo >> sh) | (w[i + 1] << (32 - sh)) : lo;
}

// ------------------------------------------------------------------------------------------------ E8
/// Nearest point of D8 + off (off 0 or 0.5) to y[8]: q = twice the point (integers), the distance.
__device__ __forceinline__ float kv_d8(const float* y, float off, int* q) {
    float f[8];
    int sum = 0, wi = 0;
    float worst = -1.f, werr = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float t = y[i] - off, r = rintf(t), e = t - r;
        f[i] = r;
        sum += (int) r;
        if (fabsf(e) > worst) {
            worst = fabsf(e);
            wi = i;
            werr = e;
        }
    }
    if (sum & 1) f[wi] += werr >= 0.f ? 1.f : -1.f;
    float d = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float e = y[i] - (f[i] + off);
        d += e * e;
        q[i] = (int) (2.f * f[i] + (off > 0.f ? 1.f : 0.f));
    }
    return d;
}
/// The E8 point nearest to y[8]: q[8] = twice its coordinates (all of one parity), then the symbols: the coset bit, s[0..6]
/// and t, with s7 = 2 t + parity(s0 + .. + s6).
__device__ __forceinline__ void kv_e8(const float* y, int& c, int* u) {
    int qa[8], qb[8];
    const float da = kv_d8(y, 0.f, qa), db = kv_d8(y, 0.5f, qb);
    const int* q = da <= db ? qa : qb;
    c = q[0] & 1;
    int sum = 0;
#pragma unroll
    for (int i = 0; i < 7; ++i) {
        const int s = (q[i] - c) >> 1;
        sum += s;
        u[i] = s >= 0 ? 2 * s : -2 * s - 1;     // zigzag
    }
    const int s7 = (q[7] - c) >> 1, t = (s7 - (sum & 1)) >> 1;
    u[7] = t >= 0 ? 2 * t : -2 * t - 1;
}
__device__ __forceinline__ int kv_rice_len(int u, int kr) { return (u >> kr) + 1 + kr; }

/// Appends `n` (<= 32) bits of `v` at bit `pos` of the shared words (which start at zero).
__device__ __forceinline__ void kv_put(unsigned* w, unsigned pos, unsigned v, unsigned n) {
    if (!n) return;
    if (n < 32) v &= (1u << n) - 1u;
    const unsigned i = pos >> 5, sh = pos & 31;
    atomicOr(&w[i], v << sh);
    if (sh && sh + n > 32) atomicOr(&w[i + 1], v >> (32 - sh));
}

/// Reads one Rice-coded symbol (zigzag value) at `pos`.
__device__ __forceinline__ unsigned kv_rice_get(const uint32_t* w, unsigned& pos, int kr) {
    unsigned q = 0, x = kv_peek(w, pos);
    while (x == 0xFFFFFFFFu) {
        q += 32;
        pos += 32;
        x = kv_peek(w, pos);
    }
    const unsigned t = (unsigned) __ffs((int) ~x) - 1u;
    q += t;
    pos += t + 1;
    const unsigned r = kr ? (kv_peek(w, pos) & ((1u << kr) - 1u)) : 0u;
    pos += kr;
    return (q << kr) | r;
}

/// The samples of an hq row, in order: f(d, value) with the value in the rotated, scaled domain.
template <class F>
__device__ __forceinline__ void kv_hq_decode(const uint32_t* row, unsigned mt, float sigma, F&& f) {
    const int kr = (int) (mt >> 5);
    const float scl = sigma * kHqDelta[mt & 31] * 0.5f;
    unsigned pos = 0;
    for (int b = 0; b < 16; ++b) {
        const int c = (int) (kv_peek(row, pos) & 1u);
        pos += 1;
        int s[8], sum = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const unsigned u = kv_rice_get(row, pos, kr);
            s[i] = (u & 1u) ? -(int) ((u + 1u) >> 1) : (int) (u >> 1);
            if (i < 7) sum += s[i];
        }
        s[7] = 2 * s[7] + (sum & 1);
#pragma unroll
        for (int i = 0; i < 8; ++i) f(b * 8 + i, scl * (float) (c + 2 * s[i]));
    }
}

/// The samples of a trellis row: the state is the last 8 bits of the stream.
template <int KB, class F>
__device__ __forceinline__ void kv_tqt_decode(const uint32_t* row, const float* tab, float sigma, F&& f) {
    unsigned S = 0;
#pragma unroll 8
    for (int i = 0; i < kHD; ++i) {
        const unsigned pos = (unsigned) (i * KB), wi = pos >> 5, sh = pos & 31;
        unsigned v = row[wi] >> sh;
        if (sh + KB > 32) v |= row[wi + 1] << (32 - sh);
        S = ((S << KB) | (v & ((1u << KB) - 1u))) & 255u;
        f(i, sigma * tab[S]);
    }
}

template <int KB>
__device__ __forceinline__ const float* kv_table() {
    return KB == 2 ? kKvTrellis2 : (KB == 3 ? kKvTrellis3 : kKvTrellis4);
}

// ------------------------------------------------------------------------------------------------ the encoders
/// The staging holds [T][HKV][128] halves of K and of V (as k_prep wrote them); this loads one vector, rotates it (signs, then
/// Hadamard), and leaves the samples divided by the RMS in xs.  Returns the RMS (0: a zero vector).  `nthr` threads, tid < 128 load.
__device__ __forceinline__ float kv_rotate_load(const __half* src, int layer, int kv, float* xs, float* red, int tid, int nthr) {
    float x = tid < kHD ? __half2float(src[tid]) * kv_sign(layer, kv, tid) : 0.f;
    const float ss = block_sum(x * x, red);
    if (tid < kHD) xs[tid] = x;
    __syncthreads();
    kv_wht(reinterpret_cast<float(*)[kHD]>(xs), 1, tid, nthr);
    const float rms = sqrtf(ss / (float) kHD);
    const float k = rms > 0.f ? kKvRs / rms : 0.f;
    if (tid < kHD) xs[tid] *= k;
    __syncthreads();
    return rms;
}

/// Trellis encoder.  grid (HKV, T, 2 = K / V), 256 threads (one per state).
template <int KB>
__global__ void __launch_bounds__(256) k_kvq_enc_tqt(const __half* kst, const __half* vst, uint8_t* kcode, uint8_t* vcode,
                                                     __half* ksc, __half* vsc, int HKV, int layer, int p0, int ncell) {
    __shared__ float xs[kHD], red[256], cost[2][256], cbs[256];
    __shared__ uint8_t back[kHD][256];
    __shared__ unsigned char sbits[kHD];
    __shared__ float bv[256];
    __shared__ int bi[256];
    const int kvh = blockIdx.x, t = blockIdx.y, kv = blockIdx.z, tid = threadIdx.x;
    const __half* src = (kv ? vst : kst) + ((size_t) t * HKV + kvh) * kHD;
    const size_t vec = (size_t) kvh * ncell + p0 + t;
    uint32_t* dst = reinterpret_cast<uint32_t*>((kv ? vcode : kcode) + vec * (16 * KB));
    cbs[tid] = kv_table<KB>()[tid];
    const float rms = kv_rotate_load(src, layer, kv, xs, red, tid, 256);
    cost[0][tid] = tid == 0 ? 0.f : 1e30f;
    __syncthreads();
    for (int i = 0; i < kHD; ++i) {
        const float e = xs[i] - cbs[tid], e2 = e * e;
        const float* cur = cost[i & 1];
        float best = 3e38f;
        int bj = 0;
#pragma unroll
        for (int j = 0; j < (1 << KB); ++j) {
            const float c = cur[(tid >> KB) | (j << (8 - KB))];
            if (c < best) {
                best = c;
                bj = j;
            }
        }
        back[i][tid] = (uint8_t) bj;
        cost[(i & 1) ^ 1][tid] = best + e2;
        __syncthreads();
    }
    bv[tid] = cost[0][tid];
    bi[tid] = tid;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s && (bv[tid + s] < bv[tid])) {
            bv[tid] = bv[tid + s];
            bi[tid] = bi[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        unsigned s = (unsigned) bi[0];
        for (int i = kHD - 1; i >= 0; --i) {
            sbits[i] = (unsigned char) (s & ((1u << KB) - 1u));
            s = (s >> KB) | ((unsigned) back[i][s] << (8 - KB));
        }
    }
    __syncthreads();
    if (tid < 4 * KB) {   // 128 * KB / 32 words
        unsigned w = 0;
        for (int b = 0; b < 32; ++b) {
            const int bit = tid * 32 + b;
            w |= (unsigned) ((sbits[bit / KB] >> (bit % KB)) & 1u) << b;
        }
        dst[tid] = w;
    }
    if (tid == 0) (kv ? vsc : ksc)[vec] = __float2half(rms);
}

/// HyperQuant encoder.  grid (HKV, T, 2), 128 threads.  `bits`: the budget is 128 * bits bits.
__global__ void __launch_bounds__(128) k_kvq_enc_hq(const __half* kst, const __half* vst, uint8_t* kcode, uint8_t* vcode, __half* ksc,
                                                    __half* vsc, uint8_t* kmt, uint8_t* vmt, int HKV, int layer, int p0, int ncell,
                                                    int bits) {
    __shared__ float xs[kHD], red[kHD];
    __shared__ int lens[32][3];
    __shared__ unsigned words[16];
    __shared__ int blen[16], boff[17], chosen[2];
    const int kvh = blockIdx.x, t = blockIdx.y, kv = blockIdx.z, tid = threadIdx.x;
    const __half* src = (kv ? vst : kst) + ((size_t) t * HKV + kvh) * kHD;
    const size_t vec = (size_t) kvh * ncell + p0 + t;
    const int pb = 16 * bits, budget = 128 * bits;
    uint32_t* dst = reinterpret_cast<uint32_t*>((kv ? vcode : kcode) + vec * pb);
    const float rms = kv_rotate_load(src, layer, kv, xs, red, tid, 128);
    for (int i = tid; i < 96; i += 128) lens[i / 3][i % 3] = 0;
    if (tid < 16) words[tid] = 0;
    __syncthreads();
    {   // the candidate steps: lane m of warp g handles 4 of the 16 blocks
        const int m = tid & 31, g = tid >> 5;
        const float inv = 1.f / kHqDelta[m];
        int l0 = 0, l1 = 0, l2 = 0;
        for (int b = g * 4; b < g * 4 + 4; ++b) {
            float y[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) y[i] = xs[b * 8 + i] * inv;
            int c, u[8];
            kv_e8(y, c, u);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                l0 += kv_rice_len(u[i], 0);
                l1 += kv_rice_len(u[i], 1);
                l2 += kv_rice_len(u[i], 2);
            }
            l0 += 1;
            l1 += 1;
            l2 += 1;
        }
        atomicAdd(&lens[m][0], l0);
        atomicAdd(&lens[m][1], l1);
        atomicAdd(&lens[m][2], l2);
    }
    __syncthreads();
    if (tid < 32) {
        const int a = lens[tid][0], b = lens[tid][1], c = lens[tid][2];
        const int kr = (b < a && b <= c) ? 1 : (c < a && c < b ? 2 : 0);
        const int L = min(a, min(b, c));
        const unsigned ok = (unsigned) __ballot_sync(0xFFFFFFFFu, L <= budget);
        const int m = ok ? (int) __ffs((int) ok) - 1 : -1;
        const int krs = __shfl_sync(0xFFFFFFFFu, kr, m < 0 ? 0 : m);
        if (tid == 0) {
            chosen[0] = m;
            chosen[1] = krs;
        }
    }
    __syncthreads();
    const int m = chosen[0], kr = chosen[1];
    int c = 0, u[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    if (m >= 0 && tid < 16) {
        const float inv = 1.f / kHqDelta[m];
        float y[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) y[i] = xs[tid * 8 + i] * inv;
        kv_e8(y, c, u);
        int l = 1;
#pragma unroll
        for (int i = 0; i < 8; ++i) l += kv_rice_len(u[i], kr);
        blen[tid] = l;
    }
    __syncthreads();
    if (tid == 0) {
        int o = 0;
        for (int b = 0; b < 16; ++b) {
            boff[b] = o;
            o += blen[b];
        }
        boff[16] = o;
    }
    __syncthreads();
    if (m >= 0 && tid < 16) {
        unsigned pos = (unsigned) boff[tid];
        kv_put(words, pos, (unsigned) c, 1);
        pos += 1;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            unsigned q = (unsigned) u[i] >> kr;
            while (q >= 32) {   // the unary part: q ones, then a zero
                kv_put(words, pos, 0xFFFFFFFFu, 32);
                pos += 32;
                q -= 32;
            }
            kv_put(words, pos, (1u << q) - 1u, q);
            pos += q + 1;
            kv_put(words, pos, (unsigned) u[i] & ((1u << kr) - 1u), (unsigned) kr);
            pos += kr;
        }
    }
    __syncthreads();
    if (tid < 4 * bits) dst[tid] = words[tid];
    if (tid == 0) {
        (kv ? vsc : ksc)[vec] = __float2half(m >= 0 ? rms : 0.f);
        (kv ? vmt : kmt)[vec] = (uint8_t) (m >= 0 ? (m | (kr << 5)) : 0);
    }
}

// ------------------------------------------------------------------------------------------------ the decoders
/// Attention part 1 on the quantised cache, for the 8 query heads of one KV head at once (each key is decoded once).  One block
/// per (chunk of 128 keys, token x KV head), 128 threads.  Same partials as k_att_part.  A full-attention layer: every key before
/// the token, no window.
template <int MODE, int KB>
__global__ void __launch_bounds__(128) k_att_part_q(const float* q, const uint8_t* kcode, const __half* ksc, const uint8_t* kmt,
                                                    const uint8_t* vcode, const __half* vsc, const uint8_t* vmt, int H, int HKV,
                                                    int p0, int ncell, float scale, float* pacc, float* pm, float* pl, int nchunk,
                                                    int layer, int bits) {
    __shared__ float qs[kKvG][kHD];
    __shared__ float ps[kKvG][kHD];
    __shared__ __half vt[kHD][kHD + 2];
    __shared__ float tab[256], sh[kHD];
    const int c = blockIdx.x, th = blockIdx.y, tid = threadIdx.x;
    const int t = th / HKV, kvh = th % HKV, G = H / HKV;
    const int pos = p0 + t, n = pos + 1, start = c * kAttChunk;
    if (start >= n) {
        if (tid < G) {
            const size_t idx = ((size_t) t * H + kvh * G + tid) * nchunk + c;
            pm[idx] = -INFINITY;
            pl[idx] = 0.f;
        }
        return;
    }
    const int cnt = min(kAttChunk, n - start), pb = 16 * bits;
    if (MODE == 1) {
        tab[tid] = kv_table<KB>()[tid];
        tab[tid + 128] = kv_table<KB>()[tid + 128];
    }
    const float ksg = kv_sign(layer, 0, tid), vsg = kv_sign(layer, 1, tid);
    for (int g = 0; g < G; ++g) qs[g][tid] = q[((size_t) t * H + kvh * G + g) * kHD + tid] * ksg;
    __syncthreads();
    kv_wht(qs, G, tid, 128);
    for (int g = 0; g < G; ++g) qs[g][tid] *= kKvRs * scale;
    __syncthreads();
    float s[kKvG];
#pragma unroll
    for (int g = 0; g < kKvG; ++g) s[g] = 0.f;
    if (tid < cnt) {
        const size_t vec = (size_t) kvh * ncell + start + tid;
        const uint32_t* row = reinterpret_cast<const uint32_t*>(kcode + vec * pb);
        const float sig = __half2float(ksc[vec]);
        auto f = [&](int d, float val) {
#pragma unroll
            for (int g = 0; g < kKvG; ++g) s[g] += qs[g][d] * val;
        };
        if (MODE == 1) kv_tqt_decode<KB>(row, tab, sig, f);
        else kv_hq_decode(row, kmt[vec], sig, f);
    }
    float lsum[kKvG], mx[kKvG];
#pragma unroll
    for (int g = 0; g < kKvG; ++g) {
        if (g >= G) break;
        mx[g] = block_max(tid < cnt ? s[g] : -INFINITY, sh);
        const float p = tid < cnt ? expf(s[g] - mx[g]) : 0.f;
        ps[g][tid] = p;
        lsum[g] = block_sum(p, sh);
    }
    // the values: decode a tile of 128 rows (a thread a row), then a thread a lane sums them
    if (tid < cnt) {
        const size_t vec = (size_t) kvh * ncell + start + tid;
        const uint32_t* row = reinterpret_cast<const uint32_t*>(vcode + vec * pb);
        const float sig = __half2float(vsc[vec]);
        auto f = [&](int d, float val) { vt[tid][d] = __float2half(val); };
        if (MODE == 1) kv_tqt_decode<KB>(row, tab, sig, f);
        else kv_hq_decode(row, vmt[vec], sig, f);
    } else {
        for (int d = 0; d < kHD; ++d) vt[tid][d] = __float2half(0.f);
    }
    __syncthreads();
    float acc[kKvG];
#pragma unroll
    for (int g = 0; g < kKvG; ++g) acc[g] = 0.f;
    for (int j = 0; j < cnt; ++j) {
        const float v = __half2float(vt[j][tid]);
#pragma unroll
        for (int g = 0; g < kKvG; ++g) acc[g] += ps[g][j] * v;
    }
    __syncthreads();
#pragma unroll
    for (int g = 0; g < kKvG; ++g) qs[g][tid] = acc[g];   // (qs is free now: the output, still rotated)
    __syncthreads();
    kv_wht(qs, G, tid, 128);
    for (int g = 0; g < G; ++g) {
        const size_t idx = ((size_t) t * H + kvh * G + g) * nchunk + c;
        pacc[idx * kHD + tid] = qs[g][tid] * kKvRs * vsg;
        if (tid == 0) {
            pm[idx] = mx[g];
            pl[idx] = lsum[g];
        }
    }
}

/// The FP16 copy of keys [k0, k0 + n) of one layer, un-rotated, for the matrix-core attention of a long prompt:
/// out[kv][head][cell][128] with the cache's own strides.  grid (ceil(n / 4), HKV, 2), 128 threads: a warp a vector.
template <int MODE, int KB>
__global__ void __launch_bounds__(128) k_kvq_dequant(const uint8_t* kcode, const __half* ksc, const uint8_t* kmt, const uint8_t* vcode,
                                                     const __half* vsc, const uint8_t* vmt, __half* kout, __half* vout, int ncell,
                                                     int k0, int n, int layer, int bits) {
    __shared__ float vals[4][kHD];
    __shared__ float tab[256];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, kvh = blockIdx.y, kv = blockIdx.z;
    const int cell = k0 + blockIdx.x * 4 + warp;
    if (MODE == 1) {
        for (int i = threadIdx.x; i < 256; i += 128) tab[i] = kv_table<KB>()[i];
        __syncthreads();
    }
    if (cell >= k0 + n) return;
    const size_t vec = (size_t) kvh * ncell + cell;
    const int pb = 16 * bits;
    const uint32_t* row = reinterpret_cast<const uint32_t*>((kv ? vcode : kcode) + vec * pb);
    const float sig = __half2float((kv ? vsc : ksc)[vec]);
    if (lane == 0) {
        float* v = vals[warp];
        auto f = [&](int d, float val) { v[d] = val; };
        if (MODE == 1) kv_tqt_decode<KB>(row, tab, sig, f);
        else kv_hq_decode(row, (kv ? vmt : kmt)[vec], sig, f);
    }
    __syncwarp();
    kv_wht_warp(vals[warp], lane);
    __half* out = (kv ? vout : kout) + vec * kHD;
    for (int d = lane; d < kHD; d += 32) out[d] = __float2half(vals[warp][d] * kKvRs * kv_sign(layer, kv, d));
}

// ------------------------------------------------------------------------------------------------ a check (STRATA_FLINT_KV_CHECK=1)
__device__ double g_kv_chk[4];   // sum of squares of the K originals, of the K errors, of the V originals, of the V errors
/// Original (staging) against the decoded FP16 copy of the same vectors.  grid (HKV, T, 2), 128 threads.
__global__ void k_kv_err(const __half* kst, const __half* vst, const __half* k16, const __half* v16, int HKV, int ncell, int p0) {
    __shared__ float sh[kHD];
    const int kvh = blockIdx.x, t = blockIdx.y, kv = blockIdx.z, d = threadIdx.x;
    const float a = __half2float((kv ? vst : kst)[((size_t) t * HKV + kvh) * kHD + d]);
    const float b = __half2float((kv ? v16 : k16)[((size_t) kvh * ncell + p0 + t) * kHD + d]);
    const float s0 = block_sum(a * a, sh), s1 = block_sum((a - b) * (a - b), sh);
    if (d == 0) {
        atomicAdd(&g_kv_chk[kv * 2], (double) s0);
        atomicAdd(&g_kv_chk[kv * 2 + 1], (double) s1);
    }
}
