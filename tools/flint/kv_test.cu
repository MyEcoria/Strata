// tools/flint/kv_test.cu - the KV codecs of src/flint/flint_kv.cuh on their own: no model, no files.  Needs a GPU.
//
//   build:  see tools/flint/kv_test.sh
//   run:    ./kv_test [keys]          (default 8192; 32768 shows the speed at the default context of a server)
//
// For every codec and bit width: K and V vectors with a few outlier channels (as real keys have) are encoded and decoded; it prints
// the signal / error of the decoded vectors, the largest error of the attention output of 32 query heads (the new kernel, which
// reads the codes) against a double-precision attention over the same decoded vectors (so only the kernel is tested, not the
// codec), and the time of that kernel against the FP16 kernel of the engine on the same number of keys.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CK(x)                                                                              \
    do {                                                                                   \
        cudaError_t e_ = (x);                                                              \
        if (e_ != cudaSuccess) {                                                           \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
            std::exit(1);                                                                  \
        }                                                                                  \
    } while (0)

constexpr int kHD = 128, kAttChunk = 128, H = 32, HKV = 4;

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
#include "../../src/flint/flint_kv.cuh"

// the engine's FP16 attention kernel (k_att_part of flint_engine.cu), for the timing
__global__ void k_att_part(const float* q, const __half* kc, const __half* vc, int H, int HKV, int p0, int window, int ring, int ncell,
                           float scale, float* pacc, float* pm, float* pl, int nchunk) {
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

template <class T>
T* dev(size_t n) {
    T* p = nullptr;
    CK(cudaMalloc(&p, n * sizeof(T)));
    CK(cudaMemset(p, 0, n * sizeof(T)));
    return p;
}

int main(int argc, char** argv) {
    const int N = argc > 1 ? std::atoi(argv[1]) : 8192;   // keys
    const int layer = 5, STG = 256;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    // K: 128-wide vectors with 6 outlier channels (x8); V: plain
    std::vector<__half> hk((size_t) N * HKV * kHD), hv(hk.size());   // [pos][kvh][128]
    for (int i = 0; i < N * HKV; ++i)
        for (int d = 0; d < kHD; ++d) {
            const float o = (d % 21 == 3) ? 8.f : 1.f;
            hk[(size_t) i * kHD + d] = __float2half(nd(rng) * o);
            hv[(size_t) i * kHD + d] = __float2half(nd(rng) * 0.7f);
        }
    std::vector<float> hq((size_t) H * kHD), sinks(H);
    for (auto& v : hq) v = nd(rng) * 0.5f;
    for (auto& v : sinks) v = nd(rng);
    __half *dk = dev<__half>(hk.size()), *dv = dev<__half>(hv.size());
    CK(cudaMemcpy(dk, hk.data(), hk.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dv, hv.data(), hv.size() * 2, cudaMemcpyHostToDevice));
    float* dq = dev<float>(hq.size());
    CK(cudaMemcpy(dq, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice));
    const int nchunk = (N + kAttChunk - 1) / kAttChunk;
    float *pacc = dev<float>((size_t) nchunk * H * kHD), *pm = dev<float>((size_t) nchunk * H), *pl = dev<float>((size_t) nchunk * H);
    // the FP16 layout the engine's FP16 kernel reads: [kvh][cell][128]
    __half *k16 = dev<__half>((size_t) HKV * N * kHD), *v16 = dev<__half>((size_t) HKV * N * kHD);
    {
        std::vector<__half> a((size_t) HKV * N * kHD), b(a.size());
        for (int p = 0; p < N; ++p)
            for (int h = 0; h < HKV; ++h)
                for (int d = 0; d < kHD; ++d) {
                    a[((size_t) h * N + p) * kHD + d] = hk[((size_t) p * HKV + h) * kHD + d];
                    b[((size_t) h * N + p) * kHD + d] = hv[((size_t) p * HKV + h) * kHD + d];
                }
        CK(cudaMemcpy(k16, a.data(), a.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(v16, b.data(), b.size() * 2, cudaMemcpyHostToDevice));
    }
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    const float scale = 1.0f / std::sqrt((float) kHD);
    auto time_ms = [&](auto&& launch) {
        for (int i = 0; i < 3; ++i) launch();
        CK(cudaEventRecord(e0));
        const int R = 20;
        for (int i = 0; i < R; ++i) launch();
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        return ms / R;
    };
    // the FP16 kernel, one token at the last position
    const double t16 = time_ms([&] {
        k_att_part<<<dim3(nchunk, H), kHD>>>(dq, k16, v16, H, HKV, N - 1, 0, 0, N, scale, pacc, pm, pl, nchunk);
    });
    CK(cudaGetLastError());
    std::printf("%d keys, 32 query heads / 4 KV heads, one token: FP16 kernel %.3f ms (reads %.1f MiB)\n", N, t16,
                2.0 * HKV * N * kHD * 2 / (1 << 20));

    __half *kst = dev<__half>((size_t) STG * HKV * kHD), *vst = dev<__half>((size_t) STG * HKV * kHD);
    __half *k16q = dev<__half>((size_t) HKV * N * kHD), *v16q = dev<__half>((size_t) HKV * N * kHD);
    int fails = 0;
    for (int mode = 1; mode <= 2; ++mode)
        for (int bits = 2; bits <= 4; ++bits) {
            const size_t nvec = (size_t) HKV * N, pbytes = nvec * 16 * bits + 64;
            uint8_t *kc = dev<uint8_t>(pbytes), *vc = dev<uint8_t>(pbytes), *kmt = dev<uint8_t>(nvec), *vmt = dev<uint8_t>(nvec);
            __half *ksc = dev<__half>(nvec), *vsc = dev<__half>(nvec);
            for (int p0 = 0; p0 < N; p0 += STG) {
                const int T = std::min(STG, N - p0);
                CK(cudaMemcpy(kst, hk.data() + (size_t) p0 * HKV * kHD, (size_t) T * HKV * kHD * 2, cudaMemcpyHostToDevice));
                CK(cudaMemcpy(vst, hv.data() + (size_t) p0 * HKV * kHD, (size_t) T * HKV * kHD * 2, cudaMemcpyHostToDevice));
                const dim3 g(HKV, T, 2);
                if (mode == 1) {
                    if (bits == 2) k_kvq_enc_tqt<2><<<g, 256>>>(kst, vst, kc, vc, ksc, vsc, HKV, layer, p0, N);
                    if (bits == 3) k_kvq_enc_tqt<3><<<g, 256>>>(kst, vst, kc, vc, ksc, vsc, HKV, layer, p0, N);
                    if (bits == 4) k_kvq_enc_tqt<4><<<g, 256>>>(kst, vst, kc, vc, ksc, vsc, HKV, layer, p0, N);
                } else {
                    k_kvq_enc_hq<<<g, 128>>>(kst, vst, kc, vc, ksc, vsc, kmt, vmt, HKV, layer, p0, N, bits);
                }
            }
            CK(cudaGetLastError());
            const dim3 gd((N + 3) / 4, HKV, 2);
#define DQ(M, B) k_kvq_dequant<M, B><<<gd, 128>>>(kc, ksc, kmt, vc, vsc, vmt, k16q, v16q, N, 0, N, layer, bits)
            if (mode == 1) {
                if (bits == 2) DQ(1, 2);
                if (bits == 3) DQ(1, 3);
                if (bits == 4) DQ(1, 4);
            } else {
                DQ(2, 4);
            }
            CK(cudaDeviceSynchronize());
            std::vector<__half> dk16((size_t) HKV * N * kHD), dv16(dk16.size());
            CK(cudaMemcpy(dk16.data(), k16q, dk16.size() * 2, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(dv16.data(), v16q, dv16.size() * 2, cudaMemcpyDeviceToHost));
            double sk = 0, ek = 0, sv = 0, ev = 0;
            for (int p = 0; p < N; ++p)
                for (int h = 0; h < HKV; ++h)
                    for (int d = 0; d < kHD; ++d) {
                        const float a = __half2float(hk[((size_t) p * HKV + h) * kHD + d]), b = __half2float(dk16[((size_t) h * N + p) * kHD + d]);
                        const float c = __half2float(hv[((size_t) p * HKV + h) * kHD + d]), e = __half2float(dv16[((size_t) h * N + p) * kHD + d]);
                        sk += a * a, ek += (a - b) * (a - b), sv += c * c, ev += (c - e) * (c - e);
                    }
            // the kernel on the codes, one token at the last position
            auto launch = [&] {
                const dim3 g(nchunk, HKV);
#define ATT(M, B) k_att_part_q<M, B><<<g, 128>>>(dq, kc, ksc, kmt, vc, vsc, vmt, H, HKV, N - 1, N, scale, pacc, pm, pl, nchunk, layer, bits)
                if (mode == 1) {
                    if (bits == 2) ATT(1, 2);
                    if (bits == 3) ATT(1, 3);
                    if (bits == 4) ATT(1, 4);
                } else {
                    ATT(2, 4);
                }
            };
            launch();
            CK(cudaDeviceSynchronize());
            CK(cudaGetLastError());
            std::vector<float> opacc((size_t) nchunk * H * kHD), opm((size_t) nchunk * H), opl(opm.size());
            CK(cudaMemcpy(opacc.data(), pacc, opacc.size() * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(opm.data(), pm, opm.size() * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(opl.data(), pl, opl.size() * 4, cudaMemcpyDeviceToHost));
            double worst = 0, ref_norm = 0;
            for (int h = 0; h < H; ++h) {   // merge the chunks like k_att_merge, against a direct double attention
                const int kvh = h / (H / HKV);
                double M = sinks[h];
                for (int c = 0; c < nchunk; ++c) M = std::max<double>(M, opm[(size_t) h * nchunk + c]);
                double den = std::exp(sinks[h] - M);
                std::vector<double> num(kHD, 0.0);
                for (int c = 0; c < nchunk; ++c) {
                    const size_t idx = (size_t) h * nchunk + c;
                    if (opm[idx] == -INFINITY) continue;
                    const double w = std::exp((double) opm[idx] - M);
                    den += opl[idx] * w;
                    for (int d = 0; d < kHD; ++d) num[d] += opacc[idx * kHD + d] * w;
                }
                std::vector<double> sc(N);
                double mx = sinks[h];
                for (int p = 0; p < N; ++p) {
                    double a = 0;
                    for (int d = 0; d < kHD; ++d) a += (double) hq[(size_t) h * kHD + d] * __half2float(dk16[((size_t) kvh * N + p) * kHD + d]);
                    sc[p] = a * scale;
                    mx = std::max(mx, sc[p]);
                }
                double dn = std::exp(sinks[h] - mx);
                std::vector<double> ref(kHD, 0.0);
                for (int p = 0; p < N; ++p) {
                    const double w = std::exp(sc[p] - mx);
                    dn += w;
                    for (int d = 0; d < kHD; ++d) ref[d] += w * __half2float(dv16[((size_t) kvh * N + p) * kHD + d]);
                }
                for (int d = 0; d < kHD; ++d) {
                    worst = std::max(worst, std::fabs(num[d] / den - ref[d] / dn));
                    ref_norm = std::max(ref_norm, std::fabs(ref[d] / dn));
                }
            }
            const double tq = time_ms(launch);
            const bool ok = worst < 5e-3 * ref_norm;
            if (!ok) ++fails;
            std::printf("%-3s %d bits: decode K %.2f dB, V %.2f dB | attention vs double over the same decoded vectors: max |diff| %.1e (max |out| %.2f) %s"
                        " | kernel %.3f ms (%.2fx the FP16 kernel)\n",
                        mode == 1 ? "tqt" : "hq", bits, 10 * std::log10(sk / ek), 10 * std::log10(sv / ev), worst, ref_norm,
                        ok ? "PASS" : "FAIL", tq, tq / t16);
            cudaFree(kc), cudaFree(vc), cudaFree(kmt), cudaFree(vmt), cudaFree(ksc), cudaFree(vsc);
        }
    std::printf(fails ? "SOME FAILED\n" : "ALL PASS\n");
    return fails ? 1 : 0;
}
