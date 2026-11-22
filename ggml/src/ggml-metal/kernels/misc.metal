#include "common.h"

kernel void kernel_argmax_f32(
        constant ggml_metal_kargs_argmax & args,
        device   const char * src0,
        device         char * dst,
        threadgroup    char * shmem [[threadgroup(0)]],
        uint  tgpig[[threadgroup_position_in_grid]],
        uint  tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint    ntg[[threads_per_threadgroup]]) {
    device const float * x_row = (device const float *) ((device const char *) src0 + tgpig * args.nb01);

    float   lmax = -INFINITY;
    int32_t larg = -1;

    for (int i00 = tpitg; i00 < args.ne00; i00 += ntg) {
        if (x_row[i00] > lmax) {
            lmax = x_row[i00];
            larg = i00;
        }
    }

    // find the argmax value in the block
    float max_val = simd_max(lmax);
    int32_t arg_val = simd_max(select(-1, larg, lmax == max_val));

    device int32_t * dst_i32 = (device int32_t *) dst;

    threadgroup   float * shared_maxval = (threadgroup   float *) shmem;
    threadgroup int32_t * shared_argmax = (threadgroup int32_t *) shmem + N_SIMDWIDTH;

    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            shared_maxval[tiisg] = -INFINITY;
            shared_argmax[tiisg] = -1;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            shared_maxval[sgitg] = max_val;
            shared_argmax[sgitg] = arg_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = shared_maxval[tiisg];
        arg_val = shared_argmax[tiisg];

        float max_val_reduced   = simd_max(max_val);
        int32_t arg_val_reduced = simd_max(select(-1, arg_val, max_val == max_val_reduced));

        dst_i32[tgpig] = arg_val_reduced;

        return;
    }

    dst_i32[tgpig] = arg_val;
}

kernel void kernel_argmax_part_f32(
        constant ggml_metal_kargs_argmax & args,
        device   const char * src0,
        device         char * tmp,
        threadgroup    char * shmem [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint3  ntg[[threads_per_threadgroup]]) {
    const int row  = tgpig.y;
    const int part = tgpig.x;
    const int ne00 = (int) args.ne00;
    const int npart = (int) args.npart;
    const int chunk = (ne00 + npart - 1) / npart;
    const int start = part * chunk;
    const int end   = min(start + chunk, ne00);

    device const float * x_row = (device const float *) ((device const char *) src0 + row * args.nb01);

    float   lmax = -INFINITY;
    int32_t larg = -1;

    for (int i00 = start + tpitg.x; i00 < end; i00 += ntg.x) {
        if (x_row[i00] > lmax) {
            lmax = x_row[i00];
            larg = i00;
        }
    }

    // find the argmax value in the block
    float max_val = simd_max(lmax);
    int32_t arg_val = simd_max(select(-1, larg, lmax == max_val));

    threadgroup   float * shared_maxval = (threadgroup   float *) shmem;
    threadgroup int32_t * shared_argmax = (threadgroup int32_t *) shmem + N_SIMDWIDTH;

    if (ntg.x > N_SIMDWIDTH) {
        if (sgitg == 0) {
            shared_maxval[tiisg] = -INFINITY;
            shared_argmax[tiisg] = -1;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            shared_maxval[sgitg] = max_val;
            shared_argmax[sgitg] = arg_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = shared_maxval[tiisg];
        arg_val = shared_argmax[tiisg];

        float max_val_reduced   = simd_max(max_val);
        int32_t arg_val_reduced = simd_max(select(-1, arg_val, max_val == max_val_reduced));

        max_val = max_val_reduced;
        arg_val = arg_val_reduced;
    }

    if (tpitg.x == 0) {
        device float * tmp_max = (device float *) tmp;
        device int32_t * tmp_arg = (device int32_t *) tmp + npart * (int) args.ne01;

        tmp_max[row*npart + part] = max_val;
        tmp_arg[row*npart + part] = arg_val;
    }
}

kernel void kernel_argmax_final_f32(
        constant ggml_metal_kargs_argmax & args,
        device   const char * tmp,
        device         char * dst,
        threadgroup    char * shmem [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint3  ntg[[threads_per_threadgroup]]) {
    const int row = tgpig.x;
    const int npart = (int) args.npart;

    device const float * tmp_max = (device const float *) tmp;
    device const int32_t * tmp_arg = (device const int32_t *) tmp + npart * (int) args.ne01;

    float   lmax = -INFINITY;
    int32_t larg = -1;

    for (int i = tpitg.x; i < npart; i += ntg.x) {
        if (tmp_max[row*npart + i] > lmax) {
            lmax = tmp_max[row*npart + i];
            larg = tmp_arg[row*npart + i];
        }
    }

    // find the argmax value in the block
    float max_val = simd_max(lmax);
    int32_t arg_val = simd_max(select(-1, larg, lmax == max_val));

    threadgroup   float * shared_maxval = (threadgroup   float *) shmem;
    threadgroup int32_t * shared_argmax = (threadgroup int32_t *) shmem + N_SIMDWIDTH;

    if (ntg.x > N_SIMDWIDTH) {
        if (sgitg == 0) {
            shared_maxval[tiisg] = -INFINITY;
            shared_argmax[tiisg] = -1;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            shared_maxval[sgitg] = max_val;
            shared_argmax[sgitg] = arg_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = shared_maxval[tiisg];
        arg_val = shared_argmax[tiisg];

        float max_val_reduced   = simd_max(max_val);
        int32_t arg_val_reduced = simd_max(select(-1, arg_val, max_val == max_val_reduced));

        max_val = max_val_reduced;
        arg_val = arg_val_reduced;
    }

    if (tpitg.x == 0) {
        ((device int32_t *) dst)[row] = arg_val;
    }
}

kernel void kernel_diag_f32(
        constant ggml_metal_kargs_diag & args,
        device   const char * src0,
        device         char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]]) {
    constexpr short NW = N_SIMDWIDTH;

    const int32_t i3 = tgpig.z;
    const int32_t i2 = tgpig.y;
    const int32_t i1 = tgpig.x;

    device const float * src0_ptr = (device const float *)(src0 +                i2*args.nb02 + i3*args.nb03);
    device       float * dst_ptr  = (device       float *)(dst  + i1*args.nb01 + i2*args.nb2  + i3*args.nb3);

    for (int i0 = tiitg; i0 < args.ne0; i0 += NW) {
        dst_ptr[i0] = i0 == i1 ? src0_ptr[i0] : 0.0f;
    }
}

kernel void kernel_roll_f32(
    constant ggml_metal_kargs_roll & args,
    device  const char * src0,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    const int64_t i3 = tgpig.z;
    const int64_t i2 = tgpig.y;
    const int64_t i1 = tgpig.x;

    device const float * src0_ptr = (device const float *) src0;
    device       float * dst_ptr  = (device       float *) dst;

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        // apply shifts and wrap around
        int64_t i00 = i0 - args.s0;
        int64_t i01 = i1 - args.s1;
        int64_t i02 = i2 - args.s2;
        int64_t i03 = i3 - args.s3;

        if (i00 < 0) { i00 += args.ne00; } else if (i00 >= args.ne00) { i00 -= args.ne00; }
        if (i01 < 0) { i01 += args.ne01; } else if (i01 >= args.ne01) { i01 -= args.ne01; }
        if (i02 < 0) { i02 += args.ne02; } else if (i02 >= args.ne02) { i02 -= args.ne02; }
        if (i03 < 0) { i03 += args.ne03; } else if (i03 >= args.ne03) { i03 -= args.ne03; }

        int64_t src_idx = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00 + i00;
        int64_t dst_idx = i3 *args.ne2 *args.ne1 *args.ne0  + i2 *args.ne1 *args.ne0  + i1 *args.ne0  + i0;

        dst_ptr[dst_idx] = src0_ptr[src_idx];
    }
}

constant bool FC_pad_circular [[function_constant(FC_PAD + 0)]];

// circular means on a torus, so the coordinates wrap around
static inline int32_t wrap_around(int32_t coord, int32_t size) {
    return (coord + size) % size;
}

kernel void kernel_pad_f32(
    constant ggml_metal_kargs_pad & args,
    device  const char * src0,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {
    const int32_t i3 = tgpig.z;
    const int32_t i2 = tgpig.y;
    const int32_t k0 = tgpig.x/args.ne1;
    const int32_t i1 = tgpig.x - k0*args.ne1;

    const int32_t ne00 = args.ne00;
    const int32_t ne01 = args.ne01;
    const int32_t ne02 = args.ne02;
    const int32_t ne03 = args.ne03;

    int32_t i01 = i1 - args.lp1;
    int32_t i02 = i2 - args.lp2;
    int32_t i03 = i3 - args.lp3;

    if (FC_pad_circular) {
        i01 = wrap_around(i01, ne01);
        i02 = wrap_around(i02, ne02);
        i03 = wrap_around(i03, ne03);
    }

    device float * dst_ptr = (device float *) (dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1);

    // the row lies in the padded region, so no source row backs it
    if (i01 < 0 || i01 >= ne01 ||
        i02 < 0 || i02 >= ne02 ||
        i03 < 0 || i03 >= ne03) {
        for (int32_t l0 = 0; l0 < 1024; l0 += ntg.x) {
            const int32_t i0 = k0*1024 + tpitg.x + l0;
            if (i0 >= args.ne0) {
                break;
            }

            dst_ptr[i0] = 0.0f;
        }

        return;
    }

    device const char * src0_row = src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01;

    for (int32_t l0 = 0; l0 < 1024; l0 += ntg.x) {
        const int32_t i0 = k0*1024 + tpitg.x + l0;
        if (i0 >= args.ne0) {
            break;
        }

        int32_t i00 = i0 - args.lp0;

        if (FC_pad_circular) {
            i00 = wrap_around(i00, ne00);
        }

        dst_ptr[i0] = i00 >= 0 && i00 < ne00 ? *((device const float *) (src0_row + i00*args.nb00)) : 0.0f;
    }
}

// TODO: this is slow - optimize
kernel void kernel_pad_reflect_1d_f32(
    constant   ggml_metal_kargs_pad_reflect_1d & args,
    device  const char * src0,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3  tgpg[[threadgroups_per_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    const int64_t i3 = tgpig.z;
    const int64_t i2 = tgpig.y;
    const int64_t i1 = tgpig.x;

    const int64_t i03 = i3;
    const int64_t i02 = i2;
    const int64_t i01 = i1;

    device const float * src0_ptr = (device const float *) (src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01);
    device       float * dst_ptr  = (device       float *) (dst  +  i3*args.nb3  +  i2*args.nb2  +  i1*args.nb1);

    if (i1 < args.ne01 && i2 < args.ne02 && i3 < args.ne03) {
        for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
            if (i0 < args.p0) {
                dst_ptr[i0] = src0_ptr[args.p0 - i0];
            } else if (i0 < args.ne0 - args.p1) {
                dst_ptr[i0] = src0_ptr[i0 - args.p0];
            } else {
                dst_ptr[i0] = src0_ptr[(args.ne0 - args.p1 - args.p0) - (args.p1 + 1 - (args.ne0 - i0)) - 1];
            }
        }
    }
}

kernel void kernel_arange_f32(
    constant   ggml_metal_kargs_arange & args,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    device float * dst_ptr = (device float *) dst;

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        dst_ptr[i0] = args.start + args.step * i0;
    }
}

kernel void kernel_timestep_embedding_f32(
    constant  ggml_metal_kargs_timestep_embedding & args,
    device  const char * src0,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    int i = tgpig.x;
    device float * embed_data = (device float *)(dst + i*args.nb1);

    int half_ = args.dim / 2;
    for (int j = tpitg.x; j < half_; j += ntg.x) {
        float timestep = ((device float *)src0)[i];
        float freq = (float)exp(-log((float)args.max_period) * j / half_);
        float arg = timestep * freq;
        embed_data[j        ] = cos(arg);
        embed_data[j + half_] = sin(arg);
    }

    if (args.dim % 2 != 0 && tpitg.x == 0) {
        embed_data[2 * half_] = 0.f;
    }
}

kernel void kernel_opt_step_adamw_f32(
        constant    ggml_metal_kargs_opt_step_adamw & args,
        device       float * x,
        device const float * g,
        device       float * g_m,
        device       float * g_v,
        device const float * pars,
        uint        gid[[thread_position_in_grid]]) {

    if (gid >= args.np) {
        return;
    }

    const float alpha  = pars[0];
    const float beta1  = pars[1];
    const float beta2  = pars[2];
    const float eps    = pars[3];
    const float wd     = pars[4];
    const float beta1h = pars[5];
    const float beta2h = pars[6];

    const float gi = g[gid];
    const float gmi = g_m[gid] * beta1 +      gi * (1.0f - beta1);
    const float gvi = g_v[gid] * beta2 + gi * gi * (1.0f - beta2);

    g_m[gid] = gmi;
    g_v[gid] = gvi;

    const float mh =      gmi * beta1h;
    const float vh = sqrt(gvi * beta2h) + eps;

    x[gid] = x[gid] * (1.0f - alpha * wd) - alpha * mh / vh;
}

kernel void kernel_opt_step_sgd_f32(
        constant    ggml_metal_kargs_opt_step_sgd & args,
        device       float * x,
        device const float * g,
        device const float * pars,
        uint        gid[[thread_position_in_grid]]) {

    if (gid >= args.np) {
        return;
    }

    x[gid] = x[gid] * (1.0f - pars[0] * pars[1]) - pars[0] * g[gid];
}

template<typename T>
kernel void kernel_memset(
        constant ggml_metal_kargs_memset & args,
        device T * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = args.val;
}

typedef decltype(kernel_memset<int64_t>) kernel_memset_t;

template [[host_name("kernel_memset_i64")]] kernel kernel_memset_t kernel_memset<int64_t>;

constant short FC_count_equal_nsg [[function_constant(FC_COUNT_EQUAL + 0)]];

template<typename T>
kernel void kernel_count_equal(
        constant ggml_metal_kargs_count_equal & args,
        device   const char * src0,
        device   const char * src1,
        device   atomic_int * dst,
        threadgroup int32_t * shmem_i32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const short NSG = FC_count_equal_nsg;

    const int i3 = tgpig.z;
    const int i2 = tgpig.y;
    const int i1 = tgpig.x;

    if (i3 >= args.ne03 || i2 >= args.ne02 || i1 >= args.ne01) {
        return;
    }

    int sum = 0;

    device const char * base0 = src0 + i1*args.nb01 + i2*args.nb02 + i3*args.nb03;
    device const char * base1 = src1 + i1*args.nb11 + i2*args.nb12 + i3*args.nb13;

    for (int64_t i0 = tpitg.x; i0 < args.ne00; i0 += ntg.x) {
        const T v0 = *(device const T *)(base0 + i0*args.nb00);
        const T v1 = *(device const T *)(base1 + i0*args.nb10);
        sum += (v0 == v1);
    }

    sum = simd_sum(sum);

    if (tiisg == 0) {
        shmem_i32[sgitg] = sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        float v = 0.0f;
        if (tpitg.x < NSG) {
            v = shmem_i32[tpitg.x];
        }

        float total = simd_sum(v);
        if (tpitg.x == 0) {
            atomic_fetch_add_explicit(dst, (int32_t) total, memory_order_relaxed);
        }
    }
}

typedef decltype(kernel_count_equal<int32_t>) kernel_count_equal_t;

template [[host_name("kernel_count_equal_i32")]] kernel kernel_count_equal_t kernel_count_equal<int32_t>;

// one SIMD group per row; the sum follows kernel_sum_rows_f32_f32_4 (one float4 per lane, simd_sum, horizontal sum)
kernel void kernel_clip_rows_f32(
        constant ggml_metal_kargs_clip_rows & args,
        device const char * src,
        device const char * scale,
        device       char * dst,
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const int i1 = tgpig.x;
    const int i2 = tgpig.y;

    float4 x = float4(0.0f);
    if (tiisg < args.nl) {
        x = ((device const float4 *) (src + i1*args.nb1 + i2*args.nb2))[tiisg];
        if (args.has_scale) {
            x = x * ((device const float4 *) (scale + i1*args.nbs1))[tiisg];
        }
    }

    float4 s = simd_sum(x * x);

    const float nrm = sqrt(sum(s));
    const float n2  = args.sc * nrm + args.bi;
    const float c   = clamp(args.thr / n2, args.lo, args.hi);

    if (tiisg < args.nl) {
        ((device float4 *) (dst + i1*args.nbd1 + i2*args.nbd2))[tiisg] = x * c;
    }
}

// prompt batches: one thread per (channel, token, sequence), same arithmetic as kernel_sconv_f32; the thread of the
// last token writes the state. Inputs and outputs must not alias (the check rejects it for T > 1).
kernel void kernel_sconv_par_f32(
        constant ggml_metal_kargs_sconv & args,
        device const char * past,
        device const char * cur,
        device const char * w,
        device       char * state,
        device       char * dst,
        device const char * resid,
        uint tgpig [[threadgroup_position_in_grid]],
        uint tpitg [[thread_position_in_threadgroup]],
        uint ntg   [[threads_per_threadgroup]]) {
    const int64_t i = (int64_t) tgpig*ntg + tpitg;
    if (i >= (int64_t) args.C*args.T*args.S) {
        return;
    }
    const int c    = i % args.C;
    const int64_t rest = i / args.C;
    const int t    = rest % args.T;
    const int s    = rest / args.T;

    device const float * p = (device const float *) (past + c*args.nbp1 + s*args.nbp2);
    float v[4];
    for (int k = 0; k < 4; ++k) {
        const int j = t + k; // window position in [past | cur]
        v[k] = j < 3 ? p[j] : ((device const float *) (cur + ((int64_t) s*args.T + j - 3)*args.nbx1))[c];
    }
    const float4 wc = *((device const float4 *) (w + c*args.nbw1));

    float conv = 0.0f;
    conv += dot(float4(v[0], v[1], v[2], v[3]), wc);

    const int64_t tok = (int64_t) s*args.T + t;
    const float r = v[3] + conv;
    const float y = args.sc * r + args.bi;
    if (args.has_res) {
        const float rv = ((device const float *) (resid + tok*args.nbr1))[c];
        ((device float *) (dst + tok*args.nbd1))[c] = rv + y;
    } else {
        ((device float *) (dst + tok*args.nbd1))[c] = y;
    }

    if (t == args.T - 1) {
        device float * st = (device float *) (state + s*args.nbs1) + 3*c;
        for (int k = 0; k < 3; ++k) {
            const int j = args.T + k;
            st[k] = j < 3 ? p[j] : ((device const float *) (cur + ((int64_t) s*args.T + j - 3)*args.nbx1))[c];
        }
    }
}

// one token, dynamic taps: out = (cur + (conv + sum_lag x[3-lag]*tap[lag, head]))*sc, in the unfused rounding order
// K clip or V rms_norm of one head row (one SIMD group), written as f16 into the cache row of its token:
// same reductions as kernel_sum_rows_f32_f32_4 / kernel_rms_norm_fuse_impl<float4,1>, then the set_rows cast
// rms_norm of one token row of x (same reduction as kernel_rms_norm_fuse_impl<float4, 1>, so the normed values are
// identical), written to nrm, and the concat [stack rows; normed row] written to cat; one threadgroup per token
kernel void kernel_norm_concat_f32(
        constant ggml_metal_kargs_norm_concat & args,
        device const char * x,
        device const char * stk,
        device       char * nrm,
        device       char * cat,
        threadgroup float * shmem_f32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    if (sgitg == 0) {
        shmem_f32[tiisg] = 0.0f;
    }

    const int t = tgpig.x;

    device const float4 * xr = (device const float4 *) (x + t*args.nbx1);

    float sumf = 0.0f;
    for (int i = tpitg.x; i < args.E4; i += ntg.x) {
        sumf += dot(xr[i], xr[i]);
    }
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem_f32[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = shmem_f32[tiisg];
    sumf = simd_sum(sumf);

    const float mean  = sumf/args.E;
    const float scale = 1.0f/sqrt(mean + args.eps);

    device float4 * yn = (device float4 *) (nrm + t*args.nbn1);
    device float4 * yc = (device float4 *) (cat + t*args.nbc2);
    for (int i = tpitg.x; i < args.E4; i += ntg.x) {
        const float4 v = xr[i]*scale;
        yn[i] = v;
        for (int r = 0; r < args.K; ++r) {
            ((device float4 *) ((device char *) yc + r*args.nbc1))[i] = ((device const float4 *) (stk + r*args.nbs1 + t*args.nbs2))[i];
        }
        ((device float4 *) ((device char *) yc + args.K*args.nbc1))[i] = v;
    }
}

kernel void kernel_row_store_f32_f16(
        constant ggml_metal_kargs_row_store & args,
        device const char * src,
        device const char * idxs,
        device       char * cache,
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const int h = tgpig.x;
    const int t = tgpig.y;

    float4 x = float4(0.0f);
    if (tiisg < args.nl) {
        x = ((device const float4 *) (src + h*args.nb1 + t*args.nb2))[tiisg];
    }

    float4 y;
    if (args.mode == 0) {
        float4 s = simd_sum(x * x);
        const float nrm = sqrt(sum(s));
        const float n2  = args.sc * nrm + args.bi;
        const float c   = clamp(args.thr / n2, args.lo, args.hi);
        y = x * c;
    } else {
        float sumf = 0.0f;
        sumf += dot(x, x);
        sumf = simd_sum(sumf);
        const float mean  = sumf/args.bi;
        const float scale = 1.0f/sqrt(mean + args.sc);
        y = x*scale;
    }

    const int64_t row = args.idx_i64 ? ((device const int64_t *) (idxs + t*args.nbi0))[0]
                                     : (int64_t) ((device const int32_t *) (idxs + t*args.nbi0))[0];
    if (tiisg < args.nl) {
        ((device half4 *) (cache + row*args.nbd1))[h*args.nl + tiisg] = (half4) y;
    }
}

// one thread per (channel, sequence), rolling the 4-wide window over the T tokens of the sequence: per token the
// kernel_ssm_conv_f32_f32_4 dot, then the 4 dynamic tap terms added one by one, the residual and the scale; the last
// 3 inputs become the new state (T = 1, S = 1 is the decode case)
kernel void kernel_sconv_dyn_f32(
        constant ggml_metal_kargs_sconv_dyn & args,
        device const char * past,
        device const char * cur,
        device const char * w,
        device       char * state,
        device       char * dst,
        device const char * dyn,
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint3 tpitg [[thread_position_in_threadgroup]],
        uint3 ntg   [[threads_per_threadgroup]]) {
    const int c = tgpig.x*ntg.x + tpitg.x;
    const int s = tgpig.y;
    if (c >= args.C) {
        return;
    }
    device const float * p = (device const float *) (past + c*args.nbp1 + s*args.nbp2);
    float x0 = p[0];
    float x1 = p[1];
    float x2 = p[2];
    const float4 wc = *((device const float4 *) (w + c*args.nbw1));

    device const char * tapc = dyn + (c/args.cph)*args.nbd1;

    for (int t = 0; t < args.T; ++t) {
        const int64_t tok = (int64_t) s*args.T + t;
        const float x3 = ((device const float *) cur)[tok*args.C + c];

        float o = 0.0f;
        o += dot(float4(x0, x1, x2, x3), wc);

        device const float * tap = (device const float *) (tapc + tok*args.nbd2);
        const float v[4] = { x0, x1, x2, x3 };
        {
            // the unfused graph rounds each tap product (MUL) before adding it (ADD): no fused multiply-add here
#pragma clang fp contract(off)
            for (int lag = 0; lag < 4; ++lag) {
                const float term = v[3 - lag] * tap[lag];
                o = o + term;
            }
            const float r = x3 + o;
            ((device float *) dst)[tok*args.C + c] = args.sc * r + args.bi;
        }

        x0 = x1;
        x1 = x2;
        x2 = x3;
    }

    device float * st = (device float *) (state + s*args.nbs1) + 3*c;
    st[0] = x0;
    st[1] = x1;
    st[2] = x2;
}

// one thread per (channel, sequence): roll the 4-wide window over the tokens, like kernel_ssm_conv_f32_f32_4
// (dot of the window with the weights), then add the input and scale; the last 3 inputs become the new state
// f16 per-head mask: out[kv, t, h, s] = f16((sum_i bias[i, rp] * R[i, h, tok] - sinks[h, tok]) + mask[kv, t, 0, s])
kernel void kernel_relpos_mask_f32(
        constant ggml_metal_kargs_relpos & args,
        device const char * bias,
        device const char * R,
        device const char * sinks,
        device const char * relp,
        device const char * mask,
        device       half * dst,
        uint tgpig [[threadgroup_position_in_grid]],
        uint tpitg [[thread_position_in_threadgroup]],
        uint ntg   [[threads_per_threadgroup]]) {
    const int64_t i = (int64_t) tgpig*ntg + tpitg;
    const int64_t n = (int64_t) args.NK*args.Ts*args.H*args.S;
    if (i >= n) {
        return;
    }
    const int kv   = i % args.NK;
    int64_t   rest = i / args.NK;
    const int t    = rest % args.Ts; rest /= args.Ts;
    const int h    = rest % args.H;
    const int s    = rest / args.H;
    const int tok  = s*args.Ts + t;

    const int rp = ((device const int32_t *) (relp + tok*args.nbp1))[kv];

    device const float * b = (device const float *) (bias + rp*args.nbb1);
    device const float * r = (device const float *) (R + h*args.nbR1 + tok*args.nbR2);

    float v = 0.0f;
    for (int k = 0; k < args.NR; ++k) {
        v += b[k] * r[k];
    }
    v = v - ((device const float *) (sinks + tok*args.nbs1))[h];

    const float m = ((device const float *) (mask + (args.mne1 == 1 ? 0 : t)*args.nbm1 + (args.mne3 == 1 ? 0 : s)*args.nbm3))[kv];

    dst[i] = (half) (v + m);
}

// kernel_relpos_mask_f32 with the bias table of HG heads of one token computed once per threadgroup (same dot
// product order, so the same values), then the mask as a gather from threadgroup memory: one thread per kv cell
// writing HG heads; grid (kv chunks of CH, head groups, tokens)
kernel void kernel_relpos_mask2_f32(
        constant ggml_metal_kargs_relpos2 & a2,
        device const char * bias,
        device const char * R,
        device const char * sinks,
        device const char * relp,
        device const char * mask,
        device       half * dst,
        threadgroup  float * tbl [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tpitg [[thread_index_in_threadgroup]],
        ushort3 ntg  [[threads_per_threadgroup]]) {
    constant ggml_metal_kargs_relpos & args = a2.r;

    const int nth = ntg.x;
    const int h0  = tgpig.y*a2.HG;
    const int tok = tgpig.z;
    const int t   = tok % args.Ts;
    const int s   = tok / args.Ts;
    const int nhg = min(a2.HG, args.H - h0);

    for (int idx = tpitg; idx < nhg*a2.Ep; idx += nth) {
        const int hh = idx / a2.Ep;
        const int rp = idx % a2.Ep;
        const int h  = h0 + hh;

        device const float * b = (device const float *) (bias + rp*args.nbb1);
        device const float * r = (device const float *) (R + h*args.nbR1 + tok*args.nbR2);

        float v = 0.0f;
        for (int k = 0; k < args.NR; ++k) {
            v += b[k] * r[k];
        }
        v = v - ((device const float *) (sinks + tok*args.nbs1))[h];

        tbl[hh*a2.Ep + rp] = v;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const int32_t * rprow = (device const int32_t *) (relp + tok*args.nbp1);
    device const float   * mrow  = (device const float *) (mask + (args.mne1 == 1 ? 0 : t)*args.nbm1 + (args.mne3 == 1 ? 0 : s)*args.nbm3);

    const int kv0 = tgpig.x*a2.CH;
    const int kv1 = min(kv0 + a2.CH, args.NK);

    for (int kv = kv0 + tpitg; kv < kv1; kv += nth) {
        const int   rp = rprow[kv];
        const float m  = mrow[kv];
        for (int hh = 0; hh < nhg; ++hh) {
            const int64_t i = kv + (int64_t) args.NK*(t + (int64_t) args.Ts*(h0 + hh + (int64_t) args.H*s));
            dst[i] = (half) (tbl[hh*a2.Ep + rp] + m);
        }
    }
}

// kernel_relpos_mask2_f32 with NR = 16: each thread keeps the R rows of the head group in registers and computes the
// table entries of one bucket for all heads of the group from one load of its bias row (same per-head dot order, so
// the same table values); the gather then handles 4 kv cells per thread with float4/int4 loads and half4 stores
// (requires NK % 4 == 0)
kernel void kernel_relpos_mask3_f32(
        constant ggml_metal_kargs_relpos2 & a2,
        device const char * bias,
        device const char * R,
        device const char * sinks,
        device const char * relp,
        device const char * mask,
        device       half * dst,
        threadgroup  float * tbl [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tpitg [[thread_index_in_threadgroup]],
        ushort3 ntg  [[threads_per_threadgroup]]) {
    constant ggml_metal_kargs_relpos & args = a2.r;
    constexpr int NR = 16;
    constexpr int HGM = 4;

    const int nth = ntg.x;
    const int h0  = tgpig.y*HGM;
    const int tok = tgpig.z;
    const int t   = tok % args.Ts;
    const int s   = tok / args.Ts;
    const int nhg = min(HGM, args.H - h0);

    float r[HGM][NR];
    float sk[HGM];
    for (int hh = 0; hh < HGM; ++hh) {
        const int h = min(h0 + hh, args.H - 1);
        device const float * rr = (device const float *) (R + h*args.nbR1 + tok*args.nbR2);
        for (int k = 0; k < NR; ++k) {
            r[hh][k] = rr[k];
        }
        sk[hh] = ((device const float *) (sinks + tok*args.nbs1))[h];
    }

    for (int rp = tpitg; rp < a2.Ep; rp += nth) {
        device const float * b = (device const float *) (bias + rp*args.nbb1);
        float bk[NR];
        for (int k = 0; k < NR; ++k) {
            bk[k] = b[k];
        }
        for (int hh = 0; hh < HGM; ++hh) {
            if (hh < nhg) {
                float v = 0.0f;
                for (int k = 0; k < NR; ++k) {
                    v = fma(bk[k], r[hh][k], v);
                }
                v = v - sk[hh];
                tbl[hh*a2.Ep + rp] = v;
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const int4   * rprow = (device const int4   *) (relp + tok*args.nbp1);
    device const float4 * mrow  = (device const float4 *) (mask + (args.mne1 == 1 ? 0 : t)*args.nbm1 + (args.mne3 == 1 ? 0 : s)*args.nbm3);

    const int kv0 = tgpig.x*a2.CH;
    const int kv1 = min(kv0 + a2.CH, args.NK);

    for (int kv = kv0 + 4*tpitg; kv < kv1; kv += 4*nth) {
        const int4   rp = rprow[kv/4];
        const float4 m  = mrow[kv/4];
        for (int hh = 0; hh < nhg; ++hh) {
            threadgroup const float * tb = tbl + hh*a2.Ep;
            const int64_t i = kv + (int64_t) args.NK*(t + (int64_t) args.Ts*(h0 + hh + (int64_t) args.H*s));
            *((device half4 *) (dst + i)) = half4((half) (tb[rp.x] + m.x), (half) (tb[rp.y] + m.y), (half) (tb[rp.z] + m.z), (half) (tb[rp.w] + m.w));
        }
    }
}

// softmax over the nr scores of query q, then out[e] = sum_r p_r * states[e, r]; one thread per channel
kernel void kernel_mix_tail_f32(
        constant ggml_metal_kargs_mix_tail & args,
        device const char * states,
        device const char * scores,
        device       char * dst,
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint3 tpitg [[thread_position_in_threadgroup]],
        uint3 ntg   [[threads_per_threadgroup]]) {
    const int e = tgpig.x*ntg.x + tpitg.x;
    const int q = tgpig.y;
    if (e >= args.E) {
        return;
    }
    states += tgpig.z*args.nbs2;
    scores += tgpig.z*args.nbp2;
    dst    += tgpig.z*args.nbd2;
    device const float * sc = (device const float *) (scores + q*args.nbp1);
    float p[8];
    float mx = -INFINITY;
    for (int r = 0; r < args.nr; ++r) {
        p[r] = sc[r]*args.scale;
        mx = max(mx, p[r]);
    }
    float sum = 0.0f;
    for (int r = 0; r < args.nr; ++r) {
        p[r] = exp(p[r] - mx);
        sum += p[r];
    }
    const float inv = 1.0f/sum;
    float o = 0.0f;
    for (int r = 0; r < args.nr; ++r) {
        o += ((device const float *) (states + r*args.nbs1))[e] * (p[r]*inv);
    }
    ((device float *) (dst + q*args.nbd1))[e] = o;
}

// kernel_mix_tail_f32 for a batch of tokens (grid z) with the softmax of each (token, query) row done by simdgroup 0
// exactly as kernel_soft_max_f32 (nr % 4 != 0) or kernel_soft_max_f32_4 (nr % 4 == 0) do it with 32 threads and no mask
// (same lanes, same expressions, same simd_max/simd_sum), so the weights match the separate softmax bit for bit; the
// weighted sum is kernel_mul_mv_f32_f32_short's dot over the nr terms. `smask` is the scores buffer bound twice, which
// makes the mask pointer null at run time like the separate kernel's src1 == src0 binding.
kernel void kernel_mix_tail_t_f32(
        constant ggml_metal_kargs_mix_tail & args,
        device const char * states,
        device const char * scores,
        device const char * smask,
        device       char * dst,
        threadgroup  float * pw [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tpitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const int q = tgpig.y;
    states += tgpig.z*args.nbs2;
    dst    += tgpig.z*args.nbd2 + q*args.nbd1;

    if (sgitg == 0) {
        const float slope = 1.0f;
        if (args.nr % 4 == 0) {
            device const float4 * psrc4 = (device const float4 *) (scores + tgpig.z*args.nbp2 + q*args.nbp1);
            device const  float * pmask = smask != scores ? (device const float *) (smask + tgpig.z*args.nbp2 + q*args.nbp1) : nullptr;

            float4 lmax4 = -INFINITY;
            for (int i00 = tiisg; i00 < args.nr/4; i00 += N_SIMDWIDTH) {
                lmax4 = fmax(lmax4, psrc4[i00]*args.scale + (float4)((pmask ? slope*pmask[i00] : 0.0f)));
            }
            const float lmax = MAX(MAX(lmax4[0], lmax4[1]), MAX(lmax4[2], lmax4[3]));
            const float max_val = simd_max(lmax);

            float4 lsum4 = 0.0f;
            float4 ev4   = 0.0f;
            for (int i00 = tiisg; i00 < args.nr/4; i00 += N_SIMDWIDTH) {
                const float4 exp_psrc4 = exp((psrc4[i00]*args.scale + (float4)((pmask ? slope*pmask[i00] : 0.0f))) - max_val);
                lsum4 += exp_psrc4;
                ev4 = exp_psrc4;
            }
            const float lsum = lsum4[0] + lsum4[1] + lsum4[2] + lsum4[3];
            const float sum = simd_sum(lsum);
            const float inv_sum = 1.0f/sum;
            if (tiisg < args.nr/4) {
                ev4 *= inv_sum;
                pw[4*tiisg + 0] = ev4[0];
                pw[4*tiisg + 1] = ev4[1];
                pw[4*tiisg + 2] = ev4[2];
                pw[4*tiisg + 3] = ev4[3];
            }
        } else {
            device const float * psrc0 = (device const float *) (scores + tgpig.z*args.nbp2 + q*args.nbp1);
            device const float * pmask = smask != scores ? (device const float *) (smask + tgpig.z*args.nbp2 + q*args.nbp1) : nullptr;

            float lmax = -INFINITY;
            for (int i00 = tiisg; i00 < args.nr; i00 += N_SIMDWIDTH) {
                lmax = MAX(lmax, psrc0[i00]*args.scale + (pmask ? slope*pmask[i00] : 0.0f));
            }
            const float max_val = simd_max(lmax);

            float lsum = 0.0f;
            float ev   = 0.0f;
            for (int i00 = tiisg; i00 < args.nr; i00 += N_SIMDWIDTH) {
                const float exp_psrc0 = exp((psrc0[i00]*args.scale + (pmask ? slope*pmask[i00] : 0.0f)) - max_val);
                lsum += exp_psrc0;
                ev = exp_psrc0;
            }
            const float sum = simd_sum(lsum);
            const float inv_sum = 1.0f/sum;
            if (tiisg < args.nr) {
                ev *= inv_sum;
                pw[tiisg] = ev;
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int e = tgpig.x*256 + tpitg;
    if (e >= args.E) {
        return;
    }
    float res = 0.0f;
    for (int i = 0; i < args.nr; ++i) {
        res += (float) ((device const float *) (states + i*args.nbs1))[e] * (float) pw[i];
    }
    ((device float *) dst)[e] = res;
}

// kernel_mix_tail_f32 for one query, with the scores computed here: each group of 4 simdgroups computes two score
// rows exactly as kernel_mul_mv_t_t_4_impl<float, float4, float, float4, 2> with NSG = 4 (same lanes, same order,
// same two-level simd_sum), so the scores match the separate product bit for bit; every threadgroup computes all
// nr scores (nr <= 8, E*nr MACs) and then its own channels of the softmax-weighted sum
kernel void kernel_mix_tail1_f32(
        constant ggml_metal_kargs_mix_tail1 & args,
        device const char * states,
        device const char * query,
        device       char * dst,
        threadgroup  float * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tpitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    constexpr short NW  = 32;
    constexpr short NSG = 4;
    constexpr short NB  = 32;
    constexpr short NF  = 16;
    constexpr short NF4 = NF/4;

    const short vg  = sgitg/NSG; // virtual threadgroup 0/1 (two rows each)
    const short vsg = sgitg%NSG; // its simdgroup index

    threadgroup float * red = shmem + vg*(2*NW); // 2 rows x 32 floats per virtual threadgroup
    threadgroup float * sc  = shmem + 4*NW;      // the nr scores

    const int nb = args.E/NB;

    device const float4 * y4 = (device const float4 *) query;

    const short ix = tiisg/(NW/NF);
    const short il = tiisg%(NW/NF);
    const int ib0 = vsg*NF + ix;

    for (int pass = 0; pass < (args.nr + 3)/4; ++pass) {
        const int r0 = (pass*2 + vg)*2;

        float sumf[2] = { 0.f, 0.f };

        if (r0 < args.nr) {
            device const float4 * ax4[2];
            ax4[0] = (device const float4 *) (states + (r0    )*args.nbs1);
            ax4[1] = (device const float4 *) (states + (r0 + 1 < args.nr ? r0 + 1 : r0)*args.nbs1);

            float4 yl4[NF4];
            device const float4 * yb4 = y4 + (ib0*NB + il*NF)/4;

            for (int ib = ib0; ib < nb; ib += NSG*NF) {
                for (short i = 0; i < NF4; ++i) {
                    yl4[i] = yb4[i];
                }
                for (short row = 0; row < 2; row++) {
                    device const float4 * xb4 = ax4[row] + (ib*NB + il*NF)/4;

                    float sumq = 0.f;
                    FOR_UNROLL (short i = 0; i < NF4; ++i) {
                        sumq += dot(float4(xb4[i]), float4(yl4[i]));
                    }

                    sumf[row] += sumq;
                }
                yb4 += NSG*NF*NW/4;
            }
        }

        // helper_mv_reduce_and_write<2> for this virtual threadgroup
        for (short row = 0; row < 2; ++row) {
            if (vsg == 0) {
                red[NW*row + tiisg] = 0.0f;
            }
            sumf[row] = simd_sum(sumf[row]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short row = 0; row < 2; ++row) {
            if (tiisg == 0) {
                red[NW*row + vsg] = sumf[row];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short row = 0; row < 2; ++row) {
            const float tot = simd_sum(red[NW*row + tiisg]);
            if (tiisg == 0 && vsg == 0 && r0 + row < args.nr) {
                sc[r0 + row] = tot;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // kernel_mix_tail_f32 for q = 0
    const int e = tgpig.x*256 + tpitg;
    if (e >= args.E) {
        return;
    }
    float p[8];
    float mx = -INFINITY;
    for (int r = 0; r < args.nr; ++r) {
        p[r] = sc[r]*args.scale;
        mx = max(mx, p[r]);
    }
    float sum = 0.0f;
    for (int r = 0; r < args.nr; ++r) {
        p[r] = exp(p[r] - mx);
        sum += p[r];
    }
    const float inv = 1.0f/sum;
    float o = 0.0f;
    for (int r = 0; r < args.nr; ++r) {
        o += ((device const float *) (states + r*args.nbs1))[e] * (p[r]*inv);
    }
    ((device float *) dst)[e] = o;
}

// kernel_mix_tail1_f32 for a batch of tokens (grid z): the scores as kernel_mul_mv_f32_f32_4 with nsg 4 computes them
// (the separate product for one query per token), then the exact softmax and second product of kernel_mix_tail_t_f32
kernel void kernel_mix_tail1_t_f32(
        constant ggml_metal_kargs_mix_tail1t & args,
        device const char * states,
        device const char * query,
        device const char * qmask,
        device       char * dst,
        threadgroup  float * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tpitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    constexpr short NW  = 32;
    constexpr short NSG = 4;
    constexpr short NB  = 32;
    constexpr short NF  = 16;
    constexpr short NF4 = NF/4;

    const short vg  = sgitg/NSG; // virtual threadgroup 0/1 (two rows each)
    const short vsg = sgitg%NSG; // its simdgroup index

    threadgroup float * red = shmem + vg*(2*NW); // 2 rows x 32 floats per virtual threadgroup
    threadgroup float * sc  = shmem + 4*NW;      // the nr scores
    threadgroup float * pw  = shmem + 4*NW + 8;  // the nr softmax weights

    states += tgpig.z*args.nbs2;
    query  += tgpig.z*args.nbq2;
    qmask  += tgpig.z*args.nbq2;
    dst    += tgpig.z*args.nbd2;

    const int nb = args.E/NB;

    device const float4 * y4 = (device const float4 *) query;

    const short ix = tiisg/(NW/NF);
    const short il = tiisg%(NW/NF);
    const int ib0 = vsg*NF + ix;

    for (int pass = 0; pass < (args.nr + 3)/4; ++pass) {
        const int r0 = (pass*2 + vg)*2;

        float sumf[2] = { 0.f, 0.f };

        if (r0 < args.nr) {
            device const float4 * ax4[2];
            ax4[0] = (device const float4 *) (states + (r0    )*args.nbs1);
            ax4[1] = (device const float4 *) (states + (r0 + 1 < args.nr ? r0 + 1 : r0)*args.nbs1);

            float4 yl4[NF4];
            device const float4 * yb4 = y4 + (ib0*NB + il*NF)/4;

            for (int ib = ib0; ib < nb; ib += NSG*NF) {
                for (short i = 0; i < NF4; ++i) {
                    yl4[i] = yb4[i];
                }
                for (short row = 0; row < 2; row++) {
                    device const float4 * xb4 = ax4[row] + (ib*NB + il*NF)/4;

                    float sumq = 0.f;
                    FOR_UNROLL (short i = 0; i < NF4; ++i) {
                        sumq += dot(float4(xb4[i]), float4(yl4[i]));
                    }

                    sumf[row] += sumq;
                }
                yb4 += NSG*NF*NW/4;
            }
        }

        // helper_mv_reduce_and_write<2> for this virtual threadgroup
        for (short row = 0; row < 2; ++row) {
            if (vsg == 0) {
                red[NW*row + tiisg] = 0.0f;
            }
            sumf[row] = simd_sum(sumf[row]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short row = 0; row < 2; ++row) {
            if (tiisg == 0) {
                red[NW*row + vsg] = sumf[row];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short row = 0; row < 2; ++row) {
            const float tot = simd_sum(red[NW*row + tiisg]);
            if (tiisg == 0 && vsg == 0 && r0 + row < args.nr) {
                sc[r0 + row] = tot;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // the softmax of the scores row, by simdgroup 0 exactly as kernel_soft_max_f32 / _f32_4 with 32 threads and no
    // mask (the mask pointer is null at run time like the separate kernel's src1 == src0 binding)
    if (sgitg == 0) {
        const float slope = 1.0f;
        if (args.nr % 4 == 0) {
            threadgroup const float4 * psrc4 = (threadgroup const float4 *) sc;
            device const float * pmask = qmask != query ? (device const float *) qmask : nullptr;

            float4 lmax4 = -INFINITY;
            for (int i00 = tiisg; i00 < args.nr/4; i00 += N_SIMDWIDTH) {
                lmax4 = fmax(lmax4, psrc4[i00]*args.scale + (float4)((pmask ? slope*pmask[i00] : 0.0f)));
            }
            const float lmax = MAX(MAX(lmax4[0], lmax4[1]), MAX(lmax4[2], lmax4[3]));
            const float max_val = simd_max(lmax);

            float4 lsum4 = 0.0f;
            float4 ev4   = 0.0f;
            for (int i00 = tiisg; i00 < args.nr/4; i00 += N_SIMDWIDTH) {
                const float4 exp_psrc4 = exp((psrc4[i00]*args.scale + (float4)((pmask ? slope*pmask[i00] : 0.0f))) - max_val);
                lsum4 += exp_psrc4;
                ev4 = exp_psrc4;
            }
            const float lsum = lsum4[0] + lsum4[1] + lsum4[2] + lsum4[3];
            const float sum = simd_sum(lsum);
            const float inv_sum = 1.0f/sum;
            if (tiisg < args.nr/4) {
                ev4 *= inv_sum;
                pw[4*tiisg + 0] = ev4[0];
                pw[4*tiisg + 1] = ev4[1];
                pw[4*tiisg + 2] = ev4[2];
                pw[4*tiisg + 3] = ev4[3];
            }
        } else {
            threadgroup const float * psrc0 = sc;
            device const float * pmask = qmask != query ? (device const float *) qmask : nullptr;

            float lmax = -INFINITY;
            for (int i00 = tiisg; i00 < args.nr; i00 += N_SIMDWIDTH) {
                lmax = MAX(lmax, psrc0[i00]*args.scale + (pmask ? slope*pmask[i00] : 0.0f));
            }
            const float max_val = simd_max(lmax);

            float lsum = 0.0f;
            float ev   = 0.0f;
            for (int i00 = tiisg; i00 < args.nr; i00 += N_SIMDWIDTH) {
                const float exp_psrc0 = exp((psrc0[i00]*args.scale + (pmask ? slope*pmask[i00] : 0.0f)) - max_val);
                lsum += exp_psrc0;
                ev = exp_psrc0;
            }
            const float sum = simd_sum(lsum);
            const float inv_sum = 1.0f/sum;
            if (tiisg < args.nr) {
                ev *= inv_sum;
                pw[tiisg] = ev;
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // the second product (kernel_mul_mv_f32_f32_short over the nr terms)
    const int e = tgpig.x*256 + tpitg;
    if (e >= args.E) {
        return;
    }
    float res = 0.0f;
    for (int i = 0; i < args.nr; ++i) {
        res += (float) ((device const float *) (states + i*args.nbs1))[e] * (float) pw[i];
    }
    ((device float *) dst)[e] = res;
}

// out = c * sigmoid(g), float4 lanes like the unfused unary/bin kernels
kernel void kernel_gate_mul_f32(
        constant ggml_metal_kargs_gate_mul & args,
        device const float4 * c,
        device const float4 * g,
        device       float4 * dst,
        uint tgpig [[threadgroup_position_in_grid]],
        uint tpitg [[thread_position_in_threadgroup]],
        uint ntg   [[threads_per_threadgroup]]) {
    const int i = tgpig*ntg + tpitg;
    if (i >= args.n4) {
        return;
    }
    const float4 x = g[i];
    const float4 s = 1 / (1 + exp(-x));
    dst[i] = c[i] * s;
}

kernel void kernel_sconv_f32(
        constant ggml_metal_kargs_sconv & args,
        device const char * past,
        device const char * cur,
        device const char * w,
        device       char * state,
        device       char * dst,
        device const char * resid,
        uint tgpig [[threadgroup_position_in_grid]],
        uint tpitg [[thread_position_in_threadgroup]],
        uint ntg   [[threads_per_threadgroup]]) {
    const int i = tgpig*ntg + tpitg;
    if (i >= args.C*args.S) {
        return;
    }
    const int c = i % args.C;
    const int s = i / args.C;

    device const float * p = (device const float *) (past + c*args.nbp1 + s*args.nbp2);
    float a = p[0];
    float b = p[1];
    float d = p[2];

    const float4 wc = *((device const float4 *) (w + c*args.nbw1));

    for (int t = 0; t < args.T; ++t) {
        const int64_t tok = (int64_t) s*args.T + t;
        const float x = ((device const float *) (cur + tok*args.nbx1))[c];

        float conv = 0.0f;
        conv += dot(float4(a, b, d, x), wc);

        const float r = x + conv;
        const float y = args.sc * r + args.bi;
        if (args.has_res) {
            const float rv = ((device const float *) (resid + tok*args.nbr1))[c];
            ((device float *) (dst + tok*args.nbd1))[c] = rv + y;
        } else {
            ((device float *) (dst + tok*args.nbd1))[c] = y;
        }

        a = b;
        b = d;
        d = x;
    }

    device float * st = (device float *) (state + s*args.nbs1) + 3*c;
    st[0] = a;
    st[1] = b;
    st[2] = d;
}

kernel void kernel_ctrl_mix_f32(
        constant ggml_metal_kargs_ctrl_mix & args,
        device const char  * c,
        device const char  * m,
        device const char  * f,
        device const char  * g,
        device       float * dst,
        uint tgpig [[threadgroup_position_in_grid]],
        uint tpitg [[thread_position_in_threadgroup]],
        uint ntg   [[threads_per_threadgroup]]) {
#pragma clang fp contract(off)
    const int i = tgpig*ntg + tpitg;
    if (i >= args.ne0*args.ne1) {
        return;
    }
    const int i0 = i % args.ne0;
    const int i1 = i / args.ne0;

    const float cv = ((device const float *) (c + i1*args.nbc1))[i0];
    const float mv = ((device const float *) (m + i1*args.nbm1))[i0];
    const float fv = ((device const float *) (f + i1*args.nbf1))[i0];
    const float gv = ((device const float *) (g + i1*args.nbg1))[i0];

    // same rounding steps as the unfused TANH, MUL, SUB, TANH, MUL, ADD kernels
    const float t1 = precise::tanh(fv) * cv;
    const float s  = cv - t1;
    const float t2 = precise::tanh(gv) * mv;

    dst[i] = s + t2;
}

kernel void kernel_situ_f32(
        constant ggml_metal_kargs_situ & args,
        device const char  * g,
        device const char  * u,
        device       float * dst,
        uint tgpig [[threadgroup_position_in_grid]],
        uint tpitg [[thread_position_in_threadgroup]],
        uint ntg   [[threads_per_threadgroup]]) {
    // float4 lanes, like the unfused kernel_unary_f32_f32_4 / kernel_bin_fuse_f32_f32_f32_4 (same rounding)
    const int ne0_4 = args.ne0/4;
    const int i = tgpig*ntg + tpitg;
    if (i >= ne0_4*args.ne1*args.ne2) {
        return;
    }
    const int i0 = i % ne0_4;
    const int i1 = (i / ne0_4) % args.ne1;
    const int i2 = i / (ne0_4*args.ne1);

    const float4 x = ((device const float4 *) (g + i2*args.nbg2 + i1*args.nbg1))[i0];
    const float4 y = ((device const float4 *) (u + i2*args.nbu2 + i1*args.nbu1))[i0];

    const float4 s1  = args.sc[0] * x + args.bi[0];
    const float4 s2  = args.sc[1] * precise::tanh(s1) + args.bi[1];
    const float4 sg  = 1 / (1 + exp(-x));
    const float4 act = s2 * sg;
    const float4 s3  = args.sc[2] * y + args.bi[2];
    const float4 s4  = args.sc[3] * precise::tanh(s3) + args.bi[3];

    ((device float4 *) dst)[i] = act * s4;
}

template <typename T>
kernel void kernel_snake(
        constant ggml_metal_kargs_snake & args,
        device const T     * x,
        device const float * a,
        device const float * inv_b,
        device       T     * dst,
        uint         tgpig [[threadgroup_position_in_grid]],
        uint         tpitg [[thread_position_in_threadgroup]],
        uint         ntg   [[threads_per_threadgroup]]) {

    const int idx = tgpig * ntg + tpitg;
    if (idx >= args.T * args.C) {
        return;
    }

    const int   c  = idx / args.T;  // x is [T, C], a / inv_b collapse to [1, C]
    const float xi = float(x[idx]);
    const float si = sin(a[c] * xi);
    dst[idx] = T(xi + si * si * inv_b[c]);
}

template [[host_name("kernel_snake_f32")]]  kernel void kernel_snake<float>(constant ggml_metal_kargs_snake &, device const float *, device const float *, device const float *, device float *, uint, uint, uint);
template [[host_name("kernel_snake_f16")]]  kernel void kernel_snake<half>(constant ggml_metal_kargs_snake &, device const half *, device const float *, device const float *, device half *, uint, uint, uint);
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_snake_bf16")]] kernel void kernel_snake<bfloat>(constant ggml_metal_kargs_snake &, device const bfloat *, device const float *, device const float *, device bfloat *, uint, uint, uint);
#endif

template<int N, typename src_t>
kernel void kernel_fwht(
        constant ggml_metal_kargs_fwht & args,
        device const src_t * src,
        device float * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort3  ntg[[threads_per_threadgroup]]) {

    constexpr int NW = N_SIMDWIDTH;
    constexpr int NE = N / NW;

    const float scale = 1.0f / sqrt((float) N);

    const int sg_per_tg = ntg.x / NW;
    const int64_t r = tgpig.x * sg_per_tg + sgitg;
    if (r >= args.nrows) {
        return;
    }

    src += r * N;
    dst += r * N;

    const int lane = tiisg;

    float reg[NE];
    for (int i = 0; i < NE; i++) {
        reg[i] = float(src[i*NW + lane])*scale;
    }
    for (int i = 1; i < NW; i *= 2) {
        for (int j = 0; j < NE; j++) {
            const float val = reg[j];
            const float val2 = simd_shuffle_xor(val, i);
            reg[j] = val2 - val + 2*((lane & i) == 0)*val;
        }
    }

    for (int i = NW; i < N; i *= 2) {
        const int step = i / NW;
        for (int j = 0; j < NE; j += (2 * step)) {
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k ];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

    for (int i = 0; i < NE; i++) {
        dst[i*NW + lane] = reg[i];
    }
}

// Wide blocks: one row per threadgroup instead of per simdgroup, so each thread keeps
// N/NT values rather than N/32. Butterflies below the simdgroup width still shuffle;
// those up to NT go through threadgroup memory; the rest stay in registers.
// TODO: try avoiding branch https://github.com/ggml-org/llama.cpp/pull/29094#discussion_r4049563223
// TODO: try to unroll loops
template<int N, int NT, typename src_t>
kernel void kernel_fwht_tg(
        constant ggml_metal_kargs_fwht & args,
        device const src_t * src,
        device float * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort3  ntg[[threads_per_threadgroup]]) {

    constexpr int NW = N_SIMDWIDTH;
    constexpr int NE = N / NT;

    threadgroup float shmem[N];

    const float scale = 1.0f / sqrt((float) N);

    const int64_t r = tgpig.x;
    if (r >= args.nrows) {
        return;
    }

    src += r * N;
    dst += r * N;

    const int tid = sgitg * NW + tiisg;

    float reg[NE];
    for (int i = 0; i < NE; i++) {
        reg[i] = float(src[i*NT + tid])*scale;
    }

    for (int i = 1; i < NW; i *= 2) {
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = simd_shuffle_xor(val, i);
            reg[j] = (tid & i) == 0 ? val2 + val : val2 - val;
        }
    }

    for (int i = NW; i < NT; i *= 2) {
        for (int j = 0; j < NE; j++) {
            shmem[j*NT + tid] = reg[j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = shmem[j*NT + (tid ^ i)];
            reg[j] = (tid & i) == 0 ? val2 + val : val2 - val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int i = NT; i < N; i *= 2) {
        const int step = i / NT;
        for (int j = 0; j < NE; j += (2 * step)) {
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k ];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

    for (int i = 0; i < NE; i++) {
        dst[i*NT + tid] = reg[i];
    }
}

typedef decltype(kernel_fwht<64, float>) kernel_fwht_f32_t;
typedef decltype(kernel_fwht<64, half>)  kernel_fwht_f16_t;

template [[host_name("kernel_fwht_f32_64")]]  kernel kernel_fwht_f32_t kernel_fwht<64,  float>;
template [[host_name("kernel_fwht_f32_128")]] kernel kernel_fwht_f32_t kernel_fwht<128, float>;
template [[host_name("kernel_fwht_f32_256")]] kernel kernel_fwht_f32_t kernel_fwht<256, float>;

template [[host_name("kernel_fwht_f16_64")]]  kernel kernel_fwht_f16_t kernel_fwht<64,  half>;
template [[host_name("kernel_fwht_f16_128")]] kernel kernel_fwht_f16_t kernel_fwht<128, half>;
template [[host_name("kernel_fwht_f16_256")]] kernel kernel_fwht_f16_t kernel_fwht<256, half>;

template [[host_name("kernel_fwht_f32_512")]]  kernel kernel_fwht_f32_t kernel_fwht_tg<512,  GGML_METAL_FWHT_TG_NT, float>;
template [[host_name("kernel_fwht_f32_1024")]] kernel kernel_fwht_f32_t kernel_fwht_tg<1024, GGML_METAL_FWHT_TG_NT, float>;
template [[host_name("kernel_fwht_f32_2048")]] kernel kernel_fwht_f32_t kernel_fwht_tg<2048, GGML_METAL_FWHT_TG_NT, float>;
template [[host_name("kernel_fwht_f32_4096")]] kernel kernel_fwht_f32_t kernel_fwht_tg<4096, GGML_METAL_FWHT_TG_NT, float>;
template [[host_name("kernel_fwht_f32_8192")]] kernel kernel_fwht_f32_t kernel_fwht_tg<8192, GGML_METAL_FWHT_TG_NT, float>;

template [[host_name("kernel_fwht_f16_512")]]  kernel kernel_fwht_f16_t kernel_fwht_tg<512,  GGML_METAL_FWHT_TG_NT, half>;
template [[host_name("kernel_fwht_f16_1024")]] kernel kernel_fwht_f16_t kernel_fwht_tg<1024, GGML_METAL_FWHT_TG_NT, half>;
template [[host_name("kernel_fwht_f16_2048")]] kernel kernel_fwht_f16_t kernel_fwht_tg<2048, GGML_METAL_FWHT_TG_NT, half>;
template [[host_name("kernel_fwht_f16_4096")]] kernel kernel_fwht_f16_t kernel_fwht_tg<4096, GGML_METAL_FWHT_TG_NT, half>;
template [[host_name("kernel_fwht_f16_8192")]] kernel kernel_fwht_f16_t kernel_fwht_tg<8192, GGML_METAL_FWHT_TG_NT, half>;

constant int FC_dsv4_hc_n_hc [[function_constant(FC_DSV4_HC + 0)]];

kernel void kernel_dsv4_hc_comb_f32(
        constant ggml_metal_kargs_dsv4_hc_comb & args,
        device const char * mixes,
        device const char * scale,
        device const char * base,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    constexpr ushort hc = 4;
    constexpr ushort comb_offset = 2*hc;

    const int it = tgpig.x*ntg.y + sgitg;
    if (it >= args.n_tokens) {
        return;
    }

    float scale_lane = 0.0f;
    if (tiisg == 0) {
        scale_lane = *(device const float *) (scale + 2*args.nb_s0);
    }
    const float scale_comb = simd_shuffle(scale_lane, 0);

    float v = 0.0f;
    if (tiisg < hc*hc) {
        v = *(device const float *) (mixes + (comb_offset + tiisg)*args.nb_m0 + it*args.nb_m1)*scale_comb
          + *(device const float *) (base   + (comb_offset + tiisg)*args.nb_b0);
    }

    // Softmax across destinations (the four contiguous lanes for each source).
    float vmax = max(v, simd_shuffle_xor(v, 1));
    vmax = max(vmax, simd_shuffle_xor(vmax, 2));
    v = exp(v - vmax);

    float sum = v + simd_shuffle_xor(v, 1);
    sum += simd_shuffle_xor(sum, 2);
    v = v/sum + args.eps;

    // Normalize columns: equal destination indices are four lanes apart.
    sum = v + simd_shuffle_xor(v, 4);
    sum += simd_shuffle_xor(sum, 8);
    v /= sum + args.eps;

    for (int i = 1; i < args.n_iter; ++i) {
        sum = v + simd_shuffle_xor(v, 1);
        sum += simd_shuffle_xor(sum, 2);
        v /= sum + args.eps;

        sum = v + simd_shuffle_xor(v, 4);
        sum += simd_shuffle_xor(sum, 8);
        v /= sum + args.eps;
    }

    if (tiisg < hc*hc) {
        const ushort idst = tiisg & 3;
        const ushort isrc = tiisg >> 2;
        *(device float *) (dst + idst*args.nb_d0 + isrc*args.nb_d1 + it*args.nb_d2) = v;
    }
}

kernel void kernel_dsv4_hc_pre_f32(
        constant ggml_metal_kargs_dsv4_hc_pre & args,
        device const char * x,
        device const char * weights,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int it = tgpig.y;
    const int i0 = ((int) tgpig.x*ntg.y + sgitg)*32 + tiisg;

    if (i0 >= args.n_embd) {
        return;
    }

    device const char * xb = x + i0*args.nb_x0 + it*args.nb_x2;
    float result = 0.0f;
    FOR_UNROLL (int ih = 0; ih < FC_dsv4_hc_n_hc; ++ih) {
        const float xv = *(device const float *) (xb + ih*args.nb_x1);
        const float wv = *(device const float *) (weights + ih*args.nb_w0 + it*args.nb_w1);
        result = fma(xv, wv, result);
    }

    *(device float *) (dst + i0*args.nb_d0 + it*args.nb_d1) = args.scale*result;
}

kernel void kernel_dsv4_hc_pre_gated_f32(
        constant ggml_metal_kargs_dsv4_hc_pre & args,
        device const char * x,
        device const char * gate,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int it = tgpig.y;
    const int i0 = ((int) tgpig.x*ntg.y + sgitg)*32 + tiisg;

    if (i0 >= args.n_embd) {
        return;
    }

    device const char * xb = x    + i0*args.nb_x0 + it*args.nb_x2;
    device const char * gb = gate + i0*args.nb_w0 + it*args.nb_w2;
    float result = 0.0f;
    FOR_UNROLL (int ih = 0; ih < FC_dsv4_hc_n_hc; ++ih) {
        const float g  = 1.0f/(1.0f + exp(-*(device const float *) (gb + ih*args.nb_w1)));
        const float xv = *(device const float *) (xb + ih*args.nb_x1);
        result = fma(xv, g, result);
    }

    *(device float *) (dst + i0*args.nb_d0 + it*args.nb_d1) = args.scale*result;
}

kernel void kernel_dsv4_hc_post_nocomb_f32(
        constant ggml_metal_kargs_dsv4_hc_post & args,
        device const char * x,
        device const char * residual,
        device const char * post,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    constexpr ushort hc = 4;

    const int it = tgpig.y;
    const int i0 = ((int) tgpig.x*ntg.y + sgitg)*32 + tiisg;

    float post_lane = 0.0f;
    if (tiisg < hc) {
        post_lane = *(device const float *) (post + tiisg*args.nb_p0 + it*args.nb_p1);
    }

    float post_reg[hc];
    FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
        post_reg[idst] = simd_shuffle(post_lane, idst);
    }

    if (i0 >= args.n_embd) {
        return;
    }

    const float xv = *(device const float *) (x + i0*args.nb_x0 + it*args.nb_x1);
    device const char * rb = residual + i0*args.nb_r0 + it*args.nb_r2;
    FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
        const float rv = *(device const float *) (rb + idst*args.nb_r1);
        *(device float *) (dst + i0*args.nb_d0 + idst*args.nb_d1 + it*args.nb_d2) = xv*post_reg[idst] + rv;
    }
}

kernel void kernel_dsv4_hc_post_f32(
        constant ggml_metal_kargs_dsv4_hc_post & args,
        device const char * x,
        device const char * residual,
        device const char * post,
        device const char * comb,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    constexpr ushort hc = 4;

    const int it = tgpig.y;
    const int i0 = ((int) tgpig.x*ntg.y + sgitg)*32 + tiisg;

    float coeff_lane = 0.0f;
    if (tiisg < hc) {
        coeff_lane = *(device const float *) (post + tiisg*args.nb_p0 + it*args.nb_p1);
    } else if (tiisg < hc + hc*hc) {
        const ushort idx  = tiisg - hc;
        const ushort idst = idx & 3;
        const ushort isrc = idx >> 2;
        coeff_lane = *(device const float *) (comb + idst*args.nb_c0 + isrc*args.nb_c1 + it*args.nb_c2);
    }

    float post_reg[hc];
    float comb_reg[hc][hc];
    FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
        post_reg[idst] = simd_shuffle(coeff_lane, idst);
    }
    FOR_UNROLL (ushort isrc = 0; isrc < hc; ++isrc) {
        FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
            comb_reg[isrc][idst] = simd_shuffle(coeff_lane, hc + idst + hc*isrc);
        }
    }

    if (i0 >= args.n_embd) {
        return;
    }

    const float xv = *(device const float *) (x + i0*args.nb_x0 + it*args.nb_x1);
    float result[hc];
    FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
        result[idst] = xv*post_reg[idst];
    }

    device const char * rb = residual + i0*args.nb_r0 + it*args.nb_r2;
    FOR_UNROLL (ushort isrc = 0; isrc < hc; ++isrc) {
        const float rv = *(device const float *) (rb + isrc*args.nb_r1);
        FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
            result[idst] = fma(rv, comb_reg[isrc][idst], result[idst]);
        }
    }

    FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
        *(device float *) (dst + i0*args.nb_d0 + idst*args.nb_d1 + it*args.nb_d2) = result[idst];
    }
}

constant bool FC_top_k_mask_has_score  [[function_constant(FC_TOP_K_MASK + 0)]];
constant bool FC_top_k_mask_has_blocks [[function_constant(FC_TOP_K_MASK + 1)]];
constant bool FC_top_k_mask_has_extra  [[function_constant(FC_TOP_K_MASK + 2)]];
constant bool FC_top_k_mask_vec        [[function_constant(FC_TOP_K_MASK + 3)]];

// the fill runs before the scatter, both are fully parallel over the whole dst and over the picks of one row
template <typename T, typename T4>
kernel void kernel_top_k_mask_fill(
        constant ggml_metal_kargs_top_k_mask & args,
        device char * dst,
        uint  tid[[thread_position_in_grid]],
        uint ntot[[threads_per_grid]]) {
    const T ninf = T(-INFINITY);

    if (FC_top_k_mask_vec) {
        device T4 * dst_4 = (device T4 *) dst;
        for (int64_t i = tid; i < args.n_elems/4; i += ntot) {
            dst_4[i] = T4(ninf);
        }
    } else {
        device T * dst_t = (device T *) dst;
        for (int64_t i = tid; i < args.n_elems; i += ntot) {
            dst_t[i] = ninf;
        }
    }
}

typedef decltype(kernel_top_k_mask_fill<half,  half4>)  kernel_top_k_mask_fill_h_t;
typedef decltype(kernel_top_k_mask_fill<float, float4>) kernel_top_k_mask_fill_f_t;

template [[host_name("kernel_top_k_mask_fill_f16")]] kernel kernel_top_k_mask_fill_h_t kernel_top_k_mask_fill<half,  half4>;
template [[host_name("kernel_top_k_mask_fill_f32")]] kernel kernel_top_k_mask_fill_f_t kernel_top_k_mask_fill<float, float4>;

// one threadgroup per row: copy the mask cells of the live picks and of the extra cells over the fill
template <typename T, typename T4>
kernel void kernel_top_k_mask(
        constant ggml_metal_kargs_top_k_mask & args,
        device const  char * mask,
        device const  char * top_k,
        device const  char * score,
        device const  char * blocks,
        device const  char * extra,
        device        char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        uint3   tpitg[[thread_position_in_threadgroup]],
        uint3 ntg_max[[threads_per_threadgroup]]) {
    const int i1 = tgpig.x;
    const int i2 = tgpig.y;
    const int i3 = tgpig.z;

    const int   tiitg = tpitg.x;
    const int     ntg = ntg_max.x;

    device const char * mask_row  = mask  + (i1 % args.m1)*args.nbm1 + (i2 % args.m2)*args.nbm2 + (i3 % args.m3)*args.nbm3;
    device       char * dst_row   = dst   + i1*args.nb1 + i2*args.nb2 + i3*args.nb3;
    device const char * top_k_row = top_k + i1*args.nbk1 + i2*args.nbk2 + i3*args.nbk3;

    device const T * mask_row_t = (device const T *) mask_row;
    device       T * dst_row_t  = (device       T *) dst_row;

    for (int i = tiitg; i < args.n_top; i += ntg) {
        const int32_t p = *(device const int32_t *) (top_k_row + i*args.nbk0);
        if (p < 0 || p >= args.n_pick) {
            continue;
        }
        if (FC_top_k_mask_has_score && !(*(device const float *) (score + p*args.nbs0 + i1*args.nbs1 + i2*args.nbs2 + i3*args.nbs3) > -INFINITY)) {
            continue;
        }
        if (FC_top_k_mask_has_blocks) {
            FOR_UNROLL (int k = 0; k < args.n_blk; ++k) {
                const int32_t c = *(device const int32_t *) (blocks + k*args.nbb0 + p*args.nbb1);
                if (c >= 0 && c < args.n_kv) {
                    dst_row_t[c] = mask_row_t[c];
                }
            }
        } else if (p < args.n_kv) {
            dst_row_t[p] = mask_row_t[p];
        }
    }

    if (FC_top_k_mask_has_extra) {
        for (int i = tiitg; i < args.n_extra; i += ntg) {
            const int32_t c = *(device const int32_t *) (extra + i*args.nbe0 + i1*args.nbe1 + i2*args.nbe2 + i3*args.nbe3);
            if (c >= 0 && c < args.n_kv) {
                dst_row_t[c] = mask_row_t[c];
            }
        }
    }
}

typedef decltype(kernel_top_k_mask<half,  half4>)  kernel_top_k_mask_h_t;
typedef decltype(kernel_top_k_mask<float, float4>) kernel_top_k_mask_f_t;

template [[host_name("kernel_top_k_mask_f16")]] kernel kernel_top_k_mask_h_t kernel_top_k_mask<half,  half4>;
template [[host_name("kernel_top_k_mask_f32")]] kernel kernel_top_k_mask_f_t kernel_top_k_mask<float, float4>;
