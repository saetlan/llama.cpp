#include "ggml-metal-fusion.h"

#include "ggml-metal-common.h"
#include "ggml-metal-device.h"

#include "ggml-backend-impl.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <set>
#include <string>
#include <vector>

// derive the non-empty op sequence from the raw `ops_all` sequence
static std::vector<ggml_op> ggml_metal_fusion_filter_ops(const std::vector<ggml_op> & ops_all) {
    std::vector<ggml_op> ops;

    for (ggml_op op : ops_all) {
        if (!ggml_op_is_empty(op)) {
            ops.push_back(op);
        }
    }

    return ops;
}

struct ggml_metal_fusion {
    ggml_metal_fusion_id id;

    std::vector<ggml_op> ops;     // non-empty op sequence, derived from ops_all
    std::vector<ggml_op> ops_all; // full raw op sequence (may include empty RESHAPE/VIEW nodes)
    std::vector<int>     outs;    // additional fused output nodes, relative to ops

    // if unsafe: the generic chain/shape + ggml_can_fuse_subgraph checks are skipped and the
    // check callback below is the sole validator (used for patterns that are not elision chains,
    // e.g. the gdn + cache-cpy write-through fusion)
    bool unsafe;

    // extra backend constraints on top of ggml_can_fuse_subgraph
    // nodes[j] is the j-th node of the pattern; node_idxs[idx + j] is its raw graph index
    bool (*check)(const struct ggml_metal_fusion   * fusion,
                  const struct ggml_tensor * const * nodes,
                  const struct ggml_cgraph         * gf,
                  const int                        * node_idxs,
                        int                          idx,
                        ggml_metal_fusion_mode       mode);

    ggml_metal_fusion(
            ggml_metal_fusion_id id,
            const std::vector<ggml_op> & ops_all,
            const std::vector<int> & outs,
            bool unsafe,
            bool (*check)(const struct ggml_metal_fusion   * fusion,
                          const struct ggml_tensor * const * nodes,
                          const struct ggml_cgraph         * gf,
                          const int                        * node_idxs,
                                int                          idx,
                                ggml_metal_fusion_mode       mode))
        : id(id),
          ops(ggml_metal_fusion_filter_ops(ops_all)),
          ops_all(ops_all),
          outs(outs),
          unsafe(unsafe),
          check(check) {
    }
};

ggml_metal_fusion_id ggml_metal_fusion_get_id(const ggml_metal_fusion * fusion) {
    return fusion->id;
}

// ---- helpers -------------------------------------------------------------

// true if two tensors live in the same Metal buffer
static bool ggml_metal_fusion_same_buffer(const ggml_tensor * a, const ggml_tensor * b) {
    if (!a || !b) {
        return false;
    }

    ggml_backend_buffer_t ba = a->view_src ? a->view_src->buffer : a->buffer;
    ggml_backend_buffer_t bb = b->view_src ? b->view_src->buffer : b->buffer;

    ggml_metal_buffer_t ca = (ggml_metal_buffer_t) ba->context;
    ggml_metal_buffer_t cb = (ggml_metal_buffer_t) bb->context;

    return ggml_metal_buffer_get_id(ca, a).metal == ggml_metal_buffer_get_id(cb, b).metal;
}

// true if the memory of two tensors overlaps in the same Metal buffer
static bool ggml_metal_fusion_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    ggml_backend_buffer_t ba = a->view_src ? a->view_src->buffer : a->buffer;
    ggml_backend_buffer_t bb = b->view_src ? b->view_src->buffer : b->buffer;

    const ggml_metal_buffer_id bid_a = ggml_metal_buffer_get_id((ggml_metal_buffer_t) ba->context, a);
    const ggml_metal_buffer_id bid_b = ggml_metal_buffer_get_id((ggml_metal_buffer_t) bb->context, b);

    if (bid_a.metal == nullptr || bid_a.metal != bid_b.metal) {
        return false;
    }

    return bid_a.offs <= bid_b.offs
        ? bid_b.offs - bid_a.offs < ggml_nbytes(a)
        : bid_a.offs - bid_b.offs < ggml_nbytes(b);
}

// ---- pattern checks ------------------------------------------------------

// NORM/RMS_NORM + MUL + ADD: the weight/bias of each fused step must match the norm input
// width, be contiguous rows, and the fused outputs must stay F32
static bool ggml_metal_fusion_check_norm(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(mode);
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);

    GGML_ASSERT(fusion->ops.size() >= 2);

    if (fusion->id == GGML_METAL_FUSION_NORM_SCALE) {
        GGML_ASSERT(fusion->ops.size() == 2);

        const ggml_tensor * scale = nodes[1];
        if (scale->op != GGML_OP_SCALE || scale->src[0] != nodes[0] || scale->src[1] ||
            scale->type != GGML_TYPE_F32) {
            return false;
        }

        return true;
    }

    for (int j = 1; j < (int) fusion->ops.size(); j++) {
        // MUL/ADD are commutative, the previous node can be either operand
        const ggml_tensor * src = nodes[j]->src[0] == nodes[j - 1] ? nodes[j]->src[1] : nodes[j]->src[0];

        // the weight/bias must have the same row width as the norm input
        if (src->ne[0] != nodes[0]->ne[0]) {
            return false;
        }

        if (!ggml_is_contiguous_rows(src)) {
            return false;
        }

        if (nodes[j]->type != GGML_TYPE_F32) {
            return false;
        }
    }

    return true;
}

// SSM_CONV + UNARY (silu)
static bool ggml_metal_fusion_check_ssm_conv_silu(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);
    GGML_UNUSED(mode);

    const ggml_tensor * conv = nodes[0];
    const ggml_tensor * un   = nodes[1];

    if (conv->op != GGML_OP_SSM_CONV || un->op != GGML_OP_UNARY || un->src[0] != conv || un->src[1]) {
        return false;
    }

    if (ggml_get_unary_op(un) != GGML_UNARY_OP_SILU) {
        return false;
    }

    if (conv->type != GGML_TYPE_F32 || un->type != GGML_TYPE_F32 || !ggml_is_contiguous_rows(un)) {
        return false;
    }

    return true;
}

// ADD x N: each ADD reads the previous ADD as src0, and all addends must share layout
// (and, in FULL mode, live in the same Metal buffer)
static bool ggml_metal_fusion_check_add_chain(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);
    GGML_ASSERT(fusion->ops.size() >= 2);

    for (int j = 1; j < (int) fusion->ops.size(); j++) {
        if (nodes[j]->src[0] != nodes[j - 1]) {
            return false;
        }

        if (!ggml_are_same_layout(nodes[j]->src[1], nodes[j - 1]->src[1])) {
            return false;
        }

        if (mode == GGML_METAL_FUSION_FULL) {
            if (!ggml_metal_fusion_same_buffer(nodes[j]->src[1], nodes[0]->src[1])) {
                return false;
            }
        }
    }

    return true;
}

// GATED_DELTA_NET + CPY: the trailing cpy scatters the gdn state snapshots into the recurrent
// cache, so the gdn kernel writes them straight to the cache and the cpy is elided.
// mirrors ggml_metal_op_can_fuse_gdn_cache (PR #25788). the gdn output has other consumers (the
// attn scores view), so unlike the other patterns this is not an elision chain: the structural
// checks live entirely in this callback (unsafe = true).
static bool ggml_metal_fusion_check_gdn_cache(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);

    const ggml_tensor * gdn = nodes[0];
    const ggml_tensor * cpy = nodes[1];

    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    if (gdn->type != GGML_TYPE_F32 || (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }

    if (cpy->op != GGML_OP_CPY || (cpy->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }

    const int64_t S_v      = gdn->src[2]->ne[0];
    const int64_t H        = gdn->src[2]->ne[1];
    const int64_t n_tokens = gdn->src[2]->ne[2];
    const int64_t n_seqs   = gdn->src[2]->ne[3];
    const int64_t K        = ggml_get_op_params_i32(gdn, 0);
    const size_t  tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    const int64_t D         = S_v * S_v * H;
    const int64_t n_written = std::min<int64_t>(n_tokens, K);

    const ggml_tensor * src = cpy->src[0]; // gdn snapshot tail view
    const ggml_tensor * dst = cpy->src[1]; // cache view

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    if (src->op != GGML_OP_VIEW || src->view_src != gdn ||
        src->view_offs != tail_off || !ggml_is_contiguous(src)) {
        return false;
    }

    const int64_t expected_ne[GGML_MAX_DIMS] = { D, n_seqs, n_written, 1 };
    if (dst->type != GGML_TYPE_F32 ||
        !std::equal(expected_ne, expected_ne + GGML_MAX_DIMS, dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) ||
        dst->nb[1] != ggml_row_size(GGML_TYPE_F32, D)) {
        return false;
    }

    if (mode == GGML_METAL_FUSION_FULL) {
        // the cache must be allocated so the kernel can write straight to its buffer
        if (dst->data == nullptr) {
            return false;
        }
    }

    return true;
}

// situ(gate, up) = (b*tanh(gate/b) + 0)*sigmoid(gate) * (lb*tanh(up/lb) + 0), as built by build_moe_ffn (LLM_FFN_SITU):
// SCALE(g) TANH SCALE SIGMOID(g) MUL SCALE(u) TANH SCALE MUL. Two branches, so the generic chain check does not apply
// (unsafe = true) and the whole validation lives here.
static bool ggml_metal_fusion_check_situ(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(mode);

    const ggml_tensor * s1  = nodes[0];
    const ggml_tensor * t1  = nodes[1];
    const ggml_tensor * s2  = nodes[2];
    const ggml_tensor * sg  = nodes[3];
    const ggml_tensor * act = nodes[4];
    const ggml_tensor * s3  = nodes[5];
    const ggml_tensor * t2  = nodes[6];
    const ggml_tensor * s4  = nodes[7];
    const ggml_tensor * out = nodes[8];

    if (ggml_get_unary_op(t1) != GGML_UNARY_OP_TANH || ggml_get_unary_op(sg) != GGML_UNARY_OP_SIGMOID ||
        ggml_get_unary_op(t2) != GGML_UNARY_OP_TANH) {
        return false;
    }

    const ggml_tensor * g = s1->src[0];
    const ggml_tensor * u = s3->src[0];

    const bool wired =
        t1->src[0] == s1 && s2->src[0] == t1 && sg->src[0] == g &&
        ((act->src[0] == s2 && act->src[1] == sg) || (act->src[0] == sg && act->src[1] == s2)) &&
        t2->src[0] == s3 && s4->src[0] == t2 &&
        ((out->src[0] == act && out->src[1] == s4) || (out->src[0] == s4 && out->src[1] == act));
    if (!wired || g == nullptr || u == nullptr) {
        return false;
    }

    // everything F32 and element-wise over one [ne0, ne1, ne2] shape; the kernel walks g and u by row strides
    const ggml_tensor * all[] = { g, u, s1, t1, s2, sg, act, s3, t2, s4, out };
    for (const ggml_tensor * t : all) {
        if (t->type != GGML_TYPE_F32 || !ggml_are_same_shape(t, out)) {
            return false;
        }
    }
    if (out->ne[3] != 1 || out->ne[0] % 4 != 0 || !ggml_is_contiguous_rows(g) || !ggml_is_contiguous_rows(u) ||
        !ggml_is_contiguous(out) || g->nb[1] % 16 != 0 || g->nb[2] % 16 != 0 || u->nb[1] % 16 != 0 ||
        u->nb[2] % 16 != 0 || (u->view_offs % 16) != 0 || (g->view_offs % 16) != 0) {
        return false;
    }

    // out is written in its own layout while g and u are read in theirs (row stride 2*ne0): any overlap with the
    // tensor they view is a race, including the in-place case that the generic range check treats as safe
    if (mode == GGML_METAL_FUSION_FULL) {
        auto root = [](const ggml_tensor * t) { while (t->view_src) { t = t->view_src; } return t; };
        auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
            if (!a->data || !b->data) {
                return false;
            }
            const char * a0 = (const char *) a->data; const char * a1 = a0 + ggml_nbytes(a);
            const char * b0 = (const char *) b->data; const char * b1 = b0 + ggml_nbytes(b);
            return a0 < b1 && b0 < a1;
        };
        if (overlap(out, root(g)) || overlap(out, root(u))) {
            return false;
        }
    }

    // only out leaves the group
    int outputs[1] = { node_idxs[idx + 8] };
    return ggml_can_fuse_subgraph_ext(gf, node_idxs + idx, 9, fusion->ops.data(), outputs, 1);
}

// row L2-norm clip, as built by inkling2 clip(): y = x*clamp(thr/(sqrt(sum(x*x)) + eps), lo, hi), with an optional
// per-channel scale x = cur*scale in front. Validated here (unsafe = true): x feeds both SQR and the last MUL.
static bool ggml_metal_fusion_check_clip_rows(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(mode);

    const int  n     = (int) fusion->ops.size();
    const bool scaled = n == 9;
    const int  o     = scaled ? 1 : 0;

    const ggml_tensor * mul0  = scaled ? nodes[0] : nullptr;
    const ggml_tensor * sq    = nodes[o + 0];
    const ggml_tensor * sr    = nodes[o + 1];
    const ggml_tensor * rt    = nodes[o + 2];
    const ggml_tensor * sc    = nodes[o + 3];
    const ggml_tensor * fill  = nodes[o + 4];
    const ggml_tensor * div   = nodes[o + 5];
    const ggml_tensor * clamp = nodes[o + 6];
    const ggml_tensor * out   = nodes[o + 7];

    const ggml_tensor * x = scaled ? mul0 : sq->src[0];
    if (x == nullptr) {
        return false;
    }

    const bool wired =
        sq->src[0] == x && sr->src[0] == sq && rt->src[0] == sr && sc->src[0] == rt && fill->src[0] == sc &&
        div->src[0] == fill && div->src[1] == sc && clamp->src[0] == div &&
        out->src[0] == x && out->src[1] == clamp;
    if (!wired) {
        return false;
    }

    const ggml_tensor * cur   = scaled ? mul0->src[0] : x;
    const ggml_tensor * scale = scaled ? mul0->src[1] : nullptr;

    // F32 rows of a multiple of 4 floats, at most 32 float4 (one SIMD group per row, like kernel_sum_rows_f32_f32_4)
    const ggml_tensor * f32s[] = { cur, x, sq, sr, rt, sc, fill, div, clamp, out };
    for (const ggml_tensor * t : f32s) {
        if (t->type != GGML_TYPE_F32) {
            return false;
        }
    }
    if (x->ne[0] % 4 != 0 || x->ne[0] > 128 || x->ne[3] != 1 || !ggml_are_same_shape(x, out) ||
        !ggml_are_same_shape(cur, x) || !ggml_are_same_shape(x, sq) ||
        sr->ne[0] != 1 || sr->ne[1] != x->ne[1] || sr->ne[2] != x->ne[2] ||
        !ggml_are_same_shape(sr, rt) || !ggml_are_same_shape(sr, sc) || !ggml_are_same_shape(sr, fill) ||
        !ggml_are_same_shape(sr, div) || !ggml_are_same_shape(sr, clamp) ||
        !ggml_is_contiguous_rows(cur) || !ggml_is_contiguous(out)) {
        return false;
    }
    if (scaled) {
        if (scale->type != GGML_TYPE_F32 || scale->ne[0] != x->ne[0] || scale->ne[1] != x->ne[1] ||
            scale->ne[2] != 1 || scale->ne[3] != 1 || !ggml_is_contiguous(scale) || mul0->src[0] != cur) {
            return false;
        }
    }

    int outputs[1] = { node_idxs[idx + n - 1] };
    return ggml_can_fuse_subgraph_ext(gf, node_idxs + idx, n, fusion->ops.data(), outputs, 1);
}

// inkling2 control(): out = (c - tanh(f)*c) + tanh(g)*m, as TANH(f) MUL SUB TANH(g) MUL ADD (validated here, unsafe = true)
static bool ggml_metal_fusion_check_ctrl_mix(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(mode);

    const ggml_tensor * tf = nodes[0];
    const ggml_tensor * m1 = nodes[1];
    const ggml_tensor * sb = nodes[2];
    const ggml_tensor * tg = nodes[3];
    const ggml_tensor * m2 = nodes[4];
    const ggml_tensor * ad = nodes[5];

    if (ggml_get_unary_op(tf) != GGML_UNARY_OP_TANH || ggml_get_unary_op(tg) != GGML_UNARY_OP_TANH) {
        return false;
    }
    // the kernel computes forget*c, gain*m and sub + mul in exactly these operand orders
    if (m1->src[0] != tf || sb->src[1] != m1 || m1->src[1] != sb->src[0] || m2->src[0] != tg ||
        ad->src[0] != sb || ad->src[1] != m2) {
        return false;
    }

    const ggml_tensor * c = sb->src[0];
    const ggml_tensor * m = m2->src[1];
    const ggml_tensor * f = tf->src[0];
    const ggml_tensor * g = tg->src[0];

    const ggml_tensor * all[] = { c, m, f, g, tf, m1, sb, tg, m2, ad };
    for (const ggml_tensor * t : all) {
        if (t == nullptr || t->type != GGML_TYPE_F32 || !ggml_are_same_shape(t, ad)) {
            return false;
        }
    }
    if (ad->ne[2] != 1 || ad->ne[3] != 1 || !ggml_is_contiguous_rows(c) || !ggml_is_contiguous_rows(m) ||
        !ggml_is_contiguous_rows(f) || !ggml_is_contiguous_rows(g) || !ggml_is_contiguous(ad)) {
        return false;
    }

    int outputs[1] = { node_idxs[idx + 5] };
    return ggml_can_fuse_subgraph_ext(gf, node_idxs + idx, 6, fusion->ops.data(), outputs, 1);
}

// inkling2 convolve() without dynamic taps, n_conv = 4:
//   x = concat(past, cur^T, 0); state <- x[last 3]; out = (cur + ssm_conv(x, w))*scale
// raw: CONCAT VIEW(last) VIEW(cell) CPY SSM_CONV RESHAPE ADD SCALE. The CPY writes through a view of the recurrent
// state, which the generic subgraph check rejects, so the use counts are checked here directly.
static bool ggml_metal_fusion_check_sconv(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    const bool has_res = fusion->ops.size() == 6;
    const ggml_tensor * cat  = nodes[0];
    const ggml_tensor * cpy  = nodes[1];
    const ggml_tensor * conv = nodes[2];
    const ggml_tensor * add  = nodes[3];
    const ggml_tensor * scl  = nodes[4];
    const ggml_tensor * res  = has_res ? nodes[5] : nullptr;
    const ggml_tensor * out  = has_res ? res : scl;

    const int r0 = node_idxs[idx];
    const int r1 = node_idxs[idx + (has_res ? 5 : 4)];
    if (r1 - r0 + 1 != (int) fusion->ops_all.size()) {
        return false;
    }
    for (int r = r0, j = 0; r <= r1; ++r, ++j) {
        if (gf->nodes[r]->op != fusion->ops_all[j]) {
            return false;
        }
    }
    const ggml_tensor * last = gf->nodes[r0 + 1];
    const ggml_tensor * cell = gf->nodes[r0 + 2];
    const ggml_tensor * outv = gf->nodes[r0 + 5];

    const ggml_tensor * past = cat->src[0];
    const ggml_tensor * xt   = cat->src[1];
    const ggml_tensor * w    = conv->src[1];
    const ggml_tensor * cur  = add->src[0];

    if (ggml_get_op_params_i32(cat, 0) != 0 || last->view_src != cat || cpy->src[0] != last || cpy->src[1] != cell ||
        conv->src[0] != cat || outv->src[0] != conv || add->src[1] != outv || scl->src[0] != add) {
        return false;
    }
    // residual: out = resid + scl (either operand order), same shape, contiguous
    const ggml_tensor * resid = nullptr;
    if (has_res) {
        resid = res->src[0] == scl ? res->src[1] : res->src[0];
        if ((res->src[0] != scl && res->src[1] != scl) || resid == nullptr || resid->type != GGML_TYPE_F32 ||
            res->type != GGML_TYPE_F32 || !ggml_are_same_shape(res, scl) || !ggml_are_same_shape(resid, scl) ||
            !ggml_is_contiguous(res) || !ggml_is_contiguous(resid)) {
            return false;
        }
    }

    const int64_t nc = 4;
    const int64_t C  = cat->ne[1];
    const int64_t S  = cat->ne[2];
    const int64_t T  = xt->ne[0];

    const ggml_tensor * f32s[] = { cat, past, xt, w, cur, conv, add, scl, cell };
    for (const ggml_tensor * t : f32s) {
        if (t->type != GGML_TYPE_F32) {
            return false;
        }
    }
    if (past->ne[0] != nc - 1 || past->ne[1] != C || past->ne[2] != S || xt->ne[1] != C || xt->ne[2] != S ||
        cat->ne[0] != nc - 1 + T || cat->ne[3] != 1 || w->ne[0] != nc || w->ne[1] != C || !ggml_is_contiguous(w) ||
        !ggml_is_contiguous(cat) || last->ne[0] != nc - 1 || last->ne[1] != C || last->ne[2] != S ||
        last->view_offs != (size_t) T*sizeof(float) || cell->ne[0] != (nc - 1)*C || cell->ne[1] != S ||
        cell->nb[0] != sizeof(float) || conv->ne[0] != C || conv->ne[1] != T || conv->ne[2] != S ||
        cur->ne[0] != C || cur->ne[1] != T*S || !ggml_is_contiguous(cur) || !ggml_is_contiguous(scl) ||
        !ggml_are_same_shape(add, scl) || !ggml_are_same_shape(add, cur) || past->nb[0] != sizeof(float)) {
        return false;
    }
    // xt is cur^T: element (t, c, s) at cur + (s*T + t)*row + c*4
    if (xt->data != cur->data && mode == GGML_METAL_FUSION_FULL) {
        return false;
    }
    // the token strides only matter for prompt batches (one token: cur may be a row view of a packed product)
    if (xt->nb[1] != sizeof(float) || (T > 1 && (xt->nb[0] != cur->nb[1] || xt->nb[2] != T*cur->nb[1]))) {
        return false;
    }

    // every node except the output (and the side-effect CPY) is consumed only inside the group
    for (int r = r0; r <= r1; ++r) {
        const ggml_tensor * node = gf->nodes[r];
        if (node == out || node == cpy) {
            continue;
        }
        if (node->flags & GGML_TENSOR_FLAG_OUTPUT) {
            return false;
        }
        int uses = 0;
        for (int q = r + 1; q <= r1; ++q) {
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                uses += gf->nodes[q]->src[s] == node;
            }
        }
        if (uses != ggml_node_get_use_count(gf, r)) {
            return false;
        }
    }
    if (ggml_node_get_use_count(gf, r0 + 3) != 0) {
        return false;
    }

    // the kernel writes out and the state; out may only alias cur element for element (in place)
    if (mode == GGML_METAL_FUSION_FULL) {
        auto root = [](const ggml_tensor * t) { while (t->view_src) { t = t->view_src; } return t; };
        auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
            if (!a->data || !b->data) {
                return false;
            }
            const char * a0 = (const char *) a->data; const char * a1 = a0 + ggml_nbytes(a);
            const char * b0 = (const char *) b->data; const char * b1 = b0 + ggml_nbytes(b);
            return a0 < b1 && b0 < a1;
        };
        // out is written element for element in the layout of cur (and of resid): only an exact in-place alias is safe
        if (overlap(out, root(past)) || overlap(out, root(w)) || overlap(out, root(cell)) ||
            overlap(cell, root(past)) || overlap(cell, root(cur)) ||
            (overlap(out, root(cur)) && (out->data != cur->data || T > 1)) ||
            (resid && overlap(cell, root(resid))) ||
            (resid && overlap(out, root(resid)) && out->data != resid->data)) {
            return false;
        }
    }

    return true;
}

// inkling2 rel-pos bias into the per-head flash-attention mask:
//   table = bias_pad^T R - sinks; kq_b = table[:, rel_pos]; mask16 = f16(kq_b + mask)
// raw: MUL_MAT RESHAPE SUB PERMUTE CONT GET_ROWS RESHAPE PERMUTE CONT ADD CPY
static bool ggml_metal_fusion_check_relpos(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    const ggml_tensor * mm    = nodes[0];
    const ggml_tensor * sub   = nodes[1];
    const ggml_tensor * cont1 = nodes[2];
    const ggml_tensor * gr    = nodes[3];
    const ggml_tensor * cont2 = nodes[4];
    const ggml_tensor * add   = nodes[5];
    const ggml_tensor * cpy   = nodes[6];

    const int r0 = node_idxs[idx];
    const int r1 = node_idxs[idx + 6];
    if (r1 - r0 + 1 != (int) fusion->ops_all.size()) {
        return false;
    }
    for (int r = r0, j = 0; r <= r1; ++r, ++j) {
        if (gf->nodes[r]->op != fusion->ops_all[j]) {
            return false;
        }
    }
    const ggml_tensor * sinks_r = gf->nodes[r0 + 1];
    const ggml_tensor * perm1   = gf->nodes[r0 + 3];
    const ggml_tensor * resh2   = gf->nodes[r0 + 6];
    const ggml_tensor * perm2   = gf->nodes[r0 + 7];

    const ggml_tensor * bias  = mm->src[0];
    const ggml_tensor * R     = mm->src[1];
    const ggml_tensor * sinks = sinks_r->src[0];
    const ggml_tensor * relp  = gr->src[1];
    const ggml_tensor * mask  = add->src[1];

    if (sub->src[0] != mm || sub->src[1] != sinks_r || perm1->src[0] != sub || cont1->src[0] != perm1 ||
        gr->src[0] != cont1 || resh2->src[0] != gr || perm2->src[0] != resh2 || cont2->src[0] != perm2 ||
        add->src[0] != cont2 || cpy->src[0] != add || cpy->src[1] != cpy || sinks == nullptr) {
        return false;
    }

    const int64_t NR = bias->ne[0];   // relative features (16)
    const int64_t Ep = bias->ne[1];   // table rows (extent + 1)
    const int64_t H  = R->ne[1];
    const int64_t T  = R->ne[2];
    const int64_t NK = relp->ne[0];

    if (bias->type != GGML_TYPE_F32 || R->type != GGML_TYPE_F32 || sinks->type != GGML_TYPE_F32 || mask->type != GGML_TYPE_F32 ||
        relp->type != GGML_TYPE_I32 || cpy->type != GGML_TYPE_F16 || add->type != GGML_TYPE_F32 ||
        R->ne[0] != NR || NR > 32 || !ggml_is_contiguous(bias) || !ggml_is_contiguous_rows(R) ||
        sinks->ne[0] != H || sinks->ne[1] != T || !ggml_is_contiguous_rows(sinks) ||
        relp->ne[1] != T || relp->nb[0] != sizeof(int32_t) || mm->ne[0] != Ep || mm->ne[1] != H || mm->ne[2] != T ||
        cpy->ne[0] != NK || cpy->ne[2] != H || cpy->ne[1]*cpy->ne[3] != T || !ggml_is_contiguous(cpy) ||
        !ggml_are_same_shape(add, cpy) || mask->ne[0] != NK || mask->nb[0] != sizeof(float) ||
        (mask->ne[1] != 1 && mask->ne[1] != cpy->ne[1]) || mask->ne[2] != 1 || (mask->ne[3] != 1 && mask->ne[3] != cpy->ne[3])) {
        return false;
    }
    // the permutes are the ones inkling2 builds: (1,0,2,3) on the table, (2,0,1,3) on the gathered rows
    if (cont1->ne[0] != H || cont1->ne[1] != Ep || gr->ne[0] != H || gr->ne[1] != NK || cont2->ne[0] != NK ||
        cont2->ne[2] != H) {
        return false;
    }

    for (int r = r0; r < r1; ++r) {
        const ggml_tensor * node = gf->nodes[r];
        if (node->flags & GGML_TENSOR_FLAG_OUTPUT) {
            return false;
        }
        int uses = 0;
        for (int q = r + 1; q <= r1; ++q) {
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                uses += gf->nodes[q]->src[s] == node;
            }
        }
        if (uses != ggml_node_get_use_count(gf, r)) {
            return false;
        }
    }

    if (mode == GGML_METAL_FUSION_FULL) {
        auto root = [](const ggml_tensor * t) { while (t->view_src) { t = t->view_src; } return t; };
        auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
            if (!a->data || !b->data) {
                return false;
            }
            const char * a0 = (const char *) a->data; const char * a1 = a0 + ggml_nbytes(a);
            const char * b0 = (const char *) b->data; const char * b1 = b0 + ggml_nbytes(b);
            return a0 < b1 && b0 < a1;
        };
        const ggml_tensor * ins[] = { bias, R, sinks, relp, mask };
        for (const ggml_tensor * t : ins) {
            if (overlap(cpy, root(t))) {
                return false;
            }
        }
    }

    return true;
}

// inkling2 mix() at one token: raw TRANSPOSE CONT RESHAPE MUL_MAT SOFT_MAX MUL_MAT. The scores product runs as is;
// the transposed copy, the softmax and the second product become one kernel that reads the states directly.
static bool ggml_metal_fusion_check_mix_tail(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    const ggml_tensor * cont = nodes[0];
    const ggml_tensor * mm1  = nodes[1];
    const ggml_tensor * sm   = nodes[2];
    const ggml_tensor * mm2  = nodes[3];

    // the leading TRANSPOSE is a view: the optimizer packs it alone and may move it, so the raw range checked here
    // starts at the CONT (ops_all[1..]) and the transpose is found through the CONT
    const int r0 = node_idxs[idx];
    const int r1 = node_idxs[idx + 3];
    if (r1 - r0 + 2 != (int) fusion->ops_all.size()) {
        return false;
    }
    for (int r = r0, j = 1; r <= r1; ++r, ++j) {
        if (gf->nodes[r]->op != fusion->ops_all[j]) {
            return false;
        }
    }
    const ggml_tensor * tr     = cont->src[0];
    const ggml_tensor * qv     = gf->nodes[r0 + 1];
    const ggml_tensor * states = mm1->src[0];

    if (tr == nullptr || tr->op != GGML_OP_TRANSPOSE || tr->src[0] != states || mm1->src[1] != qv || sm->src[0] != mm1 ||
        sm->src[1] != nullptr || mm2->src[0] != cont || mm2->src[1] != sm || qv->view_src == nullptr) {
        return false;
    }
    float max_bias;
    memcpy(&max_bias, (const float *) sm->op_params + 1, sizeof(float));

    const ggml_tensor * f32s[] = { states, qv, cont, mm1, sm, mm2 };
    for (const ggml_tensor * t : f32s) {
        if (t->type != GGML_TYPE_F32) {
            return false;
        }
    }
    const int64_t E = states->ne[0], nr = states->ne[1], T = states->ne[2];
    if (max_bias != 0.0f || T != 1 || nr < 1 || nr > 8 || states->ne[3] != 1 || !ggml_is_contiguous_rows(states) ||
        mm1->ne[0] != nr || mm2->ne[0] != E || mm2->ne[2] != T || !ggml_is_contiguous(mm2) || !ggml_is_contiguous(mm1) ||
        sm->ne[0] != nr || !ggml_is_contiguous(sm)) {
        return false;
    }

    // the copy, the softmax and the scores are consumed only inside the group (the scores by the softmax)
    for (int r = r0; r < r1; ++r) {
        const ggml_tensor * node = gf->nodes[r];
        if (node->flags & GGML_TENSOR_FLAG_OUTPUT) {
            return false;
        }
        int uses = 0;
        for (int q = r + 1; q <= r1; ++q) {
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                uses += gf->nodes[q]->src[s] == node;
            }
        }
        if (uses != ggml_node_get_use_count(gf, r)) {
            return false;
        }
    }

    if (mode == GGML_METAL_FUSION_FULL) {
        auto root = [](const ggml_tensor * t) { while (t->view_src) { t = t->view_src; } return t; };
        auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
            if (!a->data || !b->data) {
                return false;
            }
            const char * a0 = (const char *) a->data; const char * a1 = a0 + ggml_nbytes(a);
            const char * b0 = (const char *) b->data; const char * b1 = b0 + ggml_nbytes(b);
            return a0 < b1 && b0 < a1;
        };
        // the kernel reads the states and the scores (mm1) while writing mm2
        if (overlap(mm2, root(states)) || overlap(mm2, mm1)) {
            return false;
        }
    }

    return true;
}

// out = c * sigmoid(g), as UNARY(sigmoid) MUL (inkling2 attention output gate)
static bool ggml_metal_fusion_check_gate_mul(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion); GGML_UNUSED(gf); GGML_UNUSED(node_idxs); GGML_UNUSED(idx); GGML_UNUSED(mode);
    const ggml_tensor * sg  = nodes[0];
    const ggml_tensor * mul = nodes[1];
    if (ggml_get_unary_op(sg) != GGML_UNARY_OP_SIGMOID || mul->src[1] != sg) {
        return false;
    }
    const ggml_tensor * c = mul->src[0];
    const ggml_tensor * g = sg->src[0];
    return c->type == GGML_TYPE_F32 && g->type == GGML_TYPE_F32 && sg->type == GGML_TYPE_F32 && mul->type == GGML_TYPE_F32 &&
        ggml_are_same_shape(c, mul) && ggml_are_same_shape(g, mul) && ggml_is_contiguous(c) && ggml_is_contiguous(g) &&
        ggml_is_contiguous(mul) && ggml_nelements(mul) % 4 == 0;
}

// inkling2 convolve() with dynamic taps (n_conv = 4): x = concat(past, cur^T); state <- x[last 3];
// out = ssm_conv(x, w); for lag: out += x_window(lag) * tap(lag, head); res = (cur + out)*scale. Validated here.
static bool ggml_metal_fusion_check_sconv_dyn(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    const int n = (int) fusion->ops.size();   // 17
    const int r0 = node_idxs[idx];
    const int r1 = node_idxs[idx + n - 1];
    if (r1 - r0 + 1 != (int) fusion->ops_all.size()) {
        return false;
    }
    for (int r = r0, j = 0; r <= r1; ++r, ++j) {
        if (gf->nodes[r]->op != fusion->ops_all[j]) {
            return false;
        }
    }
    const ggml_tensor * cat  = nodes[0];
    const ggml_tensor * cpy  = nodes[1];
    const ggml_tensor * conv = nodes[2];
    const ggml_tensor * addc = nodes[n - 2];
    const ggml_tensor * scl  = nodes[n - 1];
    const ggml_tensor * last = gf->nodes[r0 + 1];
    const ggml_tensor * cell = gf->nodes[r0 + 2];
    const ggml_tensor * outv = gf->nodes[r0 + 5];

    const ggml_tensor * past = cat->src[0];
    const ggml_tensor * xt   = cat->src[1];
    const ggml_tensor * w    = conv->src[1];
    const ggml_tensor * cur  = addc->src[0];

    if (ggml_get_op_params_i32(cat, 0) != 0 || last->view_src != cat || cpy->src[0] != last || cpy->src[1] != cell ||
        conv->src[0] != cat || outv->src[0] != conv || scl->src[0] != addc) {
        return false;
    }
    const int64_t C = cat->ne[1], S = cat->ne[2], T = xt->ne[0];
    if (T != 1 || S != 1) {
        return false; // decode only
    }

    // the tap chain: per lag, a CONT of a transposed window view of x, a MUL by a tap view of the projection, an ADD
    const ggml_tensor * acc = outv;
    const ggml_tensor * dyn = nullptr;
    for (int lag = 0; lag < 4; ++lag) {
        const ggml_tensor * ct  = nodes[3 + 3*lag];
        const ggml_tensor * mul = nodes[4 + 3*lag];
        const ggml_tensor * ad  = nodes[5 + 3*lag];
        const ggml_tensor * tr  = ct->src[0];
        const ggml_tensor * win = tr ? tr->src[0] : nullptr;
        if (!tr || tr->op != GGML_OP_TRANSPOSE || !win || win->op != GGML_OP_VIEW || win->view_src != cat ||
            win->view_offs != (size_t) (3 - lag)*sizeof(float)) {
            return false;
        }
        const ggml_tensor * mw = mul->src[0]; // reshape of the window copy
        const ggml_tensor * tp = mul->src[1]; // tap view
        if (!mw || mw->op != GGML_OP_RESHAPE || mw->src[0] != ct || !tp || tp->op != GGML_OP_VIEW) {
            return false;
        }
        const ggml_tensor * d3 = tp->src[0];  // reshape_3d of the projection (itself possibly a view of a packed product)
        if (!d3 || d3->op != GGML_OP_RESHAPE || d3->ne[0] != 4 || tp->view_offs != d3->view_offs + (size_t) lag*sizeof(float)) {
            return false;
        }
        if (dyn == nullptr) {
            dyn = d3;
        } else if (d3 != dyn) {
            return false;
        }
        const ggml_tensor * term = ad->src[1];
        if (ad->src[0] != acc || !term || term->op != GGML_OP_RESHAPE || term->src[0] != mul) {
            return false;
        }
        acc = ad;
    }
    if (addc->src[1] != acc) {
        return false;
    }
    const int64_t H = dyn->ne[1];
    if (H < 1 || C % H != 0) {
        return false;
    }

    const ggml_tensor * f32s[] = { cat, past, xt, w, cur, conv, addc, scl, cell, dyn };
    for (const ggml_tensor * t : f32s) {
        if (t->type != GGML_TYPE_F32) {
            return false;
        }
    }
    if (past->ne[0] != 3 || past->ne[1] != C || cat->ne[0] != 3 + T || w->ne[0] != 4 || w->ne[1] != C ||
        !ggml_is_contiguous(w) || !ggml_is_contiguous(cat) || last->view_offs != (size_t) T*sizeof(float) ||
        cell->ne[0] != 3*C || cell->nb[0] != sizeof(float) || cur->ne[0] != C || cur->ne[1] != T*S ||
        !ggml_is_contiguous(cur) || !ggml_is_contiguous(scl) || past->nb[0] != sizeof(float) ||
        dyn->ne[2] != T*S || dyn->nb[0] != sizeof(float)) {
        return false;
    }
    if (mode == GGML_METAL_FUSION_FULL && xt->data != cur->data) {
        return false;
    }

    for (int r = r0; r <= r1; ++r) {
        const ggml_tensor * node = gf->nodes[r];
        if (node == scl || node == cpy) {
            continue;
        }
        if (node->flags & GGML_TENSOR_FLAG_OUTPUT) {
            return false;
        }
        int uses = 0;
        for (int q = r + 1; q <= r1; ++q) {
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                uses += gf->nodes[q]->src[s] == node;
            }
        }
        if (uses != ggml_node_get_use_count(gf, r)) {
            return false;
        }
    }

    if (mode == GGML_METAL_FUSION_FULL) {
        auto root = [](const ggml_tensor * t) { while (t->view_src) { t = t->view_src; } return t; };
        auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
            if (!a->data || !b->data) {
                return false;
            }
            const char * a0 = (const char *) a->data; const char * a1 = a0 + ggml_nbytes(a);
            const char * b0 = (const char *) b->data; const char * b1 = b0 + ggml_nbytes(b);
            return a0 < b1 && b0 < a1;
        };
        if (overlap(scl, root(past)) || overlap(scl, root(w)) || overlap(scl, root(cell)) || overlap(scl, root(dyn)) ||
            overlap(cell, root(past)) || overlap(cell, root(cur)) || overlap(cell, root(dyn)) ||
            (overlap(scl, root(cur)) && scl->data != cur->data)) {
            return false;
        }
    }
    return true;
}

// common tail of the store fusions: nodes[k] is the SET_ROWS writing the rows of `val` (via a 2D VIEW) into an f16
// cache at the given row indices; every raw node in [r0, r1] except the SET_ROWS is consumed inside the group
static bool ggml_metal_fusion_check_store_tail(const ggml_tensor * val, const ggml_tensor * sr, const ggml_cgraph * gf,
        int r0, int r1) {
    const ggml_tensor * vw = gf->nodes[r1 - 1];
    if (sr != gf->nodes[r1] || vw->op != GGML_OP_VIEW || vw->view_src != val || vw->view_offs != 0 || sr->src[0] != vw) {
        return false;
    }
    const ggml_tensor * idxs  = sr->src[1];
    const ggml_tensor * cache = sr->src[2];
    if (!idxs || !cache || (idxs->type != GGML_TYPE_I64 && idxs->type != GGML_TYPE_I32) || cache->type != GGML_TYPE_F16 ||
        vw->ne[0] != val->ne[0]*val->ne[1] || vw->ne[1] != val->ne[2] || vw->nb[1] != val->nb[2] ||
        cache->ne[0] != vw->ne[0] || cache->nb[0] != sizeof(ggml_fp16_t) || idxs->ne[0] != vw->ne[1] ||
        sr->ne[2] != 1 || sr->ne[3] != 1 || idxs->ne[1] != 1 || !ggml_is_contiguous(val)) {
        return false;
    }
    for (int r = r0; r < r1; ++r) {
        const ggml_tensor * node = gf->nodes[r];
        if (node->flags & GGML_TENSOR_FLAG_OUTPUT) {
            return false;
        }
        int uses = 0;
        for (int q = r + 1; q <= r1; ++q) {
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                uses += gf->nodes[q]->src[s] == node;
            }
        }
        if (uses != ggml_node_get_use_count(gf, r)) {
            return false;
        }
    }
    return ggml_node_get_use_count(gf, r1) == 0;
}

// raw: SQR SUM_ROWS SQRT SCALE FILL DIV CLAMP MUL VIEW SET_ROWS (the K clip of inkling2, then the K cache store)
static bool ggml_metal_fusion_check_clip_store(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(mode);
    const int r0 = node_idxs[idx];
    const int r1 = node_idxs[idx + 8];
    if (r1 - r0 + 1 != (int) fusion->ops_all.size()) {
        return false;
    }
    for (int r = r0, j = 0; r <= r1; ++r, ++j) {
        if (gf->nodes[r]->op != fusion->ops_all[j]) {
            return false;
        }
    }
    const ggml_tensor * sq = nodes[0], * sr = nodes[1], * rt = nodes[2], * sc = nodes[3], * fill = nodes[4];
    const ggml_tensor * div = nodes[5], * clamp = nodes[6], * out = nodes[7];
    const ggml_tensor * x = sq->src[0];
    if (!x || sr->src[0] != sq || rt->src[0] != sr || sc->src[0] != rt || fill->src[0] != sc || div->src[0] != fill ||
        div->src[1] != sc || clamp->src[0] != div || out->src[0] != x || out->src[1] != clamp) {
        return false;
    }
    const ggml_tensor * f32s[] = { x, sq, sr, rt, sc, fill, div, clamp, out };
    for (const ggml_tensor * t : f32s) {
        if (t->type != GGML_TYPE_F32) {
            return false;
        }
    }
    if (x->ne[0] % 4 != 0 || x->ne[0] > 128 || x->ne[3] != 1 || !ggml_are_same_shape(x, out) || !ggml_is_contiguous_rows(x) ||
        sr->ne[0] != 1 || sr->ne[1] != x->ne[1] || sr->ne[2] != x->ne[2]) {
        return false;
    }
    return ggml_metal_fusion_check_store_tail(out, nodes[8], gf, r0, r1);
}

// raw: RMS_NORM VIEW SET_ROWS (the V norm of inkling2, then the V cache store)
static bool ggml_metal_fusion_check_norm_store(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(mode);
    const int r0 = node_idxs[idx];
    const int r1 = node_idxs[idx + 1];
    if (r1 - r0 + 1 != (int) fusion->ops_all.size()) {
        return false;
    }
    for (int r = r0, j = 0; r <= r1; ++r, ++j) {
        if (gf->nodes[r]->op != fusion->ops_all[j]) {
            return false;
        }
    }
    const ggml_tensor * nrm = nodes[0];
    const ggml_tensor * x   = nrm->src[0];
    if (!x || x->type != GGML_TYPE_F32 || nrm->type != GGML_TYPE_F32 || nrm->src[1] != nullptr ||
        x->ne[0] != 128 || x->ne[3] != 1 || !ggml_is_contiguous_rows(x) || !ggml_are_same_shape(x, nrm)) {
        return false;
    }
    return ggml_metal_fusion_check_store_tail(nrm, nodes[1], gf, r0, r1);
}

// res*scale[ids] after an NVFP4 MUL_MAT_ID on the matvec path (llm_graph_context::build_lora_mm_id with w_s):
// nodes = MUL_MAT_ID, [REPEAT], GET_ROWS, MUL; the scale is read straight from w_s by expert id in the kernel
static bool ggml_metal_fusion_check_mv_id_scale(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    const int n = (int) fusion->ops.size(); // 3 or 4
    const ggml_tensor * mm  = nodes[0];
    const ggml_tensor * gr  = nodes[n - 2];
    const ggml_tensor * mul = nodes[n - 1];
    const ggml_tensor * ids = mm->src[2];

    if (mm->src[0]->type != GGML_TYPE_NVFP4 || mm->src[1]->type != GGML_TYPE_F32 || mm->type != GGML_TYPE_F32 ||
        ggml_metal_op_mul_mat_id_use_mm(mm, true) || !ggml_is_contiguous(mm) || mm->flags & GGML_TENSOR_FLAG_OUTPUT) {
        return false;
    }
    if (gr->src[1] != ids || mul->type != GGML_TYPE_F32 || !ggml_are_same_shape(mul, mm) || !ggml_is_contiguous(mul)) {
        return false;
    }
    if (!((mul->src[0] == mm && mul->src[1] == gr) || (mul->src[0] == gr && mul->src[1] == mm))) {
        return false;
    }
    // the scale vector: GET_ROWS reads RESHAPE(w_s) directly, or through REPEAT at more than one token
    const ggml_tensor * sv = gr->src[0];
    if (n == 4) {
        const ggml_tensor * rp = nodes[1];
        if (sv != rp || rp->op != GGML_OP_REPEAT) {
            return false;
        }
        sv = rp->src[0];
    }
    if (sv->op != GGML_OP_RESHAPE || sv->type != GGML_TYPE_F32 || !ggml_is_contiguous(sv) ||
        ggml_nelements(sv) != mm->src[0]->ne[2] || sv->ne[0] != 1) {
        return false;
    }
    for (int i = 0; i < n - 1; ++i) { // only the MUL leaves the group
        if (ggml_node_get_use_count(gf, node_idxs[idx + i]) != 1) {
            return false;
        }
    }
    if (mode == GGML_METAL_FUSION_FULL) {
        auto root = [](const ggml_tensor * t) { while (t->view_src) { t = t->view_src; } return t; };
        auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
            if (!a->data || !b->data) {
                return false;
            }
            const char * a0 = (const char *) a->data; const char * a1 = a0 + ggml_nbytes(a);
            const char * b0 = (const char *) b->data; const char * b1 = b0 + ggml_nbytes(b);
            return a0 < b1 && b0 < a1;
        };
        // mul is written while the activation and the ids are read
        if (overlap(mul, root(mm->src[1])) || overlap(mul, root(ids))) {
            return false;
        }
    }
    GGML_UNUSED(gf);
    return true;
}

// inkling2 mix_states: states = concat(block_stack, rms_norm(cur) as [E, 1, T], dim 1). The normed rows are a group
// output too (norm_once shares them with later stacks), so the kernel writes both; unsafe = true, validated here
static bool ggml_metal_fusion_check_norm_concat(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    const ggml_tensor * nrm = nodes[0];
    const ggml_tensor * cat = nodes[1];
    const ggml_tensor * x   = nrm->src[0];
    const ggml_tensor * stk = cat->src[0];
    const ggml_tensor * nv  = cat->src[1];

    auto root = [](const ggml_tensor * t) { while (t->view_src) { t = t->view_src; } return t; };

    if (ggml_get_op_params_i32(cat, 0) != 1 || root(nv) != nrm || (nv != nrm && nv->op != GGML_OP_RESHAPE)) {
        return false;
    }
    const int64_t E = x->ne[0];
    const int64_t T = x->ne[1];
    if (x->type != GGML_TYPE_F32 || nrm->type != GGML_TYPE_F32 || stk->type != GGML_TYPE_F32 || cat->type != GGML_TYPE_F32 ||
        E % 4 != 0 || x->ne[2] != 1 || x->ne[3] != 1 || x->nb[0] != sizeof(float) || x->nb[1] % 16 != 0 ||
        !ggml_is_contiguous(nrm) || !ggml_is_contiguous(cat) ||
        stk->ne[0] != E || stk->ne[2] != T || stk->ne[3] != 1 || stk->nb[0] != sizeof(float) || stk->nb[1] % 16 != 0 ||
        stk->nb[2] % 16 != 0 || cat->ne[0] != E || cat->ne[1] != stk->ne[1] + 1 || cat->ne[2] != T || cat->ne[3] != 1 ||
        nv->ne[0] != E || nv->ne[1] != 1 || nv->ne[2] != T) {
        return false;
    }
    if (cat->flags & GGML_TENSOR_FLAG_OUTPUT) {
        return false;
    }
    if (mode == GGML_METAL_FUSION_FULL) {
        auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
            if (!a->data || !b->data) {
                return false;
            }
            const char * a0 = (const char *) a->data; const char * a1 = a0 + ggml_nbytes(a);
            const char * b0 = (const char *) b->data; const char * b1 = b0 + ggml_nbytes(b);
            return a0 < b1 && b0 < a1;
        };
        // cat is written while x and the stack are read; nrm is written while the stack is read
        if (overlap(cat, root(x)) || overlap(cat, root(stk)) || overlap(cat, nrm) || overlap(nrm, root(stk))) {
            return false;
        }
    }
    GGML_UNUSED(fusion);
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);
    return true;
}

// out = W^T [a; b] at one token, W F32 on the kernel_mul_mv_f32_f32_4 path (nsg 4, nr0 2): replicated exactly with the
// vector read from a and b; requires the a|b boundary on a 16-float chunk boundary
static bool ggml_metal_fusion_check_cat_mv(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    const ggml_tensor * cat = nodes[0];
    const ggml_tensor * mm  = nodes[1];
    if (mm->src[1] != cat || ggml_get_op_params_i32(cat, 0) != 0) {
        return false;
    }
    const ggml_tensor * a = cat->src[0];
    const ggml_tensor * b = cat->src[1];
    const ggml_tensor * w = mm->src[0];
    if (!a || !b || a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 || cat->type != GGML_TYPE_F32 ||
        w->type != GGML_TYPE_F32 || mm->type != GGML_TYPE_F32 ||
        cat->ne[1] != 1 || cat->ne[2] != 1 || cat->ne[3] != 1 || ggml_nelements(a) != a->ne[0] || ggml_nelements(b) != b->ne[0] ||
        a->ne[0] % 16 != 0 || b->ne[0] % 16 != 0 || a->nb[0] != sizeof(float) || b->nb[0] != sizeof(float) ||
        w->ne[0] != cat->ne[0] || w->ne[0] < 512 || w->ne[0] % 32 != 0 || w->ne[2] != 1 || w->ne[3] != 1 || w->nb[0] != sizeof(float) ||
        w->nb[1] % 16 != 0 || mm->ne[0] != w->ne[1] || mm->ne[1] != 1 || mm->ne[2] != 1 || mm->ne[3] != 1 || !ggml_is_contiguous(mm)) {
        return false;
    }
    if ((((uintptr_t) a->data) % 16) != 0 || (((uintptr_t) b->data) % 16) != 0) {
        if (mode == GGML_METAL_FUSION_FULL) {
            return false;
        }
    }
    if (cat->flags & GGML_TENSOR_FLAG_OUTPUT || ggml_node_get_use_count(gf, node_idxs[idx]) != 1) {
        return false;
    }
    if (mode == GGML_METAL_FUSION_FULL) {
        auto root = [](const ggml_tensor * t) { while (t->view_src) { t = t->view_src; } return t; };
        auto overlap = [](const ggml_tensor * x, const ggml_tensor * y) {
            if (!x->data || !y->data) {
                return false;
            }
            const char * x0 = (const char *) x->data; const char * x1 = x0 + ggml_nbytes(x);
            const char * y0 = (const char *) y->data; const char * y1 = y0 + ggml_nbytes(y);
            return x0 < y1 && y0 < x1;
        };
        if (overlap(mm, root(a)) || overlap(mm, root(b))) {
            return false;
        }
    }
    GGML_UNUSED(fusion);
    return true;
}

// MUL + SIN + SQR + MUL + ADD (snake activation)
static bool ggml_metal_fusion_check_snake(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(mode);
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);

    const ggml_tensor * mul0     = nodes[0];
    const ggml_tensor * sin_node = nodes[1];
    const ggml_tensor * sqr      = nodes[2];
    const ggml_tensor * mul1     = nodes[3];
    const ggml_tensor * add      = nodes[4];

    // x carries the full activation shape, a is the broadcast operand
    const ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
    const ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

    // mul1 reads sqr and inv_b in either operand order
    const ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

    // closure check: the trailing add reads the same x as the leading mul
    const ggml_tensor * x_in_add = (add->src[0] == mul1) ? add->src[1] : add->src[0];

    // x is in the supported whitelist and every chain intermediate shares x's type.
    // a and inv_b bind as device const float * in the kernel, so they stay F32.
    const bool types_ok =
        (x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_BF16) &&
        (a->type    == GGML_TYPE_F32) && (inv_b->type    == GGML_TYPE_F32) &&
        (mul0->type == x->type)       && (sin_node->type == x->type) &&
        (sqr->type  == x->type)       && (mul1->type     == x->type) &&
        (add->type  == x->type);

    // a / inv_b collapse to [1, C, 1, 1], x and add stay 2D
    const bool shape_ok = ggml_are_same_shape(a, inv_b) && a->ne[0] == 1 && a->ne[1] == x->ne[1];
    const bool dim_ok =
        (x->ne[2]     == 1) && (x->ne[3]     == 1) &&
        (add->ne[2]   == 1) && (add->ne[3]   == 1) &&
        (a->ne[2]     == 1) && (a->ne[3]     == 1) &&
        (inv_b->ne[2] == 1) && (inv_b->ne[3] == 1);

    // kernel reads x[idx] and a[c] / inv_b[c] linearly, so every operand is contiguous
    const bool contig_ok =
        ggml_is_contiguous(x) && ggml_is_contiguous(add) &&
        ggml_is_contiguous(a) && ggml_is_contiguous(inv_b);

    return types_ok && shape_ok && dim_ok && contig_ok && x_in_add == x;
}

#define GGML_METAL_TOPK_MOE_MAX_EXPERTS 1024

static const std::vector<ggml_op> ops_topk_moe = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS
};
static const std::vector<ggml_op> ops_topk_moe_scale = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_SCALE
};
static const std::vector<ggml_op> ops_topk_moe_norm = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE
};
static const std::vector<ggml_op> ops_topk_moe_norm_scale = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE
};
static const std::vector<ggml_op> ops_topk_moe_norm_bias = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_SCALE, GGML_OP_DIV, GGML_OP_RESHAPE
};
static const std::vector<ggml_op> ops_topk_moe_norm_scale_bias = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_SCALE, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE
};

static const std::vector<ggml_op> ops_topk_moe_sp = {
    GGML_OP_UNARY, GGML_OP_SQRT, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS
};
static const std::vector<ggml_op> ops_topk_moe_sp_scale = {
    GGML_OP_UNARY, GGML_OP_SQRT, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_SCALE
};
static const std::vector<ggml_op> ops_topk_moe_sp_norm = {
    GGML_OP_UNARY, GGML_OP_SQRT, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE
};
static const std::vector<ggml_op> ops_topk_moe_sp_norm_scale = {
    GGML_OP_UNARY, GGML_OP_SQRT, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE
};
static const std::vector<ggml_op> ops_topk_moe_sp_norm_bias = {
    GGML_OP_UNARY, GGML_OP_SQRT, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_SCALE, GGML_OP_DIV, GGML_OP_RESHAPE
};
static const std::vector<ggml_op> ops_topk_moe_sp_norm_scale_bias = {
    GGML_OP_UNARY, GGML_OP_SQRT, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_SCALE, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE
};

static const std::vector<ggml_op> ops_topk_moe_sig = {
    GGML_OP_UNARY, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS
};
static const std::vector<ggml_op> ops_topk_moe_sig_scale = {
    GGML_OP_UNARY, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_SCALE
};
static const std::vector<ggml_op> ops_topk_moe_sig_norm = {
    GGML_OP_UNARY, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE
};
static const std::vector<ggml_op> ops_topk_moe_sig_norm_scale = {
    GGML_OP_UNARY, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE
};
static const std::vector<ggml_op> ops_topk_moe_sig_norm_bias = {
    GGML_OP_UNARY, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_SCALE, GGML_OP_DIV, GGML_OP_RESHAPE
};
static const std::vector<ggml_op> ops_topk_moe_sig_norm_scale_bias = {
    GGML_OP_UNARY, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_SCALE, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE
};

static bool ggml_metal_fusion_check_topk_moe(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_ASSERT(fusion->ops.size() >= 3);
    GGML_UNUSED(nodes);

    const int n_ops = (int) fusion->ops.size();

    bool with_norm = false;
    for (int j = 0; j < n_ops; ++j) {
        with_norm |= fusion->ops[j] == GGML_OP_SUM_ROWS;
    }
    const bool with_scale = fusion->ops[n_ops - 1] == GGML_OP_SCALE;

    // the fusion table operates on the non-empty node sequence; the raw graph also
    // contains the RESHAPE/VIEW nodes that the fused kernel elides.
    const std::vector<ggml_op> & ops_all = fusion->ops_all;

    const int raw_start = node_idxs[idx];
    int       raw_end   = node_idxs[idx + n_ops - 1];

    // the norm variant ends with a RESHAPE that the non-empty sequence filters out;
    // include it so the output use-count check sees the real final routing tensor
    if (with_norm && !with_scale) {
        if (raw_end + 1 >= gf->n_nodes) {
            return false;
        }
        const ggml_tensor * trailing_reshape = gf->nodes[raw_end + 1];
        if (trailing_reshape->op != GGML_OP_RESHAPE || trailing_reshape->src[0] != gf->nodes[raw_end]) {
            return false;
        }
        raw_end++;
    }

    const int raw_count = raw_end - raw_start + 1;
    if (raw_count != (int) ops_all.size()) {
        return false;
    }

    int raw_idxs[GGML_METAL_FUSION_MAX];
    for (int i = 0; i < raw_count; ++i) {
        raw_idxs[i] = raw_start + i;
        if (gf->nodes[raw_start + i]->op != ops_all[i]) {
            return false;
        }
    }

    const ggml_tensor * first = gf->nodes[raw_start];
    const ggml_tensor * out   = gf->nodes[raw_end];

    // the unary chains start at the sigmoid or the softplus node and rank biased scores
    const bool with_unary = first->op == GGML_OP_UNARY;
    const bool with_sp    = with_unary && ggml_get_unary_op(first) == GGML_UNARY_OP_SOFTPLUS;
    const bool with_sig   = with_unary && ggml_get_unary_op(first) == GGML_UNARY_OP_SIGMOID;

    if (with_unary && !with_sp && !with_sig) {
        return false;
    }

    const ggml_tensor * probs;          // the scoring output
    const ggml_tensor * scores;         // the node the argsort ranks (adds the bias for the unary chains)
    const ggml_tensor * probs_reshaped;
    const ggml_tensor * argsort;
    const ggml_tensor * ids;
    const ggml_tensor * get_rows;
    const ggml_tensor * logits = first->src[0];
    int ids_off;

    if (with_unary) {
        // the softplus chains add a SQRT node, the sigmoid chains start at the sigmoid node
        const int base = with_sp ? 1 : 0;

        probs          = gf->nodes[raw_start + base + 0];
        probs_reshaped = gf->nodes[raw_start + base + 1];
        scores         = gf->nodes[raw_start + base + 2];
        argsort        = gf->nodes[raw_start + base + 3];
        ids            = gf->nodes[raw_start + base + 4];
        get_rows       = gf->nodes[raw_start + base + 5];
        ids_off        = base + 4;

        if (scores->op != GGML_OP_ADD || scores->src[0] != probs ||
            (with_sp && (probs->op != GGML_OP_SQRT || probs->src[0] != first))) {
            return false;
        }

        // the selection bias is a per-expert vector shared by all rows
        const ggml_tensor * bias = scores->src[1];
        if (bias->type != GGML_TYPE_F32 || scores->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias) ||
            bias->ne[0] != probs->ne[0] || bias->ne[1] != 1 || bias->ne[2] != 1 || bias->ne[3] != 1) {
            return false;
        }
    } else {
        probs          = first;
        scores         = first;
        probs_reshaped = gf->nodes[raw_start + 1];
        argsort        = gf->nodes[raw_start + 2];
        ids            = gf->nodes[raw_start + 3];
        get_rows       = gf->nodes[raw_start + 4];
        ids_off        = 3;

        // the fused kernel implements plain softmax only
        float scale    = 1.0f;
        float max_bias = 0.0f;
        memcpy(&scale,    ((const int32_t *) first->op_params) + 0, sizeof(scale));
        memcpy(&max_bias, ((const int32_t *) first->op_params) + 1, sizeof(max_bias));
        if (scale != 1.0f || max_bias != 0.0f || first->src[1] || first->src[2]) {
            return false;
        }
    }

    if (logits->type != GGML_TYPE_F32 || probs->type != GGML_TYPE_F32 ||
        out->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) {
        return false;
    }

    const int64_t n_expert      = logits->ne[0];
    const int64_t n_tokens      = logits->ne[1];
    const int64_t n_expert_used = ids->ne[0];

    // note: n_tokens == 0 (no-output batch) must match so that the packing stays shape-independent
    if (n_expert <= 0 || n_expert_used <= 0 || n_expert_used > n_expert ||
        n_expert > GGML_METAL_TOPK_MOE_MAX_EXPERTS || n_expert_used > GGML_METAL_TOPK_MOE_MAX_EXPERTS) {
        return false;
    }

    if (logits->ne[2] != 1 || logits->ne[3] != 1 ||
        ids->ne[1] != n_tokens || ids->ne[2] != 1 || ids->ne[3] != 1 ||
        out->ne[0] != 1 || out->ne[1] != n_expert_used || out->ne[2] != n_tokens || out->ne[3] != 1) {
        return false;
    }

    if (!ggml_is_contiguous(logits) || !ggml_is_contiguous(out) ||
        ids->nb[0] != ggml_type_size(GGML_TYPE_I32) ||
        ids->nb[1] != ggml_type_size(GGML_TYPE_I32) * n_expert) {
        return false;
    }

    // the top-k gathers the unbiased scores, the argsort ranks the biased ones
    if (probs_reshaped->src[0] != probs || argsort->src[0] != scores ||
        ids->src[0] != argsort || get_rows->src[0] != probs_reshaped || get_rows->src[1] != ids) {
        return false;
    }

    if (with_norm) {
        const int norm_off = with_unary ? 6 + (with_sp ? 1 : 0) : 5;

        const ggml_tensor * weights_reshaped = gf->nodes[raw_start + norm_off + 0];
        const ggml_tensor * sum_rows         = gf->nodes[raw_start + norm_off + 1];
        const ggml_tensor * denom            = gf->nodes[raw_start + norm_off + 2];
        const ggml_tensor * div              = gf->nodes[raw_start + norm_off + 3];
        const ggml_tensor * out_reshaped     = gf->nodes[raw_start + norm_off + 4];

        if (weights_reshaped->src[0] != get_rows || sum_rows->src[0] != weights_reshaped ||
            denom->src[0] != sum_rows || div->src[0] != weights_reshaped || div->src[1] != denom ||
            out_reshaped->src[0] != div) {
            return false;
        }

        // the kernel applies the sum adjustment as a lower clamp or an additive bias, never a scale
        if (denom->op == GGML_OP_SCALE && ggml_get_op_params_f32(denom, 0) != 1.0f) {
            return false;
        }

        if (with_scale && out->src[0] != out_reshaped) {
            return false;
        }
    } else if (with_scale && out->src[0] != get_rows) {
        return false;
    }

    const int outputs[2] = { raw_start + ids_off, raw_end };
    if (!ggml_can_fuse_subgraph_ext(gf, raw_idxs, raw_count, ops_all.data(), outputs, 2)) {
        return false;
    }

    if (mode == GGML_METAL_FUSION_FULL) {
        if (!logits->data || !out->data || !ids->data) {
            return false;
        }
    }

    return true;
}

#define GGML_METAL_MOE_REDUCE_MAX_EXPERTS 8

struct ggml_metal_moe_reduce_match {
    const ggml_tensor * experts;
    const ggml_tensor * weights;
    const ggml_tensor * dst;
    int node_count;
};

static bool ggml_metal_fusion_match_moe_reduce(
        const ggml_cgraph * gf, int node_idx, const std::vector<ggml_op> & ops_all,
        ggml_metal_moe_reduce_match * match) {
    if (match == nullptr || node_idx < 0 || node_idx + (int) ops_all.size() > gf->n_nodes) {
        return false;
    }

    const ggml_tensor * mul = gf->nodes[node_idx];
    if (mul->op != GGML_OP_MUL || mul->type != GGML_TYPE_F32) {
        return false;
    }

    // MUL, then one VIEW per expert, then one ADD per additional expert
    const int raw_count     = (int) ops_all.size();
    const int n_expert_used = raw_count / 2;

    if (n_expert_used < 2 || n_expert_used > GGML_METAL_MOE_REDUCE_MAX_EXPERTS ||
        raw_count != 2 * n_expert_used) {
        return false;
    }

    int n_views = 0;
    while (node_idx + 1 + n_views < gf->n_nodes &&
           gf->nodes[node_idx + 1 + n_views]->op == GGML_OP_VIEW) {
        n_views++;
    }

    if (n_views != n_expert_used) {
        return false;
    }

    for (int i = n_expert_used + 1; i < raw_count; ++i) {
        if (gf->nodes[node_idx + i]->op != GGML_OP_ADD) {
            return false;
        }
    }

    int raw_idxs[GGML_METAL_FUSION_MAX];
    for (int i = 0; i < raw_count; ++i) {
        raw_idxs[i] = node_idx + i;
        if (gf->nodes[node_idx + i]->op != ops_all[i]) {
            return false;
        }
    }

    const ggml_tensor * experts = mul->src[0];
    const ggml_tensor * weights = mul->src[1];
    const ggml_tensor * dst     = gf->nodes[node_idx + raw_count - 1];

    if (experts->type != GGML_TYPE_F32 || weights->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }

    const int64_t n_embd   = experts->ne[0];
    const int64_t n_tokens = experts->ne[2];

    // note: n_tokens == 0 (no-output batch) must match so that the packing stays shape-independent
    if (n_embd <= 0 || experts->ne[1] != n_expert_used || experts->ne[3] != 1 ||
        weights->ne[0] != 1 || weights->ne[1] != n_expert_used || weights->ne[2] != n_tokens || weights->ne[3] != 1 ||
        dst->ne[0] != n_embd || dst->ne[1] != n_tokens || dst->ne[2] != 1 || dst->ne[3] != 1) {
        return false;
    }

    if (!ggml_is_contiguous(experts) || !ggml_is_contiguous(weights) || !ggml_is_contiguous(dst)) {
        return false;
    }

    for (int i = 1; i <= n_expert_used; ++i) {
        const ggml_tensor * view = gf->nodes[node_idx + i];
        if (view->view_src != mul || view->src[0] != mul ||
            view->view_offs != (size_t) (i - 1) * mul->nb[1] ||
            view->ne[0] != n_embd || view->ne[1] != n_tokens ||
            view->nb[1] != mul->nb[2]) {
            return false;
        }
    }

    const ggml_tensor * prev_add = nullptr;
    for (int j = 1; j < n_expert_used; ++j) {
        const ggml_tensor * add = gf->nodes[node_idx + n_expert_used + j];
        const ggml_tensor * rhs = gf->nodes[node_idx + j + 1];
        const ggml_tensor * lhs = j == 1 ? gf->nodes[node_idx + 1] : prev_add;
        if (add->src[0] != lhs || add->src[1] != rhs) {
            return false;
        }
        prev_add = add;
    }

    const int outputs[1] = { node_idx + raw_count - 1 };
    if (!ggml_can_fuse_subgraph_ext(gf, raw_idxs, raw_count, ops_all.data(), outputs, 1)) {
        return false;
    }

    match->experts    = experts;
    match->weights    = weights;
    match->dst        = dst;
    match->node_count = raw_count;
    return true;
}

static bool ggml_metal_fusion_check_moe_reduce(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(nodes);

    ggml_metal_moe_reduce_match match;
    if (!ggml_metal_fusion_match_moe_reduce(gf, node_idxs[idx], fusion->ops_all, &match)) {
        return false;
    }

    if ((int) fusion->ops.size() != match.experts->ne[1]) {
        return false;
    }

    const int raw_end = node_idxs[idx] + match.node_count - 1;
    if (node_idxs[idx + (int) fusion->ops.size() - 1] != raw_end) {
        return false;
    }

    if (mode == GGML_METAL_FUSION_FULL) {
        if (!match.experts->data || !match.weights->data || !match.dst->data) {
            return false;
        }

        // the fused kernel reads experts and weights while writing dst, so dst must not overlap either input
        if (ggml_metal_fusion_overlap(match.dst, match.experts) || ggml_metal_fusion_overlap(match.dst, match.weights)) {
            return false;
        }
    }

    return true;
}

// true if t is or views a tensor in a buffer marked as weights, such as a bias; the model loader marks its buffers before
// any graph is optimized, and tensors in unmarked or not yet allocated buffers count as non-weights in both phases
static bool ggml_metal_tensor_is_weight(const struct ggml_tensor * t) {
    const ggml_tensor * base = t->view_src != NULL ? t->view_src : t;

    return base->buffer != NULL && ggml_backend_buffer_get_usage(base->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

static const struct ggml_tensor * ggml_metal_mul_mat_add_operand(const struct ggml_tensor * mm, const struct ggml_tensor * add) {
    if (add->op != GGML_OP_ADD || (add->src[0] == mm) == (add->src[1] == mm)) {
        return NULL;
    }

    const ggml_tensor * other = add->src[0] == mm ? add->src[1] : add->src[0];

    const bool ok = other->type == GGML_TYPE_F32 && add->type == GGML_TYPE_F32 && !ggml_metal_tensor_is_weight(other);

    return ok ? other : NULL;
}

static const struct ggml_tensor * ggml_metal_mul_mat_add_residual(const struct ggml_tensor * mm, const struct ggml_tensor * add) {
    const ggml_tensor * res = ggml_metal_mul_mat_add_operand(mm, add);

    const bool ok = res != NULL && ggml_are_same_shape(res, mm) &&
        ggml_is_contiguous(res) && ggml_is_contiguous(mm) && ggml_is_contiguous(add);

    return ok ? res : NULL;
}

// MUL_MAT + ADD of an f32 non-weight: the reorder packs it without reading row counts, so ubatch sizes share one order;
// the encoder fuses only a same-shape residual in the few-row MMA store, which the sum may overlap only in place
static bool ggml_metal_fusion_check_mul_mat_add(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);
    GGML_UNUSED(fusion);

    const ggml_tensor * mm  = nodes[0];
    const ggml_tensor * add = nodes[1];

    if (ggml_metal_mul_mat_add_operand(mm, add) == nullptr ||
        !ggml_metal_op_mul_mat_may_use_mma(mm)) {
        return false;
    }

    if (mode == GGML_METAL_FUSION_STRUCTURAL) {
        return true;
    }

    const ggml_tensor * res = ggml_metal_mul_mat_add_residual(mm, add);

    if (res == nullptr || !ggml_metal_op_mul_mat_use_mma(mm)) {
        return false;
    }

    return !ggml_metal_fusion_overlap(add, mm->src[0]) && !ggml_metal_fusion_overlap(add, mm->src[1]) &&
        (add->data == res->data || !ggml_metal_fusion_overlap(add, res));
}

// ---- patterns ------------------------------------------------------------

static const std::vector<ggml_op> ops_norm_mul         = { GGML_OP_NORM, GGML_OP_MUL };
static const std::vector<ggml_op> ops_norm_mul_add     = { GGML_OP_NORM, GGML_OP_MUL, GGML_OP_ADD };
static const std::vector<ggml_op> ops_norm_scale       = { GGML_OP_NORM, GGML_OP_SCALE };
static const std::vector<ggml_op> ops_rms_norm_mul     = { GGML_OP_RMS_NORM, GGML_OP_MUL };
static const std::vector<ggml_op> ops_rms_norm_mul_add = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD };
static const std::vector<ggml_op> ops_rms_norm_scale   = { GGML_OP_RMS_NORM, GGML_OP_SCALE };

static const std::vector<ggml_op> ops_add_2 = { GGML_OP_ADD, GGML_OP_ADD };
static const std::vector<ggml_op> ops_add_3 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const std::vector<ggml_op> ops_add_4 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const std::vector<ggml_op> ops_add_5 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const std::vector<ggml_op> ops_add_6 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const std::vector<ggml_op> ops_add_7 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const std::vector<ggml_op> ops_snake = { GGML_OP_MUL, GGML_OP_SIN, GGML_OP_SQR, GGML_OP_MUL, GGML_OP_ADD };
static const std::vector<ggml_op> ops_clip_rows        = { GGML_OP_SQR, GGML_OP_SUM_ROWS, GGML_OP_SQRT, GGML_OP_SCALE,
                                                          GGML_OP_FILL, GGML_OP_DIV, GGML_OP_CLAMP, GGML_OP_MUL };
static const std::vector<ggml_op> ops_clip_rows_scaled = { GGML_OP_MUL, GGML_OP_SQR, GGML_OP_SUM_ROWS, GGML_OP_SQRT, GGML_OP_SCALE,
                                                          GGML_OP_FILL, GGML_OP_DIV, GGML_OP_CLAMP, GGML_OP_MUL };
// raw order includes the VIEW of up (gate_up second half), which the graph visits right before its SCALE
// raw order: the message VIEW (m) is visited right before its MUL when nothing read it earlier
static const std::vector<ggml_op> ops_ctrl_mix      = { GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_SUB, GGML_OP_UNARY, GGML_OP_VIEW, GGML_OP_MUL, GGML_OP_ADD };
static const std::vector<ggml_op> ops_ctrl_mix_nov  = { GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_SUB, GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_ADD };
static const std::vector<ggml_op> ops_sconv = { GGML_OP_CONCAT, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_CPY, GGML_OP_SSM_CONV,
                                                GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_SCALE };
static const std::vector<ggml_op> ops_sconv_res = { GGML_OP_CONCAT, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_CPY, GGML_OP_SSM_CONV,
                                                    GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_SCALE, GGML_OP_ADD };
static const std::vector<ggml_op> ops_relpos = { GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_SUB, GGML_OP_PERMUTE, GGML_OP_CONT,
                                                 GGML_OP_GET_ROWS, GGML_OP_RESHAPE, GGML_OP_PERMUTE, GGML_OP_CONT, GGML_OP_ADD, GGML_OP_CPY };
static const std::vector<ggml_op> ops_mix_tail = { GGML_OP_TRANSPOSE, GGML_OP_CONT, GGML_OP_RESHAPE, GGML_OP_MUL_MAT, GGML_OP_SOFT_MAX, GGML_OP_MUL_MAT };
static const std::vector<ggml_op> ops_gate_mul = { GGML_OP_UNARY, GGML_OP_MUL };
static const std::vector<ggml_op> ops_cat_mv          = { GGML_OP_CONCAT, GGML_OP_MUL_MAT };
static const std::vector<ggml_op> ops_norm_concat     = { GGML_OP_RMS_NORM, GGML_OP_CONCAT };
static const std::vector<ggml_op> ops_norm_concat_r   = { GGML_OP_RMS_NORM, GGML_OP_RESHAPE, GGML_OP_CONCAT };
static const std::vector<ggml_op> ops_mv_id_scale     = { GGML_OP_MUL_MAT_ID, GGML_OP_RESHAPE, GGML_OP_GET_ROWS, GGML_OP_MUL };
static const std::vector<ggml_op> ops_mv_id_scale_rep = { GGML_OP_MUL_MAT_ID, GGML_OP_RESHAPE, GGML_OP_REPEAT, GGML_OP_GET_ROWS, GGML_OP_MUL };
// raw: CONCAT VIEW VIEW CPY SSM_CONV RESHAPE, lag 0 (window VIEW TRANSPOSE CONT RESHAPE, RESHAPE of the projection, tap VIEW,
// MUL RESHAPE ADD), lags 1-3 (the projection reshape is visited already), ADD SCALE
static const std::vector<ggml_op> ops_sconv_dyn = { GGML_OP_CONCAT, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_CPY, GGML_OP_SSM_CONV, GGML_OP_RESHAPE,
    GGML_OP_VIEW, GGML_OP_TRANSPOSE, GGML_OP_CONT, GGML_OP_RESHAPE, GGML_OP_RESHAPE, GGML_OP_VIEW, GGML_OP_MUL, GGML_OP_RESHAPE, GGML_OP_ADD,
    GGML_OP_VIEW, GGML_OP_TRANSPOSE, GGML_OP_CONT, GGML_OP_RESHAPE, GGML_OP_VIEW, GGML_OP_MUL, GGML_OP_RESHAPE, GGML_OP_ADD,
    GGML_OP_VIEW, GGML_OP_TRANSPOSE, GGML_OP_CONT, GGML_OP_RESHAPE, GGML_OP_VIEW, GGML_OP_MUL, GGML_OP_RESHAPE, GGML_OP_ADD,
    GGML_OP_VIEW, GGML_OP_TRANSPOSE, GGML_OP_CONT, GGML_OP_RESHAPE, GGML_OP_VIEW, GGML_OP_MUL, GGML_OP_RESHAPE, GGML_OP_ADD,
    GGML_OP_ADD, GGML_OP_SCALE };
static const std::vector<ggml_op> ops_clip_store = { GGML_OP_SQR, GGML_OP_SUM_ROWS, GGML_OP_SQRT, GGML_OP_SCALE, GGML_OP_FILL,
                                                     GGML_OP_DIV, GGML_OP_CLAMP, GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_SET_ROWS };
static const std::vector<ggml_op> ops_norm_store = { GGML_OP_RMS_NORM, GGML_OP_VIEW, GGML_OP_SET_ROWS };
static const std::vector<ggml_op> ops_situ  = { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_MUL,
                                                GGML_OP_VIEW, GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_MUL };

static const std::vector<ggml_op> ops_gdn_cache = { GGML_OP_GATED_DELTA_NET, GGML_OP_CPY };

static const std::vector<ggml_op> ops_ssm_conv_silu = { GGML_OP_SSM_CONV, GGML_OP_UNARY };

static const std::vector<ggml_op> ops_moe_reduce_2 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_ADD
};
static const std::vector<ggml_op> ops_moe_reduce_3 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_ADD, GGML_OP_ADD
};
static const std::vector<ggml_op> ops_moe_reduce_4 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
static const std::vector<ggml_op> ops_moe_reduce_5 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
static const std::vector<ggml_op> ops_moe_reduce_6 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
static const std::vector<ggml_op> ops_moe_reduce_7 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
static const std::vector<ggml_op> ops_moe_reduce_8 = {
    GGML_OP_MUL,
    GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};

static const std::vector<ggml_op> ops_mul_mat_add = { GGML_OP_MUL_MAT, GGML_OP_ADD };

static const std::vector<ggml_metal_fusion> ggml_metal_fusions = {
    { GGML_METAL_FUSION_NORM_MUL,     ops_norm_mul,         {}, false, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL_ADD, ops_norm_mul_add,     {}, false, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_SCALE,   ops_norm_scale,       {}, false, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL,     ops_rms_norm_mul,     {}, false, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL_ADD, ops_rms_norm_mul_add, {}, false, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_SCALE,   ops_rms_norm_scale,   {}, false, ggml_metal_fusion_check_norm },

    { GGML_METAL_FUSION_ADD_CHAIN, ops_add_2, {}, false, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN, ops_add_3, {}, false, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN, ops_add_4, {}, false, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN, ops_add_5, {}, false, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN, ops_add_6, {}, false, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN, ops_add_7, {}, false, ggml_metal_fusion_check_add_chain },

    { GGML_METAL_FUSION_SNAKE, ops_snake, {}, false, ggml_metal_fusion_check_snake },
    { GGML_METAL_FUSION_SITU,  ops_situ,  {}, true,  ggml_metal_fusion_check_situ  },
    { GGML_METAL_FUSION_CLIP_STORE, ops_clip_store, {}, true, ggml_metal_fusion_check_clip_store },
    { GGML_METAL_FUSION_NORM_STORE, ops_norm_store, {}, true, ggml_metal_fusion_check_norm_store },
    { GGML_METAL_FUSION_SCONV_DYN, ops_sconv_dyn, {}, true, ggml_metal_fusion_check_sconv_dyn },
    { GGML_METAL_FUSION_MIX_TAIL, ops_mix_tail, {}, true,  ggml_metal_fusion_check_mix_tail },
    { GGML_METAL_FUSION_GATE_MUL, ops_gate_mul, {}, false, ggml_metal_fusion_check_gate_mul },
    { GGML_METAL_FUSION_CAT_MV,      ops_cat_mv,        {}, true, ggml_metal_fusion_check_cat_mv },
    { GGML_METAL_FUSION_NORM_CONCAT, ops_norm_concat,   {}, true, ggml_metal_fusion_check_norm_concat },
    { GGML_METAL_FUSION_NORM_CONCAT, ops_norm_concat_r, {}, true, ggml_metal_fusion_check_norm_concat },
    { GGML_METAL_FUSION_MV_ID_SCALE, ops_mv_id_scale,     {}, true, ggml_metal_fusion_check_mv_id_scale },
    { GGML_METAL_FUSION_MV_ID_SCALE, ops_mv_id_scale_rep, {}, true, ggml_metal_fusion_check_mv_id_scale },
    { GGML_METAL_FUSION_RELPOS, ops_relpos, {}, true, ggml_metal_fusion_check_relpos },
    { GGML_METAL_FUSION_SCONV, ops_sconv, {}, true,  ggml_metal_fusion_check_sconv },
    { GGML_METAL_FUSION_SCONV, ops_sconv_res, {}, true, ggml_metal_fusion_check_sconv },
    { GGML_METAL_FUSION_CTRL_MIX, ops_ctrl_mix,     {}, true, ggml_metal_fusion_check_ctrl_mix },
    { GGML_METAL_FUSION_CTRL_MIX, ops_ctrl_mix_nov, {}, true, ggml_metal_fusion_check_ctrl_mix },
    { GGML_METAL_FUSION_CLIP_ROWS, ops_clip_rows,        {}, true, ggml_metal_fusion_check_clip_rows },
    { GGML_METAL_FUSION_CLIP_ROWS, ops_clip_rows_scaled, {}, true, ggml_metal_fusion_check_clip_rows },

    { GGML_METAL_FUSION_GDN_CACHE, ops_gdn_cache, {}, true, ggml_metal_fusion_check_gdn_cache },

    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe,                     {1}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_scale,               {1}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_norm,                {1}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_norm_scale,          {1}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_norm_bias,           {1}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_norm_scale_bias,     {1}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sp,                  {3}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sp_scale,            {3}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sp_norm,             {3}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sp_norm_scale,       {3}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sp_norm_bias,        {3}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sp_norm_scale_bias,  {3}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sig,                 {2}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sig_scale,           {2}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sig_norm,            {2}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sig_norm_scale,      {2}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sig_norm_bias,       {2}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sig_norm_scale_bias, {2}, true, ggml_metal_fusion_check_topk_moe },

    { GGML_METAL_FUSION_MOE_REDUCE, ops_moe_reduce_2, {}, true, ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE, ops_moe_reduce_3, {}, true, ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE, ops_moe_reduce_4, {}, true, ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE, ops_moe_reduce_5, {}, true, ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE, ops_moe_reduce_6, {}, true, ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE, ops_moe_reduce_7, {}, true, ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE, ops_moe_reduce_8, {}, true, ggml_metal_fusion_check_moe_reduce },

    { GGML_METAL_FUSION_SSM_CONV_SILU, ops_ssm_conv_silu, {}, false, ggml_metal_fusion_check_ssm_conv_silu },

    { GGML_METAL_FUSION_MUL_MAT_ADD, ops_mul_mat_add, {}, false, ggml_metal_fusion_check_mul_mat_add },
};

// ---- alloc deps -----------------------------------------------------------

static bool ggml_metal_fusion_match_raw_pattern(
        const ggml_cgraph * gf, int node_idx, const std::vector<ggml_op> & ops) {
    if (node_idx < 0 || node_idx + (int) ops.size() > gf->n_nodes) {
        return false;
    }

    for (int i = 0; i < (int) ops.size(); ++i) {
        if (gf->nodes[node_idx + i]->op != ops[i]) {
            return false;
        }
    }

    return true;
}

static void ggml_metal_fusion_add_pattern_alloc_deps(
        void * user_data,
        void (*add_alloc_dep)(void *, ggml_tensor *, ggml_tensor *),
        ggml_cgraph * gf,
        const ggml_metal_fusion * fusion,
        int node_idx) {
    const int last_node = node_idx + (int) fusion->ops_all.size() - 1;

    // keep all external inputs alive until the fused output
    std::set<ggml_tensor *> seen;
    for (int j = 0; j < (int) fusion->ops_all.size(); ++j) {
        ggml_tensor * node = gf->nodes[node_idx + j];
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            ggml_tensor * src = node->src[s];
            if (src && seen.insert(src).second) {
                add_alloc_dep(user_data, src, gf->nodes[last_node]);
            }
        }
        seen.insert(node);
    }

    // the mix tail kernel reads the scores product (an inner node) while writing the output: keep it alive too
    if (fusion->id == GGML_METAL_FUSION_MIX_TAIL) {
        add_alloc_dep(user_data, gf->nodes[node_idx + 3], gf->nodes[last_node]);
    }
}

void ggml_metal_fusion_add_alloc_deps(
        void * user_data,
        void (*add_alloc_dep)(void *, ggml_tensor *, ggml_tensor *),
        ggml_cgraph * gf) {
    for (int i = 0; i < gf->n_nodes; ++i) {
        const ggml_metal_fusion * best = nullptr;
        int best_raw = 0;

        for (const ggml_metal_fusion & fusion : ggml_metal_fusions) {
            if ((int) fusion.ops_all.size() <= best_raw) {
                continue;
            }
            if (ggml_metal_fusion_match_raw_pattern(gf, i, fusion.ops_all)) {
                best = &fusion;
                best_raw = (int) fusion.ops_all.size();
            }
        }

        if (best) {
            ggml_metal_fusion_add_pattern_alloc_deps(user_data, add_alloc_dep, gf, best, i);
            i += best_raw - 1;
        }
    }
}

// ---- shared fusion info ---------------------------------------------------

static std::string ggml_metal_fusion_label(const ggml_metal_fusion * fusion) {
    GGML_ASSERT(fusion != nullptr);

    std::string label;
    for (int j = 0; j < (int) fusion->ops.size(); j++) {
        if (j > 0) {
            label += '+';
        }
        label += ggml_op_name(fusion->ops[j]);
    }
    return label;
}

struct ggml_metal_fusion_info {
    std::vector<std::string> labels;
    std::vector<uint64_t>    counts;
    bool enabled;
    bool stats;
    bool labels_set;
    int  debug;
};

struct ggml_metal_fusion_info * ggml_metal_fusion_info_init(bool enabled, int debug) {
    ggml_metal_fusion_info * finfo = new ggml_metal_fusion_info;
    finfo->enabled    = enabled;
    finfo->stats      = debug > 0;
    finfo->labels_set = false;
    finfo->debug      = debug;

    if (finfo->stats) {
        ggml_metal_fusion_info_labels_init(finfo);
    }

    return finfo;
}

void ggml_metal_fusion_info_free(ggml_metal_fusion_info * finfo) {
    delete finfo;
}

bool ggml_metal_fusion_info_enabled(const ggml_metal_fusion_info * finfo) {
    return finfo->enabled;
}

bool ggml_metal_fusion_info_stats(const ggml_metal_fusion_info * finfo) {
    return finfo->stats;
}

int ggml_metal_fusion_info_debug(const ggml_metal_fusion_info * finfo) {
    return finfo->debug;
}

int ggml_metal_fusion_info_n_fusions(const ggml_metal_fusion_info * finfo) {
    return (int) finfo->labels.size();
}

const char * ggml_metal_fusion_info_label(const ggml_metal_fusion_info * finfo, int idx) {
    GGML_ASSERT(idx >= 0 && idx < (int) finfo->labels.size());
    return finfo->labels[idx].c_str();
}

uint64_t ggml_metal_fusion_info_count(const ggml_metal_fusion_info * finfo, int idx) {
    GGML_ASSERT(idx >= 0 && idx < (int) finfo->counts.size());
    return finfo->counts[idx];
}

void ggml_metal_fusion_info_count_fusion(ggml_metal_fusion_info * finfo, const ggml_metal_fusion * fusion) {
    if (!finfo->stats || fusion == nullptr) {
        return;
    }

    const ptrdiff_t idx = fusion - ggml_metal_fusions.data();
    if (idx >= 0 && idx < (ptrdiff_t) finfo->counts.size()) {
        finfo->counts[idx]++;
    }
}

void ggml_metal_fusion_info_set_enabled(ggml_metal_fusion_info * finfo, bool enabled) {
    finfo->enabled = enabled;
}

void ggml_metal_fusion_info_labels_init(ggml_metal_fusion_info * finfo) {
    if (finfo->labels_set) {
        return;
    }

    finfo->labels.clear();
    finfo->counts.assign(ggml_metal_fusions.size(), 0);
    finfo->labels.reserve(ggml_metal_fusions.size());

    for (const ggml_metal_fusion & fusion : ggml_metal_fusions) {
        finfo->labels.emplace_back(ggml_metal_fusion_label(&fusion));
    }

    finfo->labels_set = true;
}

void ggml_metal_fusion_info_stats_init(ggml_metal_fusion_info * finfo) {
    finfo->stats = true;
    ggml_metal_fusion_info_labels_init(finfo);
}

void ggml_metal_fusion_info_stats_reset(ggml_metal_fusion_info * finfo) {
    std::fill(finfo->counts.begin(), finfo->counts.end(), 0);
}

int ggml_metal_fusion_info_stats_get(const ggml_metal_fusion_info * finfo, const char ** labels, uint64_t * counts, int n) {
    const int n_fusions = (int) finfo->labels.size();

    if (labels == nullptr) {
        return n_fusions;
    }

    const int n_fill = std::min(n, n_fusions);
    for (int i = 0; i < n_fill; i++) {
        labels[i] = finfo->labels[i].c_str();
        if (counts != nullptr) {
            counts[i] = finfo->counts[i];
        }
    }

    return n_fill;
}

// ---- memory-range checks -------------------------------------------------

// reject fusions where an external source overlaps any fused output. the fused
// kernels elide intermediate nodes, so only sources that are not part of the
// fused subgraph can cause read/write races with the output.
static bool ggml_metal_fusion_check_memory_ranges(
        const ggml_metal_fusion      * fusion,
        const ggml_tensor * const    * nodes,
        int                            node_count) {
    // some fused kernels write through a tensor that also appears as a source (e.g. the gdn
    // cache cpy), so a source that is the same memory as the output is not an external read
    // source
    auto same_memory = [](const ggml_tensor * a, const ggml_tensor * b) {
        if (a->data && b->data && a->data == b->data) {
            return true;
        }
        for (const ggml_tensor * v = a; v; v = v->view_src) {
            if (v == b) {
                return true;
            }
        }
        for (const ggml_tensor * v = b; v; v = v->view_src) {
            if (v == a) {
                return true;
            }
        }
        return false;
    };

    auto nodes_overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
        if (!a || !b || !a->data || !b->data || !a->buffer || !b->buffer) {
            return false;
        }

        if (a->buffer != b->buffer) {
            return false;
        }

        const int64_t a_start = (int64_t) a->data;
        const int64_t a_end   = a_start + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);
        const int64_t b_start = (int64_t) b->data;
        const int64_t b_end   = b_start + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);

        return (b_start <= a_start && a_start < b_end) ||
               (a_start <= b_start && b_start < a_end);
    };

    auto is_intermediate = [](const ggml_tensor * src, const ggml_tensor * const * nodes, int j) {
        for (int k = 0; k < j; ++k) {
            if (src == nodes[k]) {
                return true;
            }
            for (const ggml_tensor * view_src = src->view_src; view_src; view_src = view_src->view_src) {
                if (view_src == nodes[k]) {
                    return true;
                }
            }
        }
        return false;
    };

    auto check_dst = [&](const ggml_tensor * dst) {
        for (int j = 0; j < node_count; ++j) {
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                const ggml_tensor * src = nodes[j]->src[s];
                if (!src || src->op == GGML_OP_NONE || same_memory(src, dst)) {
                    continue;
                }

                if (nodes_overlap(dst, src) && !is_intermediate(src, nodes, j)) {
                    return false;
                }
            }
        }
        return true;
    };

    if (!check_dst(nodes[node_count - 1])) {
        return false;
    }

    for (int offset : fusion->outs) {
        GGML_ASSERT(offset >= 0 && offset < node_count);
        if (!check_dst(nodes[offset])) {
            return false;
        }
    }

    return true;
}

// ---- queries -------------------------------------------------------------

// find the longest pattern matching the node sequence starting at idx
// (idx is a position in node_idxs, which maps to graph node indices)
const ggml_metal_fusion * ggml_metal_fusion_next(
        const ggml_cgraph * gf,
        const int * node_idxs,
        int n_idxs,
        int idx,
        ggml_metal_fusion_mode mode,
        int * n_out) {
    const ggml_metal_fusion * res = nullptr;
    int best = 1;

    for (const ggml_metal_fusion & fusion : ggml_metal_fusions) {
        const int n_ops = (int) fusion.ops.size();

        // only look for a longer match than the current best
        if (n_ops <= best) {
            continue;
        }
        if (idx + n_ops > n_idxs) {
            continue;
        }

        const ggml_tensor * nodes[GGML_METAL_FUSION_MAX];

        // the op sequence must match exactly
        bool ok = true;
        for (int j = 0; j < n_ops; j++) {
            nodes[j] = gf->nodes[node_idxs[idx + j]];
            if (nodes[j]->op != fusion.ops[j]) {
                ok = false;
                break;
            }
        }
        if (!ok) {
            continue;
        }

        if (!fusion.unsafe) {
            // common element-wise chain constraints: each node reads the previous one,
            // and all nodes have the same shape
            for (int j = 1; j < n_ops && ok; j++) {
                if (nodes[j]->src[0] != nodes[j - 1] && nodes[j]->src[1] != nodes[j - 1]) {
                    ok = false;
                    break;
                }
                if (!ggml_are_same_shape(nodes[j], nodes[j - 1])) {
                    ok = false;
                    break;
                }
            }
            if (!ok) {
                continue;
            }

            // primary output is the last node; additional outputs come from fusion.outs
            int outputs_buf[GGML_METAL_FUSION_MAX];
            outputs_buf[0] = node_idxs[idx + n_ops - 1];
            for (size_t i = 0; i < fusion.outs.size(); ++i) {
                const int out_offset = fusion.outs[i];
                GGML_ASSERT(out_offset >= 0 && out_offset < n_ops);
                outputs_buf[i + 1] = node_idxs[idx + out_offset];
            }

            const int n_outputs = 1 + (int) fusion.outs.size();

            // structural subgraph checks (op sequence, elidable uses, view containment)
            if (!ggml_can_fuse_subgraph_ext(gf, node_idxs + idx, n_ops, fusion.ops.data(), outputs_buf, n_outputs)) {
                continue;
            }
        }

        // pattern-specific checks (the sole validator for unsafe patterns)
        if (fusion.check && !fusion.check(&fusion, nodes, gf, node_idxs, idx, mode)) {
            continue;
        }

        // the compute phase has allocated tensors and can detect aliasing between
        // external sources and fused outputs; the optimizer phase cannot do this yet
        if (mode == GGML_METAL_FUSION_FULL &&
                !ggml_metal_fusion_check_memory_ranges(&fusion, nodes, n_ops)) {
            continue;
        }

        best = n_ops;
        res = &fusion;
    }

    *n_out = best;

    return res;
}

// optimize phase: maximum number of nodes starting at idx (a raw sequential graph index) that
// could be fused, chaining patterns back-to-back. matching runs on the same filtered (view
// transparent) node sequence that the compute phase uses, so the returned count is the raw index
// span from idx to the last matched node (intermediate views are packed along).
int ggml_metal_fusion_max(const ggml_cgraph * gf, int idx) {
    // a view node cannot start a pattern - pack it alone
    if (ggml_op_is_empty(gf->nodes[idx]->op)) {
        return 1;
    }

    // collect the non-view node indices starting at idx; 0-element tensors are included so
    // that empty graphs pack like their non-empty counterparts (see ggml_metal_fusion_filter_ops)
    int idxs[GGML_METAL_FUSION_MAX];
    int n_idxs = 0;
    for (int i = idx; i < gf->n_nodes && n_idxs < GGML_METAL_FUSION_MAX; i++) {
        if (!ggml_op_is_empty(gf->nodes[i]->op)) {
            idxs[n_idxs++] = i;
        }
    }
    if (n_idxs == 0) {
        return 1;
    }

    int total = 0;
    int i_f = 0;

    while (i_f < n_idxs && total < GGML_METAL_FUSION_MAX) {
        int len = 1;
        const ggml_metal_fusion * fusion = ggml_metal_fusion_next(gf, idxs, n_idxs, i_f, GGML_METAL_FUSION_STRUCTURAL, &len);
        if (!fusion || total + len > GGML_METAL_FUSION_MAX) {
            break;
        }
        // a single pattern may span up to GGML_METAL_FUSION_MAX raw nodes; chaining more patterns behind it stops at
        // GGML_METAL_FUSION_CHAIN_MAX, and never cuts a pattern (a cut tail is packed apart and can be reordered)
        if (i_f > 0 && idxs[i_f + len - 1] - idx + 1 > GGML_METAL_FUSION_CHAIN_MAX) {
            break;
        }

        total += len;
        i_f += len;
    }

    if (i_f == 0) {
        return 1;
    }

    // map the matched non-empty nodes back to the raw index span (views are included)
    return std::min(GGML_METAL_FUSION_MAX, idxs[i_f - 1] - idx + 1);
}
