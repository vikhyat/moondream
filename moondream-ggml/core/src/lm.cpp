#include <cassert>
#include <cstring>
#include <climits>
#include <cmath>
#include <queue>
#include <stdexcept>

#include "helpers.hpp"
#include "unicode.hpp"
#include "lm.hpp"

static ggml_tensor * lm_build_inp_embd(
    ggml_context * ctx,
    moondream_lm_context & mctx,
    const moondream_lm_hparams & hparams,
    const moondream_lm_batch & batch,
    ggml_tensor * tok_embd
) {
    // TODO: what does the L stand for?
    ggml_tensor * inpL;

    // If batch has tokens (integers) then set inp_tokens to the input and
    // take the embeddings from tok_embd, otherwise use the token embeddings
    // (inp_embd) and set them as the input.
    if (batch.token) {
        mctx.inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.n_tokens);
        ggml_set_input(mctx.inp_tokens);
        inpL = ggml_get_rows(ctx, tok_embd, mctx.inp_tokens);
    } else {
        mctx.inp_embd = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hparams.n_embd, batch.n_tokens);
        inpL = mctx.inp_embd;
        ggml_set_input(mctx.inp_embd);
    }
    return inpL;
}

static ggml_tensor * lm_build_inp_pos(
    ggml_context * ctx,
    moondream_lm_context & mctx,
    moondream_lm_batch & batch
) {
    mctx.inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.n_tokens);
    ggml_set_input(mctx.inp_pos);
    return mctx.inp_pos;
}

static ggml_tensor * lm_build_inp_kq_mask(
    ggml_context * ctx,
    moondream_lm_context & mctx,
    moondream_lm_batch & batch,
    moondream_lm_cparams & cparams,
    int32_t n_kv
) {
    if (cparams.causal_attn) {
        mctx.inp_kq_mask = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, n_kv, GGML_PAD(batch.n_tokens, GGML_KQ_MASK_PAD)
        );
    } else {
        mctx.inp_kq_mask = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, batch.n_tokens, GGML_PAD(batch.n_tokens, GGML_KQ_MASK_PAD)
        );
    }
    ggml_set_input(mctx.inp_kq_mask);
    return cparams.flash_attn ? ggml_cast(ctx, mctx.inp_kq_mask, GGML_TYPE_F16) : mctx.inp_kq_mask;
};

static ggml_tensor * lm_build_norm(
    ggml_context * ctx,
    ggml_tensor * cur,
    moondream_lm_hparams & hparams,
    ggml_tensor * mw,
    ggml_tensor * mb,
    int il
) {
    cur = ggml_norm(ctx, cur, hparams.f_norm_eps);
    // Weight
    cur = ggml_mul(ctx, cur, mw);
    // Bias
    if (mb) {
        set_tensor_name(cur, "norm_w", il);
        cur = ggml_add(ctx, cur, mb);
    }
    return cur;
}

static void lm_build_kv_cache(
    ggml_context * ctx,
    moondream_lm_hparams & hparams,
    moondream_lm_cparams & cparams,
    moondream_kv_cache & kv,
    ggml_cgraph * graph,
    ggml_tensor * k_cur,
    ggml_tensor * v_cur,
    int32_t n_tokens,
    int32_t kv_head,
    int il
) {
    const int64_t n_ctx = cparams.n_ctx;
    const int64_t n_embd = hparams.n_embd;

    GGML_ASSERT(kv.size == n_ctx);

    ggml_tensor * k_cache_view = ggml_view_1d(
        ctx, kv.k_l[il], n_tokens*n_embd,
        (ggml_row_size(kv.k_l[il]->type, n_embd))*kv_head
    );
    set_tensor_name(k_cache_view, "k_cache_view", il);
    ggml_build_forward_expand(graph, ggml_cpy(ctx, k_cur, k_cache_view));

    assert(v_cur->ne[0] == n_embd && v_cur->ne[1] == n_tokens);

    ggml_tensor * v_cache_view = nullptr;
    if (cparams.flash_attn) {
        v_cache_view = ggml_view_1d(
            ctx, kv.v_l[il], n_tokens*n_embd,
            (kv_head)*ggml_row_size(kv.v_l[il]->type, n_embd)
        );
    } else {
        // The v cache is transposed when not using flash attention.
        v_cache_view = ggml_view_2d(
            ctx, kv.v_l[il], n_tokens, n_embd,
            (n_ctx)*ggml_element_size(kv.v_l[il]),
            (kv_head)*ggml_element_size(kv.v_l[il])
        );
        v_cur = ggml_transpose(ctx, v_cur);
    }
    set_tensor_name(v_cache_view, "v_cache_view", il);
    ggml_build_forward_expand(graph, ggml_cpy(ctx, v_cur, v_cache_view));
}

static ggml_tensor * lm_build_kqv(
    ggml_context * ctx,
    moondream_lm & model,
    moondream_lm_hparams & hparams,
    moondream_lm_cparams & cparams,
    moondream_kv_cache & kv,
    ggml_cgraph * graph,
    ggml_tensor * wo,
    ggml_tensor * wo_b,
    ggml_tensor * q_cur,
    ggml_tensor * kq_mask,
    int32_t n_tokens,
    int32_t n_kv,
    float kq_scale,
    int il
) {
    const int64_t n_ctx = cparams.n_ctx;
    const int64_t n_head = hparams.n_head;
    const int64_t n_head_kv = hparams.n_head_kv;
    const int64_t n_embd = hparams.n_embd;
    const int64_t n_embd_head_k = hparams.n_embd_head_k;
    const int64_t n_embd_head_v = hparams.n_embd_head_v;

    ggml_tensor * q = ggml_permute(ctx, q_cur, 0, 2, 1, 3);
    ggml_tensor * k = ggml_view_3d(
        ctx, kv.k_l[il],
        n_embd_head_k, n_kv, n_head_kv,
        ggml_row_size(kv.k_l[il]->type, n_embd),
        ggml_row_size(kv.k_l[il]->type, n_embd_head_k),
        0
    );
    set_tensor_name(k, "k", il);

    ggml_tensor * cur;
    if (cparams.flash_attn) {
        // Split cached v into n_head heads (not transposed).
        ggml_tensor * v = ggml_view_3d(
            ctx, kv.v_l[il],
            n_embd_head_v, n_kv, n_head_kv,
            ggml_row_size(kv.v_l[il]->type, n_embd),
            ggml_row_size(kv.v_l[il]->type, n_embd_head_v),
            0
        );
        set_tensor_name(v, "v", il);
        cur = ggml_flash_attn_ext(ctx, q, k, v, kq_mask, kq_scale, hparams.f_max_alibi_bias);
        // For phi2 the KQ multiplication must be done with F32 precision, otherwise we get NaNs.
        // Ref: https://github.com/ggerganov/llama.cpp/pull/4490#issuecomment-1859055847
        ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);
        cur = ggml_reshape_2d(ctx, cur, n_embd_head_v*n_head, n_tokens);
    } else {
        ggml_tensor * kq = ggml_mul_mat(ctx, k, q);
        set_tensor_name(kq, "kq", il);
        // For phi2 the KQ multiplication must be done with F32 precision, otherwise we get NaNs.
        // Ref: https://github.com/ggerganov/llama.cpp/pull/4490#issuecomment-1859055847
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
        kq = ggml_soft_max_ext(ctx, kq, kq_mask, kq_scale, hparams.f_max_alibi_bias);
        set_tensor_name(kq, "kq_soft_max_ext", il);
        GGML_ASSERT(kv.size == n_ctx);
        // Split cached v into n_head heads.
        ggml_tensor * v = ggml_view_3d(
            ctx, kv.v_l[il],
            n_kv, n_embd_head_v, n_head_kv,
            ggml_element_size(kv.v_l[il])*n_ctx,
            ggml_element_size(kv.v_l[il])*n_ctx*n_embd_head_v,
            0
        );
        set_tensor_name(v, "v", il);
        ggml_tensor * kqv = ggml_mul_mat(ctx, v, kq);
        set_tensor_name(kqv, "kqv", il);
        ggml_tensor * kqv_merged = ggml_permute(ctx, kqv, 0, 2, 1, 3);
        set_tensor_name(kqv_merged, "kqv_merged", il);
        // Make contiguous, with new shape.
        cur = ggml_cont_2d(ctx, kqv_merged, n_embd_head_v*n_head, n_tokens);
        set_tensor_name(cur, "kqv_merged_cont", il);
    }

    ggml_build_forward_expand(graph, cur);
    cur = ggml_mul_mat(ctx, wo, cur);
    if (wo_b) {
        // Only set the name of the output projection if there is also a bias.
        // The bias name will be set outside the function.
        set_tensor_name(cur, "kqv_wo", il);
        cur = ggml_add(ctx, cur, wo_b);
    }
    return cur;
}

static ggml_tensor * lm_build_kv(
    ggml_context * ctx,
    moondream_lm & model,
    moondream_lm_hparams & hparams,
    moondream_lm_cparams & cparams,
    moondream_kv_cache & kv,
    ggml_cgraph * graph,
    ggml_tensor * wo,
    ggml_tensor * wo_b,
    ggml_tensor * k_cur,
    ggml_tensor * v_cur,
    ggml_tensor * q_cur,
    ggml_tensor * kq_mask,
    // TODO: some of these can probably be replaced with the structs that contain them
    int32_t n_tokens,
    int32_t kv_head,
    int32_t n_kv,
    float kq_scale,
    int il
) {
    // These nodes are added to the graph together so that they are not reordered.
    // By doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(graph, q_cur);
    ggml_build_forward_expand(graph, k_cur);
    ggml_build_forward_expand(graph, v_cur);

    lm_build_kv_cache(ctx, hparams, cparams, kv, graph, k_cur, v_cur, n_tokens, kv_head, il);
    ggml_tensor * cur;
    cur = lm_build_kqv(
        ctx, model, hparams, cparams, kv, graph, wo, wo_b,
        q_cur, kq_mask, n_tokens, n_kv, kq_scale, il
    );
    set_tensor_name(cur, "kqv_out", il);
    return cur;
}

static ggml_tensor * lm_build_inp_out_ids(
    ggml_context * ctx,
    moondream_lm_context & mctx,
    int n_outputs
) {
    mctx.inp_out_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_outputs);
    set_tensor_name(mctx.inp_out_ids, "inp_out_ids", -1);
    ggml_set_input(mctx.inp_out_ids);
    return mctx.inp_out_ids;
}

// Modification of llama.cpp build_phi2.
// Ref: https://github.com/ggerganov/llama.cpp/blob/da799b41891e34aac86ce4e173f9c4c0afd4fab3/llama.cpp
static ggml_cgraph * lm_build_phi2(
    moondream_lm & model,
    moondream_lm_batch & batch,
    moondream_lm_context & mctx
) {
    moondream_lm_hparams & hparams = model.hparams;
    moondream_lm_cparams & cparams = mctx.cparams;
    moondream_kv_cache & kv_cache = mctx.kv_cache;

    ggml_init_params build_ctx_params = {
        .mem_size = mctx.compute_buffer.size(),
        .mem_buffer = mctx.compute_buffer.data(),
        .no_alloc = true
    };
    ggml_context * ctx0 = ggml_init(build_ctx_params);

    // TODO: Fix all the inconsistent integer types

    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, LLAMA_MAX_NODES, false);

    const int rope_type = MOONDREAM_ROPE_TYPE;
    int n_rot = hparams.n_rot;
    const int n_head = hparams.n_head;
    const int n_head_kv = hparams.n_head_kv;
    const int n_ctx = cparams.n_ctx;

    // NOTE: llama.cpp has some additional initialization logic for n_outputs
    const int n_outputs = mctx.n_outputs;

    // NOTE: llama.cpp has some additional initialization logic for n_kv which may be relevant
    // REF:
    // n_kv (worst_case ? kv_self.size : kv_self.n)
    kv_cache.n = kv_cache.size;          // Only consider worst case for now.
    const int32_t n_kv = kv_cache.n;     // size of KV cache to consider (n_kv <= kv_self.size)
    // NOTE: llama.cpp has some additional initialization logic for kv_head which may be relevant
    // REF:
    // kv_head (worst_case ? (kv_self.recurrent ? 0 : kv_self.size - n_tokens) : kv_self.head)
    const int32_t kv_head = kv_cache.head;
    const int32_t n_tokens = batch.n_tokens;
    const int64_t n_layer = hparams.n_layer;
    const int64_t n_embd = hparams.n_embd;
    const int64_t n_embd_head = hparams.n_embd_head_v;
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

    const uint32_t n_ctx_orig = cparams.n_ctx_orig_yarn;
    const float freq_base = cparams.rope_freq_base;
    const float freq_scale = cparams.rope_freq_scale;
    const float ext_factor = cparams.yarn_ext_factor;
    const float attn_factor = cparams.yarn_attn_factor;
    const float beta_fast = cparams.yarn_beta_fast;
    const float beta_slow = cparams.yarn_beta_slow;

    ggml_tensor * cur;
    ggml_tensor * attn_norm_output;
    ggml_tensor * ffn_output;
    ggml_tensor * inpL;

    inpL = lm_build_inp_embd(ctx0, mctx, hparams, batch, model.tok_embd);
    ggml_tensor * inp_pos = lm_build_inp_pos(ctx0, mctx, batch);
    // Mask for 1 head. It will be broadcasted to all heads.
    ggml_tensor * kq_mask = lm_build_inp_kq_mask(ctx0, mctx, batch, cparams, n_kv);

    for (int il = 0; il < n_layer; ++il) {
        attn_norm_output = lm_build_norm(
            ctx0, inpL, hparams,
            model.layers[il].attn_norm,
            model.layers[il].attn_norm_b,
            il
        );
        set_tensor_name(attn_norm_output, "attn_norm", il);

        // Self-attention
        {
            struct ggml_tensor * q_cur = nullptr;
            struct ggml_tensor * k_cur = nullptr;
            struct ggml_tensor * v_cur = nullptr;

            cur = ggml_mul_mat(ctx0, model.layers[il].wqkv, attn_norm_output);
            set_tensor_name(cur, "wqkv", il);
            cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
            set_tensor_name(cur, "bqkv", il);

            q_cur = ggml_cont(
                ctx0, ggml_view_2d(ctx0, cur, n_embd, n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd))
            );
            k_cur = ggml_cont(
                ctx0, ggml_view_2d(ctx0, cur, n_embd, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd))
            );
            v_cur = ggml_cont(
                ctx0, ggml_view_2d(ctx0, cur, n_embd, n_tokens, cur->nb[1], 2*sizeof(float)*(n_embd))
            );

            set_tensor_name(q_cur, "q_cur", il);
            set_tensor_name(k_cur, "k_cur", il);
            set_tensor_name(v_cur, "v_cur", il);

            q_cur = ggml_reshape_3d(ctx0, q_cur, n_embd_head, n_head, n_tokens);
            k_cur = ggml_reshape_3d(ctx0, k_cur, n_embd_head, n_head_kv, n_tokens);

            q_cur = ggml_rope_ext(
                ctx0, q_cur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
            );
            set_tensor_name(q_cur, "q_cur", il);

            // With phi2, we scale the Q to avoid precision issues.
            // Ref: https://github.com/ml-explore/mlx-examples/blob/08e862336ade809bc37d1035f94b359e7d1a5152/phi2/phi2.py#L64-L66
            q_cur = ggml_scale(ctx0, q_cur, 1.0f/sqrtf(float(n_embd_head)));
            set_tensor_name(q_cur, "q_cur", il);

            k_cur = ggml_rope_ext(
                ctx0, k_cur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
            );
            set_tensor_name(k_cur, "k_cur", il);

            cur = lm_build_kv(
                ctx0, model, hparams, cparams, kv_cache, gf,
                model.layers[il].wo, model.layers[il].bo,
                k_cur, v_cur, q_cur, kq_mask, n_tokens, kv_head, n_kv, 1.0f, il
            );
        }

        if (il == n_layer - 1) {
            // Skip computing output for unused tokens.
            ggml_tensor * inp_out_ids = lm_build_inp_out_ids(ctx0, mctx, n_outputs);
            cur = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            attn_norm_output = ggml_get_rows(ctx0, attn_norm_output, inp_out_ids);
        }

        // Feed forward
        {
            ffn_output = ggml_mul_mat(ctx0, model.layers[il].ffn_up, attn_norm_output);
            ffn_output = ggml_add(ctx0, ffn_output, model.layers[il].ffn_up_b);
            ffn_output = ggml_gelu(ctx0, ffn_output);
            ffn_output = ggml_mul_mat(ctx0, model.layers[il].ffn_down, ffn_output);
            ffn_output = ggml_add(ctx0, ffn_output, model.layers[il].ffn_down_b);

            set_tensor_name(ffn_output, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, ffn_output);
        cur = ggml_add(ctx0, cur, inpL);
        set_tensor_name(cur, "l_out", il);

        // Set input for next layer.
        inpL = cur;
    }

    cur = lm_build_norm(
        ctx0, inpL, hparams,
        model.output_norm,
        model.output_norm_b,
        -1
    );
    set_tensor_name(cur, "result_norm", -1);

    cur = ggml_mul_mat(ctx0, model.output, cur);
    set_tensor_name(cur, "result_output_no_bias", -1);

    cur = ggml_add(ctx0, cur, model.output_b);
    set_tensor_name(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

bool moondream_lm_batch_init(
    moondream_lm_batch & batch,
    int32_t n_tokens_alloc,
    int32_t n_embd,
    bool alloc_embd
) {
    if (alloc_embd) {
        batch.embd = (float *)malloc(sizeof(float) * n_tokens_alloc * n_embd);
        if (!batch.embd) {
            printf("could not allocate memory for moondream_lm_batch token embeddings\n");
            return false;
        }
    } else {
        batch.token = (int32_t *)malloc(sizeof(int32_t) * n_tokens_alloc);
        if (!batch.token) {
            printf("could not allocate memory for moondream_lm_batch tokens\n");
            return false;
        }
    }
    batch.pos = (int32_t *)malloc(sizeof(int32_t) * n_tokens_alloc);
    if (!batch.pos) {
        printf("could not allocate memory for moondream_lm_batch token positions\n");
        return false;
    }
    for (int i = 0; i < n_tokens_alloc; ++i) {
        batch.pos[i] = -1;
    }

    batch.n_tokens = 0;
    batch.n_tokens_alloc = n_tokens_alloc;
    return true;
}

void moondream_lm_batch_free(moondream_lm_batch & batch) {
    if (batch.token) {
        free(batch.token);
        batch.token = nullptr;
    }
    if (batch.embd) {
        free(batch.embd);
        batch.embd = nullptr;
    }
    if (batch.pos) {
        free(batch.pos);
        batch.pos = nullptr;
    }
}

bool moondream_kv_cache_init(
    moondream_kv_cache & kv_cache,
    moondream_lm_hparams & hparams,
    moondream_lm_cparams & cparams,
    ggml_backend_t backend,
    ggml_type type_k,
    ggml_type type_v
) {
    const uint32_t n_embd = hparams.n_embd;
    const int64_t n_layer = hparams.n_layer;
    const uint32_t kv_size = cparams.n_ctx;

    kv_cache.k_l.reserve(n_layer);
    kv_cache.v_l.reserve(n_layer);

    ggml_init_params init_params = {
        .mem_size = 2u * n_layer * ggml_tensor_overhead(),
        .mem_buffer = NULL,
        .no_alloc = true
    };
    ggml_context * ctx = ggml_init(init_params);
    if (!ctx) {
        printf("failed to initialize ggml_context for kv cache\n");
        return false;
    }

    // Create k/v cache tensors for each attention layer but don't allocate memory for them.
    for (int i = 0; i < n_layer; ++i) {
        ggml_tensor * k = ggml_new_tensor_1d(ctx, type_k, n_embd * kv_size);
        ggml_tensor * v = ggml_new_tensor_1d(ctx, type_v, n_embd * kv_size);
        kv_cache.k_l.push_back(k);
        kv_cache.v_l.push_back(v);
    }

    // For the sake of simplicity, we're only using one buffer type right now,
    // but this will probablly have to change in the future.
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    // Allocate memory for the k/v cache tensors.
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buf) {
        printf("failed to allocate ggml_backend_buffer for kv cache\n");
        return false;
    }
    // Initialize buffer to avoid NaNs in the padding.
    ggml_backend_buffer_clear(buf, 0);

    kv_cache.head = 0;
    kv_cache.size = kv_size;
    kv_cache.used = 0;
    // Value tensor is only transposed when not using flash attention.
    kv_cache.v_trans = !cparams.flash_attn;
    kv_cache.type_k = type_k;
    kv_cache.type_v = type_v;
    kv_cache.ctx = ctx;
    kv_cache.buf = buf;
    return true;
}

bool moondream_lm_context_init(
    moondream_lm_context & mctx,
    moondream_lm_hparams & hparams,
    moondream_lm_cparams & cparams,
    moondream_lm & model,
    ggml_type type_k,
    ggml_type type_v,
    bool normal_logs_enabled
) {
    memcpy(&mctx.cparams, &cparams, sizeof(moondream_lm_cparams));

    // For the sake of simplicity, we're only using one buffer type right now,
    // but this will probablly have to change in the future.
    mctx.backend_cpu = ggml_backend_cpu_init();
    if (!mctx.backend_cpu) {
        printf("failed to initialize cpu backend\n");
        return false;
    }
    ggml_backend_cpu_set_n_threads(mctx.backend_cpu, cparams.n_threads);
    mctx.backend_cpu_buft = ggml_backend_get_default_buffer_type(mctx.backend_cpu);

    bool result = moondream_kv_cache_init(mctx.kv_cache, hparams, cparams, mctx.backend_cpu, type_k, type_v);
    if (!result) {
        printf("failed to initialize moondream_kv_cache\n");
        return false;
    }

    // Buffer used to store the computation graph and the tensor meta data.
    const size_t compute_buf_size =
        ggml_tensor_overhead() * LLAMA_MAX_NODES
        + ggml_graph_overhead_custom(LLAMA_MAX_NODES, false);
    if (normal_logs_enabled) {
        const double compute_buf_size_gib = bytes_to_gib(compute_buf_size);
        printf("new compute_buf_size is %zu B, %lf GiB\n", compute_buf_size, compute_buf_size_gib);
    }
    mctx.compute_buffer.resize(compute_buf_size);

    // Initialize scheduler with worst-case graph.
    mctx.sched = ggml_backend_sched_new(&mctx.backend_cpu, &mctx.backend_cpu_buft, 1, LLAMA_MAX_NODES, false);
    moondream_lm_batch dummy_batch;
    result = moondream_lm_batch_init(dummy_batch, cparams.n_ctx, 0, false);
    if (!result) {
        printf("failed to initialize batch\n");
        return false;
    }
    dummy_batch.n_tokens = cparams.n_ctx;
    for (int i = 0; i < dummy_batch.n_tokens; ++i) {
        dummy_batch.token[i] = i;
    }
    mctx.n_outputs = 1;
    ggml_cgraph * gf = lm_build_phi2(model, dummy_batch, mctx);
    if (!ggml_backend_sched_reserve(mctx.sched, gf)) {
        printf("failed to reserve buffers for compute graph\n");
        return false;
    }
    moondream_lm_batch_free(dummy_batch);

    // TODO: equivalent of llama_output_reserve()

    return true;
}

void moondream_lm_context_free(moondream_lm_context & mctx) {
    if (mctx.backend_cpu) {
        ggml_backend_free(mctx.backend_cpu);
        mctx.backend_cpu = nullptr;
    }
    // TODO: figure out why this causes a segfault
    /*if (mctx.sched) {
        ggml_backend_sched_free(mctx.sched);
        mctx.sched = nullptr;
    }*/
    if (mctx.kv_cache.buf) {
        ggml_backend_buffer_free(mctx.kv_cache.buf);
        mctx.kv_cache.buf = nullptr;
    }
    if (mctx.kv_cache.ctx) {
        ggml_free(mctx.kv_cache.ctx);
        mctx.kv_cache.ctx = nullptr;
    }
    if (mctx.ctx) {
        ggml_free(mctx.ctx);
        mctx.ctx = nullptr;
    }
}

struct lm_symbol {
    using index = int;
    index prev;
    index next;
    const char * text;
    size_t n;
};

struct lm_bigram_bpe {
    struct comparator {
        bool operator()(const lm_bigram_bpe & l, const lm_bigram_bpe & r) const {
            return l.rank > r.rank || (l.rank == r.rank && l.left > r.left);
        }
    };
    using queue_storage = std::vector<lm_bigram_bpe>;
    using queue = std::priority_queue<lm_bigram_bpe, queue_storage, comparator>;
    lm_symbol::index left;
    lm_symbol::index right;
    std::string text;
    int rank;
    size_t size;
};

static int vocab_find_bpe_rank(
    const moondream_vocab & vocab, const std::string & token_left, const std::string & token_right
) {
    assert(token_left.find(' ') == std::string::npos);
    assert(token_left.find('\n') == std::string::npos);
    assert(token_right.find(' ') == std::string::npos);
    assert(token_right.find('\n') == std::string::npos);
    auto r = vocab.bpe_ranks.find(
        std::make_pair(token_left, token_right)
    );
    if (r == vocab.bpe_ranks.end()) {
        return -1;
    }
    return r->second;
}

static void add_new_bigram(
    const moondream_vocab & vocab,
    const lm_symbol * symbols,
    int left,
    int right,
    lm_bigram_bpe::queue & work_queue
) {
    if (left == -1 || right == -1) { return; }

    std::string left_token = std::string(symbols[left].text, symbols[left].n);
    std::string right_token = std::string(symbols[right].text, symbols[right].n);

    int rank_found = -1;
    rank_found = vocab_find_bpe_rank(vocab, left_token, right_token);
    if (rank_found < 0) { return; }

    lm_bigram_bpe bigram;
    bigram.left = left;
    bigram.right = right;
    bigram.text = left_token + right_token;
    bigram.size = left_token.size() + right_token.size();
    bigram.rank = rank_found;
    work_queue.push(bigram);
}

// token_ids should point to a buffer with `sizeof(int32_t) * text_len` bytes because text_len
// is the maximum number of tokens.
int moondream_lm_tokenize(
    moondream_vocab & vocab,
    const char * text,
    int text_len,
    int32_t * token_ids_output
) {
    if (!text || text[0] == '\0') {
        printf("could not tokenize text because text is NULL or empty");
        return -1;
    }
    if (!token_ids_output) {
        printf("could not tokenize text because pointer to output buffer for token ids is NULL\n");
        return -1;
    }

    // Moondream 2 uses BPE tokenizer.
    printf(
        "%s%s\n",
        "WARNING: defaulting to gpt2 pre-tokenizer because the pre-tokenizer was not specified, ",
        "this may degrade output quality"
    );

    std::vector<std::string> word_collection = unicode_regex_split(
        text,
        {"'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"}
    );

    const size_t symbol_buf_size = sizeof(lm_symbol) * text_len;
    lm_symbol * symbols = (lm_symbol *)malloc(symbol_buf_size);
    if (!symbols) {
        printf("failed to allocate memory for symbols buffer during tokenization\n");
        return -1;
    }
    lm_symbol * symbols_final = (lm_symbol *)malloc(symbol_buf_size);
    if (!symbols_final) {
        printf("failed to allocate memory for symbols_final buffer during tokenization\n");
        free(symbols);
        return -1;
    }
    int n_symbols = 0;
    int n_symbols_final = 0;

    for (auto & word : word_collection) {
        lm_bigram_bpe::queue work_queue = lm_bigram_bpe::queue();
        const int symbols_cur_word_start_idx = n_symbols;
        size_t char_offset = 0;

        while (char_offset < word.size()) {
            lm_symbol cur_symbol;
            cur_symbol.text = word.c_str() + char_offset;
            cur_symbol.n = std::min(word.size() - char_offset, (size_t) ::utf8_len(word[char_offset]));
            char_offset += cur_symbol.n;
            cur_symbol.prev = n_symbols - 1;
            cur_symbol.next = char_offset == word.size() ? -1 : n_symbols + 1;
            symbols[n_symbols] = cur_symbol;
            ++n_symbols;
        }

#ifdef MOONDREAM_EXTRA_LOGS
        for (int j = symbols_cur_word_start_idx; j < n_symbols; ++j) {
            lm_symbol cur_symbol = symbols[j];
            std::string sym_text = std::string(cur_symbol.text, cur_symbol.n);
            printf("(DEBUG) symbols[%d]: %s\n", j, sym_text.c_str());
        }
#endif // MOONDREAM_EXTRA_LOGS

        for (int k = symbols_cur_word_start_idx + 1; k < n_symbols; ++k) {
            add_new_bigram(vocab, symbols, k - 1, k, work_queue);
        }

        // Build token(s).
        while (!work_queue.empty()) {
            lm_bigram_bpe bigram = work_queue.top();
            work_queue.pop();
            lm_symbol & left_symbol = symbols[bigram.left];
            lm_symbol & right_symbol = symbols[bigram.right];
            if (left_symbol.n == 0 || right_symbol.n == 0) { continue; }

            std::string left_token = std::string(left_symbol.text, left_symbol.n);
            std::string right_token = std::string(right_symbol.text, right_symbol.n);
            // Skip bigram if it's outdated.
            if (left_token + right_token != bigram.text) { continue; }

            // Merge right symbol into left symbol.
            left_symbol.n += right_symbol.n;
            right_symbol.n = 0;
            left_symbol.next = right_symbol.next;
            // Remove the right symbol from the chain.
            left_symbol.next = right_symbol.next;
            if (right_symbol.next >= 0) {
                symbols[right_symbol.next].prev = bigram.left;
            }

            // Left side of current symbol.
            add_new_bigram(vocab, symbols, left_symbol.prev, bigram.left, work_queue);
            // Right side of current symbol.
            add_new_bigram(vocab, symbols, bigram.left, left_symbol.next, work_queue);
        }

        // Add the finished tokens to the final list keeping correct order for next and prev.
        int cur_symbol_idx = symbols_cur_word_start_idx;
        while (cur_symbol_idx >= 0) {
            lm_symbol cur_symbol = symbols[cur_symbol_idx];
            // Prepare cur_symbol_idx for the next iteration of the loop.
            cur_symbol_idx = cur_symbol.next;
            // Skip zero length symbols.
            if (cur_symbol.n <= 0) { continue; }
            const int prev_idx = n_symbols_final - 1;
            cur_symbol.prev = prev_idx;
            cur_symbol.next = -1;
            symbols_final[n_symbols_final] = cur_symbol;

            if (prev_idx >= 0) {
                // Update the next index of the previous symbol.
                symbols_final[prev_idx].next = n_symbols_final;
            }
            ++n_symbols_final;
        }
    }

#ifdef MOONDREAM_EXTRA_LOGS
    for (int k = 0; k < n_symbols_final; ++k) {
        lm_symbol f = symbols_final[k];
        std::string sym_final = std::string(f.text, f.n);
        printf("(DEBUG) symbols_final[%d] %s\n", k, sym_final.c_str());
    }
#endif // MOONDREAM_EXTRA_LOGS

    int n_token_ids = 0;
    //token_ids_output[token_ids_output_offset++] = vocab.bos_token_id;
    if (n_symbols_final >= 0) {
        int cur_symbol_idx = 0;
        while (cur_symbol_idx >= 0) {
            lm_symbol & cur_symbol = symbols_final[cur_symbol_idx];
            // Prepare cur_symbol_idx for the next iteration of the loop.
            cur_symbol_idx = cur_symbol.next;
            // Skip zero length symbols.
            if (cur_symbol.n == 0) { continue; }

            const std::string token_str = std::string(cur_symbol.text, cur_symbol.n);
            const auto token = vocab.token_to_id.find(token_str);
            if (token == vocab.token_to_id.end()) {
                for (auto k = token_str.begin(); k != token_str.end(); ++k) {
                    std::string byte_str(1, *k);
                    auto token_multibyte = vocab.token_to_id.find(byte_str);
                    if (token_multibyte == vocab.token_to_id.end()) {
                        printf("byte note found in vocab\n");
                        free(symbols);
                        free(symbols_final);
                        return -1;
                    } else if (n_token_ids >= text_len) {
                        printf("exceeded maximum number of tokens in token id output buffer\n");
                        free(symbols);
                        free(symbols_final);
                        return -1;
                    } else {
                        token_ids_output[n_token_ids++] = (*token_multibyte).second;
                    }
                }
            } else if (n_token_ids >= text_len) {
                printf("exceed maximum number of tokens in token id output buffer\n");
                free(symbols);
                free(symbols_final);
                return -1;
            } else {
                token_ids_output[n_token_ids++] = (*token).second;
            }
        }
    }

    free(symbols);
    free(symbols_final);
    return n_token_ids;
}

bool moondream_lm_load_from_file(const char * gguf_file_path, moondream_lm & model, bool normal_logs_enabled) {
    ggml_context * ctx;
    gguf_init_params init_params = {.no_alloc = false, .ctx = &ctx};
    gguf_context * meta = gguf_init_from_file(gguf_file_path, init_params);
    if (meta == NULL) {
        return false;
    }
    int gguf_version = gguf_get_version(meta);
    size_t gguf_alignment = gguf_get_alignment(meta);
    size_t gguf_data_offset = gguf_get_data_offset(meta);
    const char * model_arch = gguf_get_val_str(meta, gguf_find_key(meta, "general.architecture"));
    const char * model_name = gguf_get_val_str(meta, gguf_find_key(meta, "general.name"));

    /* Start of hparams load. */
    moondream_lm_hparams hparams;
    hparams.n_ctx_train = (int)gguf_get_val_u32(meta, gguf_find_key(meta, ARCH_PREFIX("context_length")));
    hparams.n_embd = (int)gguf_get_val_u32(meta, gguf_find_key(meta, ARCH_PREFIX("embedding_length")));
    hparams.n_rot = (int)gguf_get_val_u32(meta, gguf_find_key(meta, ARCH_PREFIX("rope.dimension_count")));
    hparams.n_layer = (int)gguf_get_val_u32(meta, gguf_find_key(meta, ARCH_PREFIX("block_count")));
    hparams.n_ff = (int)gguf_get_val_u32(meta, gguf_find_key(meta, ARCH_PREFIX("feed_forward_length")));
    hparams.n_head = (int)gguf_get_val_u32(meta, gguf_find_key(meta, ARCH_PREFIX("attention.head_count")));
    hparams.n_head_kv = (int)gguf_get_val_u32(meta, gguf_find_key(meta, ARCH_PREFIX("attention.head_count_kv")));
    hparams.f_norm_eps = gguf_get_val_f32(
        meta, gguf_find_key(meta, ARCH_PREFIX("attention.layer_norm_epsilon"))
    );
    hparams.f_norm_rms_eps = 0.0f; // Not present in file.
    // Calculate n_head_k and n_head_v because they are not specified.
    hparams.n_embd_head_k = hparams.n_embd / hparams.n_head;
    hparams.n_embd_head_v = hparams.n_embd_head_k;
    hparams.n_vocab = 51200;
    hparams.f_max_alibi_bias = 0.0f;
    model.hparams = hparams;
    /* End of hparams load. */

    /* Start of vocab load. */
    moondream_vocab vocab;
    // NOTE: the pre-tokenizer is missing, this might degrade generation quality.
    const char * tokenizer_model_name = gguf_get_val_str(meta, gguf_find_key(meta, TOK_PREFIX("model")));
    vocab.bos_token_id = (int32_t)gguf_get_val_u32(meta, gguf_find_key(meta, TOK_PREFIX("bos_token_id")));
    vocab.eos_token_id = (int32_t)gguf_get_val_u32(meta, gguf_find_key(meta, TOK_PREFIX("eos_token_id")));
    vocab.unknown_token_id = (int32_t)gguf_get_val_u32(
        meta, gguf_find_key(meta, TOK_PREFIX("unknown_token_id"))
    );
    vocab.separator_token_id = -1;
    vocab.padding_token_id = -1;
    const int tokens_key_id = gguf_find_key(meta, TOK_PREFIX("tokens"));
    vocab.n_tokens = gguf_get_arr_n(meta, tokens_key_id);
    if (vocab.n_tokens != model.hparams.n_vocab) {
        model.hparams.n_vocab = vocab.n_tokens;
        printf("expected gguf vocab size to be %d but got %d\n", hparams.n_vocab, vocab.n_tokens);
        return false;
    }
    for (int i = 0; i < vocab.n_tokens; ++i) {
        std::string token_str = gguf_get_arr_str(meta, tokens_key_id, i);
        vocab.id_to_token.emplace_back(token_str);
        vocab.token_to_id[token_str] = i;
    }
    vocab.scores = nullptr; // Scores are not present.
    vocab.token_type = (const int32_t *)gguf_get_arr_data(meta, gguf_find_key(meta, TOK_PREFIX("token_type")));
    const int merges_key_id = gguf_find_key(meta, TOK_PREFIX("merges"));
    vocab.n_merges = gguf_get_arr_n(meta, merges_key_id);
    for (int i = 0; i < vocab.n_merges; ++i) {
        std::string word = gguf_get_arr_str(meta, merges_key_id, i);
        std::string first;
        std::string second;
        const size_t pos = word.find(' ', 1);
        if (pos != std::string::npos) {
            first = word.substr(0, pos);
            second = word.substr(pos + 1);
        }
        vocab.bpe_ranks.emplace(std::make_pair(first, second), i);
    }
    model.vocab = vocab;
    /* End of vocab load. */

    /* Start of tensors load. */
    ggml_tensor * cur = ggml_get_first_tensor(ctx);
    if (cur == NULL) {
        return false;
    }
    // For some reason the first tensor doesn't have a name, but the second one is the token embedding,
    // so we just skip over the first one and start with the token embedding.
    // Note that this may be incorrect.
    cur = ggml_get_next_tensor(ctx, cur);
    if (cur == NULL) {
        return false;
    }
#ifdef MOONDREAM_EXTRA_LOGS
    printf("(DEBUG) found %s\n", cur->name);
#endif
    model.tok_embd = cur; // token_embd.weight

    const int n_tensors_per_layer = 10;
    for (int i = 0; i < hparams.n_layer; ++i) {
        moondream_lm_layer cur_layer;
        for (int k = 0; k < n_tensors_per_layer; ++k) {
            cur = ggml_get_next_tensor(ctx, cur);
            if (cur == NULL) {
                return false;
            }
#ifdef MOONDREAM_EXTRA_LOGS
            printf("(DEBUG) found %s\n", cur->name);
#endif
            switch (k) {
                case 0: // attn_norm.weight
                    cur_layer.attn_norm = cur;
                    break;
                case 1: // attn_norm.bias
                    cur_layer.attn_norm_b = cur;
                    break;
                case 2: // attn_qkv.weight
                    cur_layer.wqkv = cur;
                    break;
                case 3: // attn_qkv.bias
                    cur_layer.bqkv = cur;
                    break;
                case 4: // attn_output.weight
                    cur_layer.wo = cur;
                    break;
                case 5: // attn_output.bias
                    cur_layer.bo = cur;
                    break;
                case 6: // ffn_up.weight
                    cur_layer.ffn_up = cur;
                    break;
                case 7: // ffn_up.bias
                    cur_layer.ffn_up_b = cur;
                    break;
                case 8: // ffn_down.weight
                    cur_layer.ffn_down = cur;
                    break;
                case 9: // ffn_down.bias
                    cur_layer.ffn_down_b = cur;
                    break;
                default:
                    return false;
            }
        }
        model.layers.push_back(cur_layer);
    }

    const int n_output_layer_tensors = 4;
    for (int i = 0; i < n_output_layer_tensors; ++i) {
        cur = ggml_get_next_tensor(ctx, cur);
        if (cur == NULL) {
            return false;
        }
#ifdef MOONDREAM_EXTRA_LOGS
        printf("(DEBUG) found %s\n", cur->name);
#endif
        switch (i) {
            case 0: // output_norm.weight
                model.output_norm = cur;
                break;
            case 1: // output_norm.bias
                model.output_norm_b = cur;
                break;
            case 2: // output.weight
                model.output = cur;
                break;
            case 3: // output.bias
                model.output_b = cur;
                break;
            default:
                return false;
        }
    }
    /* End of tensors load. */

    model.ctx = ctx;

    if (normal_logs_enabled) {
        printf("------------\nloaded %s from %s\n", model_name, gguf_file_path);
        printf("gguf_version: %d\n", gguf_version);
        printf("gguf_alignment: %zu\n", gguf_alignment);
        printf("gguf_data_offset: %zu\n", gguf_data_offset);
        printf("model_arch: %s\n", model_arch);
        printf("mem_size: %lf GiB\n", bytes_to_gib(ggml_get_mem_size(model.ctx)));
        printf("------------\nHyperparameters\n------------\n");
        printf("n_ctx_train: %u\n", hparams.n_ctx_train);
        printf("n_embd: %d\n", hparams.n_embd);
        printf("n_layer: %d\n", hparams.n_layer);
        printf("n_ff: %d\n", hparams.n_ff);
        printf("n_head: %d\n", hparams.n_head);
        printf("n_head_kv: %d\n", hparams.n_head_kv);
        printf("n_embd_head_k: %d\n", hparams.n_embd_head_k);
        printf("n_embd_head_v: %d\n", hparams.n_embd_head_v);
        printf("n_vocab: %d\n", hparams.n_vocab);
        printf("------------\nVocab\n------------\n");
        printf("tokenizer_model_name: %s\n", tokenizer_model_name);
        printf("bos_token_id: %d\n", vocab.bos_token_id);
        printf("eos_token_id: %d\n", vocab.eos_token_id);
        printf("unknown_token_id: %d\n", vocab.separator_token_id);
        printf("separator_token_id: %d\n", vocab.separator_token_id);
        printf("padding_token_id: %d\n", vocab.padding_token_id);
        printf("n_tokens: %d\n", vocab.n_tokens);
        printf("n_merges: %d\n", vocab.n_merges);
        printf("------------\n");
    }

    gguf_free(meta);
    return true;
}

static int sample_top_logit(const float * logits, const int n_logits) {
    if (n_logits <= 0) {
        printf("cannot sample when n_logits <= 0");
        return -1;
    }
    int id_max = 0;
    float val_max = logits[0];
    for (int i = 0; i < n_logits; ++i) {
        float val_cur = logits[i];
        if (val_cur > val_max) {
            val_max = val_cur;
            id_max = i;
        }
    }
    return id_max;
}

static bool lm_set_inputs(moondream_lm_context & mctx, moondream_lm_batch & batch) {
    // This function should be called after build_phi2() so the corresponding input tensors
    // in mctx should already be created.
    if (batch.token) {
        ggml_backend_tensor_set(
            mctx.inp_tokens, batch.token, 0, batch.n_tokens * ggml_element_size(mctx.inp_tokens)
        );
    } else {
        printf("only batch.token inputs are supported but batch.token is NULL\n");
        return false;
    }

    if (batch.pos) {
        ggml_backend_tensor_set(
            mctx.inp_pos, batch.pos, 0, batch.n_tokens * ggml_element_size(mctx.inp_pos)
        );
    } else {
        printf("could not set mctx.inp_pos because batch.pos is NULL\n");
        return false;
    }

    int32_t * inp_out_ids_data = (int32_t *)mctx.inp_out_ids->data;
    if (mctx.n_outputs == 1) {
        // Only keep last output.
        inp_out_ids_data[0] = batch.n_tokens - 1;
    } else {
        printf("only mctx.n_outputs = 1 is supported but got mctx.n_outputs = %d\n", mctx.n_outputs);
        return false;
    }

    uint32_t n_kv = mctx.kv_cache.n;
    float * inp_kq_mask_data = (float *)mctx.inp_kq_mask->data;
    for (int i = 0; i < batch.n_tokens; ++i) {
        int32_t cur_pos = batch.pos[i];
        for (int k = 0; k < n_kv; ++k) {
            float f = (k > cur_pos) ? -INFINITY : 0.0f;
            inp_kq_mask_data[(i * n_kv) + k] = f;
        }
    }

    return true;
}

static std::string lm_decode_token_str(const std::string & text) {
    std::string decoded_text;
    const auto cpts = unicode_cpts_from_utf8(text);
    for (const auto cpt : cpts) {
        const auto utf8 = unicode_cpt_to_utf8(cpt);
        try {
            decoded_text += unicode_utf8_to_byte(utf8);
        } catch (const std::out_of_range & /*e*/) {
            decoded_text += "[UNK]";
        }
    }
    return decoded_text;
}

static bool lm_prefill_embeddings(
    moondream_lm_context & mctx,
    moondream_lm & model,
    moondream_lm_batch & batch,
    int embd_dim
) {
    // TODO: set n_output to zero. We don't want any logits when pre-filling the embeddings.

    ggml_cgraph * gf = lm_build_phi2(model, batch, mctx);

    if (!ggml_backend_sched_alloc_graph(mctx.sched, gf)) {
        printf("failed to allocate graph for ggml_backend_sched_t\n");
        return false;
    }

    ggml_backend_tensor_set(
        mctx.inp_embd, batch.embd, 0,
        batch.n_tokens * embd_dim * ggml_element_size(mctx.inp_embd)
    );
    ggml_backend_tensor_set(
        mctx.inp_pos, batch.pos, 0,
        batch.n_tokens * ggml_element_size(mctx.inp_pos)
    );

    const enum ggml_status compute_status = ggml_backend_sched_graph_compute(mctx.sched, gf);
    if (compute_status != GGML_STATUS_SUCCESS) {
        printf("graph computation failed (%s)\n", ggml_status_to_string(compute_status));
        return false;
    }
    ggml_backend_sched_synchronize(mctx.sched);
    ggml_backend_sched_reset(mctx.sched);
    return true;
}

static int lm_decode_inner(
    moondream_lm_context & mctx,
    moondream_lm & model,
    moondream_lm_batch & batch
) {
    ggml_cgraph * gf = lm_build_phi2(model, batch, mctx);

    if (!ggml_backend_sched_alloc_graph(mctx.sched, gf)) {
        printf("failed to allocate graph for ggml_backend_sched_t\n");
        return -1;
    }

    if (!lm_set_inputs(mctx, batch)) {
        printf("failed to set model inputs\n");
        return -1;
    }

    const enum ggml_status compute_status = ggml_backend_sched_graph_compute(mctx.sched, gf);
    if (compute_status != GGML_STATUS_SUCCESS) {
        printf("graph computation failed (%s)\n", ggml_status_to_string(compute_status));
        return -1;
    }
    ggml_backend_sched_synchronize(mctx.sched);

    // Increment the head of the KV cache so previous entries are not overwritten.
    mctx.kv_cache.head += batch.n_tokens;

    // Extract logits and sample.
    ggml_tensor * logits = gf->nodes[gf->n_nodes - 1];
    const int64_t logits_n_elements = ggml_nelements(logits);
    const float * logits_data = (float *)logits->data;
    const int sampled_token_id = sample_top_logit(logits_data, logits_n_elements);

    ggml_backend_sched_reset(mctx.sched);
    return sampled_token_id;
}

bool moondream_lm_decode(
    moondream_lm_context & mctx,
    moondream_lm & model,
    moondream_lm_batch & batch,
    std::string & response,
    int32_t n_prompt_tokens,
    int32_t * prompt_token_ids,
    int n_max_gen,
    bool log_response_stream,
    float * mmproj_embd,
    int n_embd,
    int embd_dim
) {
    assert(n_prompt_tokens <= batch.n_tokens_alloc);

    if (log_response_stream) {
        fprintf(stdout, "Response:");
        fflush(stdout);
    }

    mctx.n_ctx_active = 0;
    int sampled_token_id = -1;
    std::string local_response = "";

    if (mmproj_embd) {
        // TODO: cleanup this interface. moondream_batch is too entangled with everything.

        // Temporarily replace token ID batch with embd batch while re-using position buffer.
        assert(batch.n_tokens_alloc >= n_embd);
        int32_t * temp_token = batch.token;
        batch.token = nullptr;
        batch.embd = mmproj_embd;
        batch.n_tokens = n_embd;
        for (int i = 0; i < batch.n_tokens; ++i) {
            batch.pos[i] = mctx.n_ctx_active;
            ++mctx.n_ctx_active;
            /*
            for (int k = 0; k < batch.n_tokens; ++k) {
                printf("%f ", batch.embd[i * 2048 + k]);
            }
            printf("\n");
            */
        }
        lm_prefill_embeddings(mctx, model, batch, embd_dim);
        batch.token = temp_token;
        batch.embd = nullptr;
    }

    // Evaluate prompt and generate first response token.
    batch.n_tokens = n_prompt_tokens;
    for (int i = 0; i < batch.n_tokens; ++i) {
        batch.token[i] = prompt_token_ids[i];
        batch.pos[i] = mctx.n_ctx_active;
        ++mctx.n_ctx_active;
    }
    sampled_token_id = lm_decode_inner(mctx, model, batch);
    if (sampled_token_id < 0) {
        // Negative token IDs are invalid, so something went wrong.
        return false;
    }
    std::string cur_token_str = lm_decode_token_str(model.vocab.id_to_token[sampled_token_id]);
    local_response += cur_token_str;
    if (log_response_stream) {
        fprintf(stdout, "%s", cur_token_str.c_str());
        fflush(stdout);
    }
    if (sampled_token_id == model.vocab.eos_token_id) {
        // Return early (but without error) if the first response token was the eos token.
        response = local_response;
        return true;
    }

    // Clear batch.
    for (int i = 0; i < batch.n_tokens; ++i) {
        batch.token[i] = -1;
        batch.pos[i] = -1;
    }
    // We only evaluate one token per step from this point onwards.
    // Previous tokens are accessed through the KV cache.
    batch.n_tokens = 1;

    // Generate the rest of the response tokens.
    for (int i = 0; i < n_max_gen; ++i) {
        batch.token[0] = sampled_token_id;
        batch.pos[0] = mctx.n_ctx_active;
        ++mctx.n_ctx_active;

        sampled_token_id = lm_decode_inner(mctx, model, batch);
        if (sampled_token_id < 0) {
            // Negative token IDs are invalid, so something went wrong.
            return false;
        }
        cur_token_str = lm_decode_token_str(model.vocab.id_to_token[sampled_token_id]);
        local_response += cur_token_str;
        if (log_response_stream) {
            fprintf(stdout, "%s", cur_token_str.c_str());
            fflush(stdout);
        }
        if (sampled_token_id == model.vocab.eos_token_id) {
            break;
        }
    }
    printf("\n");
    printf("final mctx.n_ctx_active %d\n", mctx.n_ctx_active);
    response = local_response;
    return true;
}
