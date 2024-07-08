#include <cstdio>
#include <cstring>
#include <cassert>
#include <cmath>
#include <fstream>
#include <random>
#include <string>
#include <queue>
#include "moondream.h"
#include "unicode.h"

#define MD_TEXT_MODEL_FNAME "moondream2-text-model-f16.gguf"
#define MD_MMPROJ_FNAME "moondream2-mmproj-f16.gguf"
#define DATA_PATH_MAX_LEN 512
#define ARCH_PREFIX(t) ("phi2." t)
#define TOK_PREFIX(t) ("tokenizer.ggml." t)
#define LLAMA_MAX_NODES 8192
// Corresponds to LLAMA_ROPE_TYPE_NEOX from llama.cpp which is what is used for phi2.
#define MOONDREAM_ROPE_TYPE 2
// Define MOONDREAM_EXTRA_LOGS if you want additional logs for debugging.
//#define MOONDREAM_EXTRA_LOGS 

/* Start of helpers. */
static size_t utf8_len(char src) {
    const size_t lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}

static double bytes_to_gib(size_t n_bytes) {
    return static_cast<double>(n_bytes) / (1024.0 * 1024.0 * 1024.0);
}
/* End of helpers. */

// NOTE: skipping the usage of llm_build_cb (build callback) because I have a feeling
// it won't be necessary, may need to revisit this though
ggml_tensor * llm_build_inp_embd(
    ggml_context * ctx,
    moondream_context & mctx,
    const moondream_hparams & hparams,
    const moondream_batch & batch,
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

// NOTE: version of build_inp_pos without build callback
ggml_tensor * build_inp_pos(ggml_context * ctx, moondream_context & mctx, moondream_batch & batch) {
    mctx.inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.n_tokens);
    ggml_set_input(mctx.inp_pos);
    return mctx.inp_pos;
}

// NOTE: version of build_inp_KQ_mask without build callback
ggml_tensor * build_inp_KQ_mask(
    ggml_context * ctx, 
    moondream_context & mctx, 
    moondream_batch & batch,
    moondream_cparams & cparams,
    int32_t n_kv
) {
    // How does the causal branch differ from the non-causal branch?
    if (cparams.causal_attn) {
        mctx.inp_KQ_mask = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, n_kv, GGML_PAD(batch.n_tokens, GGML_KQ_MASK_PAD)
        );
    } else {
        mctx.inp_KQ_mask = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, batch.n_tokens, GGML_PAD(batch.n_tokens, GGML_KQ_MASK_PAD)
        );
    }
    ggml_set_input(mctx.inp_KQ_mask);
    return cparams.flash_attn ? ggml_cast(ctx, mctx.inp_KQ_mask, GGML_TYPE_F16) : mctx.inp_KQ_mask;
};

// Note build callback seems important for layer names so it might be needed here
ggml_tensor * llm_build_norm(
    ggml_context * ctx, 
    ggml_tensor * cur, 
    moondream_hparams & hparams,
    ggml_tensor * mw,
    ggml_tensor * mb,
    llm_norm_type type,
    int il
) {
    switch(type) {
        case LLM_NORM:
            cur = ggml_norm(ctx, cur, hparams.f_norm_eps);
            break;
        case LLM_NORM_RMS:
            cur = ggml_rms_norm(ctx, cur, hparams.f_norm_rms_eps);
            break;
    }
    
    // Weight
    if (mw) {
        cur = ggml_mul(ctx, cur, mw);
    }
    // Bias
    if (mb) {
        cur = ggml_add(ctx, cur, mb);
    }
    return cur;
}

// Maybe this should be renamed to llm_build_kv_cache?
void llm_build_kv_store(
    ggml_context * ctx, 
    moondream_hparams & hparams, 
    moondream_cparams & cparams, 
    moondream_kv_cache & kv,
    ggml_cgraph * graph,
    ggml_tensor * k_cur,
    ggml_tensor * v_cur,
    int32_t n_tokens,
    int32_t kv_head,
    int il
) {
    const int64_t n_ctx = cparams.n_ctx;
    const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa;
    const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa;
    
    // Why use GGML_ASSERT here and the regular c assert below?
    GGML_ASSERT(kv.size == n_ctx);

    // NOTE: I think this creates a view into the key cache, copies the key for the current head
    // into it, then builds it into the graph, idk why the build is necessary here though.
    ggml_tensor * k_cache_view = ggml_view_1d(
        ctx, kv.k_l[il], n_tokens*n_embd_k_gqa, 
        // Why are there parentheses around ggml_row_size?
        (ggml_row_size(kv.k_l[il]->type, n_embd_k_gqa))*kv_head
    );
    ggml_build_forward_expand(graph, ggml_cpy(ctx, k_cur, k_cache_view));

    assert(v_cur->ne[0] == n_embd_v_gqa && v_cur->ne[1] == n_tokens);

    ggml_tensor * v_cache_view = nullptr;
    if (cparams.flash_attn) {
        v_cache_view = ggml_view_1d(
            ctx, kv.v_l[il], n_tokens*n_embd_v_gqa, 
            // Why are there parantheses around kv_head?
            (kv_head)*ggml_row_size(kv.v_l[il]->type, n_embd_v_gqa)
        );
    } else {
        // TODO: figure out exactly what view 2d is doing under the hood
        // The v cache is transposed when not using flash attention.
        v_cache_view = ggml_view_2d(
            ctx, kv.v_l[il], n_tokens, n_embd_v_gqa, 
            (n_ctx)*ggml_element_size(kv.v_l[il]),
            (kv_head)*ggml_element_size(kv.v_l[il])
        );
        v_cur = ggml_transpose(ctx, v_cur);
    }
    ggml_build_forward_expand(graph, ggml_cpy(ctx, v_cur, v_cache_view));
}

ggml_tensor * llm_build_kqv(
    ggml_context * ctx,
    moondream_model & model,
    moondream_hparams & hparams,
    moondream_cparams & cparams,
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
    const int64_t n_embd_head_k = hparams.n_embd_head_k;
    const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa;
    const int64_t n_embd_head_v = hparams.n_embd_head_v;
    const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa;
    
    ggml_tensor * q = ggml_permute(ctx, q_cur, 0, 2, 1, 3);
    // TODO: figure out exactly how ggml_view_3d works under the hood
    ggml_tensor * k = ggml_view_3d(
        ctx, kv.k_l[il],
        n_embd_head_v, n_kv, n_head_kv,
        ggml_row_size(kv.k_l[il]->type, n_embd_k_gqa),
        ggml_row_size(kv.k_l[il]->type, n_embd_head_k),
        0
    );

    ggml_tensor * cur;
    if (cparams.flash_attn) {
        // llama uses GGML_UNUSED here but I'm not sure what it does
        // see llama.cpp line 6989 for more details

        // Split cached v into n_head heads (not transposed).
        ggml_tensor * v = ggml_view_3d(
            ctx, kv.v_l[il], 
            n_embd_head_v, n_kv, n_head_kv,
            ggml_row_size(kv.v_l[il]->type, n_embd_v_gqa),
            ggml_row_size(kv.v_l[il]->type, n_embd_head_v),
            0
        );
        cur = ggml_flash_attn_ext(ctx, q, k, v, kq_mask, kq_scale, hparams.f_max_alibi_bias);
        // For phi2 the KQ multiplication must be done with F32 precision, otherwise we get NaNs.
        // Ref: https://github.com/ggerganov/llama.cpp/pull/4490#issuecomment-1859055847
        ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);
        cur = ggml_reshape_2d(ctx, cur, n_embd_head_v*n_head, n_tokens);
    } else {
        ggml_tensor * kq = ggml_mul_mat(ctx, k, q);
        // For phi2 the KQ multiplication must be done with F32 precision, otherwise we get NaNs.
        // Ref: https://github.com/ggerganov/llama.cpp/pull/4490#issuecomment-1859055847
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
        kq = ggml_soft_max_ext(ctx, kq, kq_mask, kq_scale, hparams.f_max_alibi_bias);
        GGML_ASSERT(kv.size == n_ctx);
        // Split cached v into n_head heads.
        ggml_tensor * v = ggml_view_3d(
            ctx, kv.v_l[il], 
            n_kv, n_embd_head_v, n_head_kv,
            ggml_element_size(kv.v_l[il])*n_ctx,
            ggml_element_size(kv.v_l[il])*n_ctx*n_embd_head_v,
            0
        );
        // TODO: go over caching and clarify what's happening
        ggml_tensor * kqv = ggml_mul_mat(ctx, v, kq);
        ggml_tensor * kqv_merged = ggml_permute(ctx, kqv, 0, 2, 1, 3);
        // Make contiguous, with new shape.
        cur = ggml_cont_2d(ctx, kqv_merged, n_embd_head_v*n_head, n_tokens);
    }
    
    ggml_build_forward_expand(graph, cur);
    cur = ggml_mul_mat(ctx, wo, cur);
    if (wo_b) {
        cur = ggml_add(ctx, cur, wo_b);
    }
    return cur;
}

ggml_tensor * llm_build_kv(
    ggml_context * ctx, 
    moondream_model & model, 
    moondream_hparams & hparams,
    moondream_cparams & cparams,
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

    llm_build_kv_store(ctx, hparams, cparams, kv, graph, k_cur, v_cur, n_tokens, kv_head, il);
    ggml_tensor * cur;
    cur = llm_build_kqv(
        ctx, model, hparams, cparams, kv, graph, wo, wo_b, 
        q_cur, kq_mask, n_tokens, n_kv, kq_scale, il
    );
    return cur;
}

ggml_tensor * build_inp_out_ids(ggml_context * ctx, moondream_context & mctx, int n_outputs) {
    mctx.inp_out_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_outputs);
    //cb(lctx.inp_out_ids, "inp_out_ids", -1);
    ggml_set_input(mctx.inp_out_ids);
    return mctx.inp_out_ids;
}

ggml_tensor * llm_build_ffn(
    ggml_context * ctx,
    ggml_tensor * cur,
    ggml_tensor * up,
    ggml_tensor * up_b,
    ggml_tensor * gate,
    ggml_tensor * gate_b,
    ggml_tensor * down,
    ggml_tensor * down_b,
    ggml_tensor * act_scales,
    // NOTE: these flags might not be necessary if they don't vary for phi2 models.
    llm_ffn_op_type type_op,
    llm_ffn_gate_type type_gate,
    int il
) {
    ggml_tensor * tmp = up ? ggml_mul_mat(ctx, up, cur) : cur;
    if (up_b) {
        tmp = ggml_add(ctx, tmp, up_b);
    }
    if (gate) {
        switch (type_gate) {
            case LLM_FFN_SEQ: {
                cur = ggml_mul_mat(ctx, gate, tmp);
                break;
            }
            case LLM_FFN_PAR: {
                cur = ggml_mul_mat(ctx, gate, cur);
                break;
            }
        }
        if (gate_b) {
            cur = ggml_add(ctx, cur, gate_b);
        }
    } else {
        cur = tmp;
    }

    switch (type_op) {
        case LLM_FFN_SILU: {
            cur = ggml_silu(ctx, cur);
            break;
        }
        case LLM_FFN_GELU: {
            cur = ggml_gelu(ctx, cur);
            if (act_scales != NULL) {
                cur = ggml_div(ctx, cur, act_scales);
            }
            break;
        }
        case LLM_FFN_RELU: {
            cur = ggml_relu(ctx, cur);
            break;
        }
        case LLM_FFN_RELU_SQR: {
            cur = ggml_relu(ctx, cur);
            cur = ggml_sqr(ctx, cur);
            break;
        }
    }

    if (type_gate == LLM_FFN_PAR) {
        cur = ggml_mul(ctx, cur, tmp);
    }

    cur = ggml_mul_mat(ctx, down, cur);
    if (down_b) {
        cur = ggml_add(ctx, cur, down_b);
    }
    return cur;
}

// Modification of llama.cpp build_phi2.
// Ref: https://github.com/ggerganov/llama.cpp/blob/da799b41891e34aac86ce4e173f9c4c0afd4fab3/llama.cpp
ggml_cgraph * build_phi2(
    moondream_model & model,
    moondream_batch & batch,
    moondream_context & mctx
) {
    moondream_hparams & hparams = model.hparams;
    moondream_cparams & cparams = mctx.cparams;
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
    const int64_t n_embd_gqa = hparams.n_embd_v_gqa;
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

    // TODO: implement llm_build_inp_embd (see llama.cpp line 6654) - done but needs check
    //inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
    // NOTE: using a version of llm_build_inp_embd that doesn't use build cb
    inpL = llm_build_inp_embd(ctx0, mctx, hparams, batch, model.tok_embd);
    
    // TODO: implement build_inp_pos (see llama.cpp line 7346)
    // inp_pos - contains the positions
    // NOTE: using a version of llm_build_inp_embd that doesn't use build cb - done but needs check
    ggml_tensor * inp_pos = build_inp_pos(ctx0, mctx, batch);

    // TODO: implement build_inp_KQ_mask (see llama.cpp line 7371) - done but needs check
    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    ggml_tensor * KQ_mask = build_inp_KQ_mask(ctx0, mctx, batch, cparams, n_kv);

    for (int il = 0; il < n_layer; ++il) {
        // TODO: implement llm_build_norm (see llama.cpp line 6728) - done but needs check
        attn_norm_output = llm_build_norm(
            ctx0, inpL, hparams,
            model.layers[il].attn_norm,
            model.layers[il].attn_norm_b,
            // TODO: since LLM_NORM is hardcoded the arg might not be needed
            LLM_NORM, il
        );

        //cb(attn_norm_output, "attn_norm", il);

        // Self-attention
        {
            struct ggml_tensor * Qcur = nullptr;
            struct ggml_tensor * Kcur = nullptr;
            struct ggml_tensor * Vcur = nullptr;

            if (model.layers[il].wqkv) {
                cur = ggml_mul_mat(ctx0, model.layers[il].wqkv, attn_norm_output);
                //cb(cur, "wqkv", il);

                cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                //cb(cur, "bqkv", il);

                Qcur = ggml_cont(
                    ctx0, ggml_view_2d(ctx0, cur, n_embd, n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd))
                );
                Kcur = ggml_cont(
                    ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd))
                );
                Vcur = ggml_cont(
                    ctx0, 
                    ggml_view_2d(
                        ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)
                    )
                );
            } else {
                Qcur = ggml_add(
                    ctx0, ggml_mul_mat(ctx0, model.layers[il].wq, attn_norm_output), model.layers[il].bq
                );
                Kcur = ggml_add(
                    ctx0, ggml_mul_mat(ctx0, model.layers[il].wk, attn_norm_output), model.layers[il].bk
                );
                Vcur = ggml_add(
                    ctx0, ggml_mul_mat(ctx0, model.layers[il].wv, attn_norm_output), model.layers[il].bv
                );
            }

            //cb(Qcur, "Qcur", il);
            //cb(Kcur, "Kcur", il);
            //cb(Vcur, "Vcur", il);
            
            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);

            Qcur = ggml_rope_ext(
                ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx, n_ctx_orig,
                freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
            );
            //cb(Qcur, "Qcur", il);

            // With phi2, we scale the Q to avoid precision issues.
            // Ref: https://github.com/ml-explore/mlx-examples/blob/08e862336ade809bc37d1035f94b359e7d1a5152/phi2/phi2.py#L64-L66
            Qcur = ggml_scale(ctx0, Qcur, 1.0f/sqrtf(float(n_embd_head)));
            //cb(Qcur, "Qcur", il);

            Kcur = ggml_rope_ext(
                ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx, n_ctx_orig,
                freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
            );
            //cb(Kcur, "Kcur", il);

            // TODO: implement llm_build_kv (see llama.cpp line 7070) - done but needs check
            cur = llm_build_kv(
                ctx0, model, hparams, cparams, kv_cache, gf,
                model.layers[il].wo, model.layers[il].bo,
                Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f, il
            );
        }

        if (il == n_layer - 1) {
            // TODO: implement build_inp_out_ids (see llama.cpp line 7464) - done but needs check
            // Skip computing output for unused tokens.
            ggml_tensor * inp_out_ids = build_inp_out_ids(ctx0, mctx, n_outputs);
            cur = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            attn_norm_output = ggml_get_rows(ctx0, attn_norm_output, inp_out_ids);
        }

        // Feed forward
        {
            // TODO: implement llm_build_ffn (see llama.cpp line 6760) - done but needs check
            ffn_output = llm_build_ffn(
                ctx0, attn_norm_output,
                model.layers[il].ffn_up, model.layers[il].ffn_up_b,
                NULL, NULL, /* I guess this means that phi2 doesn't have a ff gate */
                model.layers[il].ffn_down, model.layers[il].ffn_down_b,
                NULL, LLM_FFN_GELU, LLM_FFN_SEQ, il
            );
            //cb(ffn_output, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, ffn_output);
        //cb(cur, "l_out", il);

        cur = ggml_add(ctx0, cur, inpL);
        //cb(cur, "l_out", il);

        inpL = cur;
    }

    // TODO: implement llm_build_norm (see llama.cpp line 6728) - done but needs check
    cur = llm_build_norm(
        ctx0, inpL, hparams,
        model.output_norm,
        model.output_norm_b,
        LLM_NORM, -1
    );
    //cb(cur, "result_norm", -1);

    cur = ggml_mul_mat(ctx0, model.output, cur);
    //cb(cur, "result_output_no_bias", -1);

    cur = ggml_add(ctx0, cur, model.output_b);
    //cb(cur, "result_output", -1);
    
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

// Modification of llama.cpp/examples/llava/clip.pp clip_image_build_graph.
// Ref: https://github.com/ggerganov/llama.cpp/blob/da799b41891e34aac86ce4e173f9c4c0afd4fab3/examples/llava/clip.cpp
ggml_cgraph * build_clip(
    moondream_mmproj_model & model,
    moondream_mmproj_batch & batch,
    moondream_mmproj_context & mctx
) {
    moondream_mmproj_hparams & hparams = model.hparams;
    // TODO: find some way to do narrowing conversions safely. 
    // Maybe convert all unsigned ints to signed ints when hyperparameters are loaded
    // and do a max val check there. We can't keep them as unsigned ints because most of
    // the ggml API calls take signed ints and we don't want to scatter narrowing conversions
    // everywhere.
    const int image_size = hparams.image_size;
    const int patch_size = hparams.patch_size;
    const int num_patches_per_side = image_size / patch_size;
    const int num_patches = num_patches_per_side * num_patches_per_side;
    const int num_positions = num_patches; /* + (ctx->has_class_embedding ? 1 : 0); */
    const int n_embd = hparams.n_embd;
    const int n_head = hparams.n_head;
    const int n_head_qkv = n_embd / n_head;
    const int n_layer = hparams.n_layer;
    const float eps = hparams.f_norm_eps; 
    
    // TODO: move this into moondream_mmproj_batch struct.
    const int batch_size = 1;
    assert(batch_size == 1);

    ggml_init_params build_ctx_params = {
        mctx.compute_buffer.size(),
        mctx.compute_buffer.data(),
        true
    };
    ggml_context * ctx0 = ggml_init(build_ctx_params);
    ggml_cgraph * gf = ggml_new_graph(ctx0);

    ggml_tensor * inp_raw = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, image_size, image_size, 3, batch_size);
    ggml_set_name(inp_raw, "inp_raw");
    ggml_set_input(inp_raw);

    ggml_tensor * inp = ggml_conv_2d(ctx0, model.patch_embd, inp_raw, patch_size, patch_size, 0, 0, 1, 1);
    inp = ggml_reshape_3d(ctx0, inp, num_patches, n_embd, batch_size);
    inp = ggml_cont(ctx0, ggml_permute(ctx0, inp, 1, 0, 2, 3));
    if (model.patch_bias != nullptr) {
        inp = ggml_add(ctx0, inp, model.patch_bias);
    }

    ggml_tensor * embeddings = inp;
    // NOTE: skipped class embeddings.
    ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, num_positions);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);
    embeddings = ggml_add(ctx0, embeddings, ggml_get_rows(ctx0, model.pos_embd, positions));
    // NOTE: skipped pre-layernorm.
    
    for (int il = 0; il < n_layer - 1; ++il) {
        // embeddings = residual, cur = hidden_states
        ggml_tensor * cur = embeddings;

        // Layernorm 1
        cur = ggml_norm(ctx0, cur, eps);
        cur = ggml_add(ctx0, ggml_mul(ctx0, cur, model.layers[il].ln_1_w), model.layers[il].ln_1_b);

        // Self-attention
        ggml_tensor * Q = ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].q_w, cur), model.layers[il].q_b);
        Q = ggml_scale_inplace(ctx0, Q, 1.0f / sqrt((float)n_head_qkv));
        Q = ggml_reshape_4d(ctx0, Q, n_head_qkv, n_head, num_positions, batch_size);
        Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
        Q = ggml_reshape_3d(ctx0, Q, n_head_qkv, num_positions, n_head * batch_size);

        ggml_tensor * K = ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].k_w, cur), model.layers[il].k_b);
        K = ggml_reshape_4d(ctx0, K, n_head_qkv, n_head, num_positions, batch_size);
        K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
        K = ggml_reshape_3d(ctx0, K, n_head_qkv, num_positions, n_head * batch_size);

        ggml_tensor * V = ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].v_w, cur), model.layers[il].v_b);
        V = ggml_reshape_4d(ctx0, V, n_head_qkv, n_head, num_positions, batch_size);
        V = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 2, 0, 3));
        V = ggml_reshape_3d(ctx0, V, num_positions, n_head_qkv, n_head * batch_size);

        ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
        KQ = ggml_soft_max_inplace(ctx0, KQ);
        ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ);
        KQV = ggml_reshape_4d(ctx0, KQV, n_head_qkv, num_positions, n_head, batch_size);
        KQV = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

        cur = ggml_cont_3d(ctx0, KQV, n_embd, num_positions, batch_size);
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].o_w, cur), model.layers[il].o_b);
        // Add the residual.
        cur = ggml_add(ctx0, cur, embeddings);
        // embeddings = residual, cur = hidden_states
        embeddings = cur;
    
        // Layernorm 2
        cur = ggml_norm(ctx0, cur, eps);
        cur = ggml_add(ctx0, ggml_mul(ctx0, cur, model.layers[il].ln_2_w), model.layers[il].ln_2_b);

        // Feed forward
        cur = ggml_mul_mat(ctx0, model.layers[il].ff_i_w, cur);
        cur = ggml_add(ctx0, cur, model.layers[il].ff_i_b);
        if (hparams.use_gelu) {
            cur = ggml_gelu_inplace(ctx0, cur);
        } else {
            cur = ggml_gelu_quick_inplace(ctx0, cur);
        }
        cur = ggml_mul_mat(ctx0, model.layers[il].ff_o_w, cur);
        cur = ggml_add(ctx0, cur, model.layers[il].ff_o_b);

        // Add the resiudal.
        cur = ggml_add(ctx0, embeddings, cur);
        embeddings = cur;
    }

    // Post-layernorm
    embeddings = ggml_norm(ctx0, embeddings, eps);
    ggml_set_name(embeddings, "post_ln");
    embeddings = ggml_add(ctx0, ggml_mul(ctx0, embeddings, model.post_ln_w), model.post_ln_b);

    // LLaVa projector
    embeddings = ggml_reshape_2d(ctx0, embeddings, embeddings->ne[0], embeddings->ne[1]);
    ggml_tensor * patches = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, num_patches);
    ggml_set_name(patches, "patches");
    ggml_set_input(patches);
    embeddings = ggml_get_rows(ctx0, embeddings, patches);
    if (hparams.proj_type == PROJECTOR_TYPE_MLP) {
        embeddings = ggml_mul_mat(ctx0, model.mm_0_w, embeddings);
        embeddings = ggml_add(ctx0, embeddings, model.mm_0_b);
        embeddings = ggml_gelu(ctx0, embeddings);
        embeddings = ggml_mul_mat(ctx0, model.mm_2_w, embeddings);
        embeddings = ggml_add(ctx0, embeddings, model.mm_2_b);
    } else {
        printf("unimplemented projector type, only the MLP projector type is implemented\n");
        ggml_free(ctx0);
        return nullptr;
    }

    ggml_build_forward_expand(gf, embeddings);
    ggml_free(ctx0);
    return gf;
}

bool moondream_init_batch(
    moondream_batch & batch, 
    int32_t n_tokens_alloc, 
    int32_t n_embd, 
    bool alloc_embd, 
    int32_t n_seq_max
) {
    batch.n_tokens = 0;
    if (alloc_embd) {
        batch.embd = (float *)malloc(sizeof(float) * n_tokens_alloc * n_embd);
        if (!batch.embd) {
            printf("could not allocate memory for moondream_batch token embeddings\n");
            return false;
        }
    } else {
        batch.token = (int32_t *)malloc(sizeof(int32_t) * n_tokens_alloc);
        if (!batch.token) {
            printf("could not allocate memory for moondream_batch tokens\n");
            return false;
        }
    }
    batch.pos = (int32_t *)malloc(sizeof(int32_t) * n_tokens_alloc);
    if (!batch.pos) {
        printf("could not allocate memory for moondream_batch token positions\n");
        return false;
    }
    /*batch.n_seq_id = (int32_t *)malloc(sizeof(int32_t) * n_tokens_alloc);
    if (!batch.n_seq_id) {
        printf("could not allocate memeory for moondream_batch n_seq_id\n");
        return false;
    }*/
    // TODO: this could probably be allocated as a single chunk with the for loop
    // setting pointers at (i * n_tokens_alloc * sizeof(int32_t)) strides.
    // TODO: re-enable seq_id allocation if necessary.
    /*batch.seq_id = (int32_t **)malloc(sizeof(int32_t *) * (n_tokens_alloc + 1));
    if (!batch.seq_id) {
        printf("could not allocated memory for moondream_batch seq_id\n");
        return false;
    }
    for (int32_t i = 0; i < n_tokens_alloc; ++i) {
        batch.seq_id[i] = (int32_t *)malloc(sizeof(int32_t) * n_seq_max);
        if (!batch.seq_id) {
            printf("could not allocate memory for moondream_batch seq_id[%d]\n", i);
            return false;
        }
    }
    batch.seq_id[n_tokens_alloc] = nullptr;*/
    batch.seq_id = nullptr;
    batch.logits = (int8_t *)malloc(sizeof(int8_t) * n_tokens_alloc);
    if (!batch.logits) {
        printf("coulld not allocate memory for moondream_batch logits\n");
        return false;
    }
    
    return true;
}

void moondream_free_batch(moondream_batch & batch) {
    if (batch.token) { free(batch.token); }
    if (batch.embd) { free(batch.embd); }
    if (batch.pos) { free(batch.pos); }
    // TODO: free each seq_id if seq_id allocation is re-enabled.
    //if (batch.seq_id) { free(batch.seq_id); }
    if (batch.logits) { free(batch.logits); }
}

bool moondream_init_kv_cache(
    moondream_kv_cache & kv_cache,
    moondream_hparams & hparams, 
    moondream_cparams & cparams, 
    ggml_backend_t backend,
    ggml_type type_k,
    ggml_type type_v
) {
    // TODO: double check this
    const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa;
    const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa;
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
        ggml_tensor * k = ggml_new_tensor_1d(ctx, type_k, n_embd_k_gqa * kv_size);
        ggml_tensor * v = ggml_new_tensor_1d(ctx, type_v, n_embd_v_gqa * kv_size);
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
    printf("succesfully allocated memory for moondream_kv_cache tensors\n");
    // Initialize buffer to avoid NaNs in the padding.
    ggml_backend_buffer_clear(buf, 0);

    kv_cache.head = 0;
    kv_cache.size = kv_size;
    kv_cache.used = 0;
    // Value tensor is only transposed when not using flash attention.
    kv_cache.v_trans = !cparams.flash_attn; // TODO: double check this
    kv_cache.type_k = type_k;
    kv_cache.type_v = type_v;
    kv_cache.ctx = ctx;
    kv_cache.buf = buf;
    return true;
}

bool moondream_init_context(
    moondream_context & mctx,
    moondream_hparams & hparams,
    moondream_cparams & cparams,
    moondream_model & model,
    ggml_type type_k, 
    ggml_type type_v
) {
    memcpy(&mctx.cparams, &cparams, sizeof(moondream_cparams));
    
    // For the sake of simplicity, we're only using one buffer type right now,
    // but this will probablly have to change in the future.
    mctx.backend_cpu = ggml_backend_cpu_init();
    if (!mctx.backend_cpu) {
        printf("failed to initialize cpu backend\n");
        return false;
    }
    printf("succesfully initialized cpu backend\n");
    ggml_backend_cpu_set_n_threads(mctx.backend_cpu, cparams.n_threads);
    mctx.backend_cpu_buft = ggml_backend_get_default_buffer_type(mctx.backend_cpu);
    
    bool result = moondream_init_kv_cache(mctx.kv_cache, hparams, cparams, mctx.backend_cpu, type_k, type_v);
    if (!result) {
        printf("failed to initialize moondream_kv_cache\n");
        return false;
    }
    printf("succesfully initialized moondream_kv_cache\n");

    // Buffer used to store the computation graph and the tensor meta data.
    const size_t compute_buf_size = 
        ggml_tensor_overhead() * LLAMA_MAX_NODES
        + ggml_graph_overhead_custom(LLAMA_MAX_NODES, false);
#ifdef MOONDREAM_EXTRA_LOGS
    const double compute_buf_size_gib = bytes_to_gib(compute_buf_size);
    printf("new compute_buf_size is %zu B, %lf GiB\n", compute_buf_size, compute_buf_size_gib);
#endif // MOONDREAM_EXTRA_LOGS
    mctx.compute_buffer.resize(compute_buf_size);
    
    // Initialize scheduler with worst-case graph.
    mctx.sched = ggml_backend_sched_new(&mctx.backend_cpu, &mctx.backend_cpu_buft, 1, LLAMA_MAX_NODES, false);
    int32_t dummy_token = 0;
    moondream_batch dummy_batch;
    dummy_batch.n_tokens = cparams.n_ctx; // narrowing conversion
    dummy_batch.token = &dummy_token;
    dummy_batch.embd = nullptr;
    dummy_batch.pos = nullptr;
    dummy_batch.seq_id = nullptr;
    dummy_batch.logits = nullptr;
    mctx.n_outputs = 1; // TODO: figure out what n_outputs should be during initialization.
    ggml_cgraph * gf = build_phi2(model, dummy_batch, mctx);
    if (!ggml_backend_sched_reserve(mctx.sched, gf)) {
        printf("failed to reserve buffers for compute graph\n");
        return false;
    }
    printf("succesfully reserved buffers for compute graph\n");

    // TODO: equivalent of llama_output_reserve(), see llama.cpp line 11949

    return true;
}

// WIP implementation of mmproj context initializaton.
// Currently fails with:
// GGML_ASSERT: ggml/src/ggml-backend.c:1431: node_backend_id != -1
bool moondream_init_mmproj_context(
    moondream_mmproj_context & mctx, 
    moondream_mmproj_model & model, 
    int n_threads
) {
    mctx.backend_cpu = ggml_backend_cpu_init();
    if (!mctx.backend_cpu) {
        printf("failed to initialize mmproj cpu backend\n");
        return false;
    }
    printf("succesfully initialized mmproj cpu backend\n");
    ggml_backend_cpu_set_n_threads(mctx.backend_cpu, n_threads);
    mctx.backend_cpu_buft = ggml_backend_get_default_buffer_type(mctx.backend_cpu);
    //const size_t compute_buf_size = GGML_DEFAULT_GRAPH_SIZE * ggml_tensor_overhead() + ggml_graph_overhead();
    // TODO: figure out a way to dynamically determine the number of required nodes because LLAMA_MAX_NODES
    // is probably overkill for the clip graph.
    const size_t compute_buf_size = 
        ggml_tensor_overhead() * LLAMA_MAX_NODES 
        + ggml_graph_overhead_custom(LLAMA_MAX_NODES, false);
#ifdef MOONDREAM_EXTRA_LOGS
    const double compute_buf_size_gib = bytes_to_gib(compute_buf_size);
    printf("new mmproj compute_buf_size is %zu B, %lf GiB\n", compute_buf_size, compute_buf_size_gib);
#endif // MOONDREAM_EXTRA_LOGS
    mctx.compute_buffer.resize(compute_buf_size);
    
    // Initialize scheduler.
    moondream_mmproj_batch dummy_batch;
    ggml_cgraph * gf = build_clip(model, dummy_batch, mctx);
    if (!gf) {
        printf("failed to build mmrpoj compute graph\n");
        return false;
    }
    printf("n_nodes: %d\n", gf->n_nodes);
    printf("built mmproj graph\n");
    mctx.sched = ggml_backend_sched_new(&mctx.backend_cpu, &mctx.backend_cpu_buft, 1, gf->n_nodes, false);
    if (!ggml_backend_sched_reserve(mctx.sched, gf)) {
        printf("failed to reserve buffers for mmproj compute graph\n");
        return false;
    }
    printf("succesfully reserved buffers for mmproj compute graph\n");
    return true;
}

void moondream_free_context(moondream_context & mctx) {
    if (mctx.backend_cpu) { ggml_backend_free(mctx.backend_cpu); }
    if (mctx.sched) { ggml_backend_sched_free(mctx.sched); }
    if (mctx.kv_cache.buf) { ggml_backend_buffer_free(mctx.kv_cache.buf); }
    if (mctx.kv_cache.ctx) { ggml_free(mctx.kv_cache.ctx); }
    if (mctx.ctx) { ggml_free(mctx.ctx); }
}

struct llm_symbol {
    using index = int;
    index prev;
    index next;
    const char * text;
    size_t n;
};

struct llm_bigram_bpe {
    struct comparator {
        bool operator()(const llm_bigram_bpe & l, const llm_bigram_bpe & r) const {
            return l.rank > r.rank || (l.rank == r.rank && l.left > r.left);
        }
    };
    using queue_storage = std::vector<llm_bigram_bpe>;
    using queue = std::priority_queue<llm_bigram_bpe, queue_storage, comparator>;
    llm_symbol::index left;
    llm_symbol::index right;
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
    const llm_symbol * symbols,
    int left, 
    int right, 
    llm_bigram_bpe::queue & work_queue
) {
    if (left == -1 || right == -1) { return; }

    std::string left_token = std::string(symbols[left].text, symbols[left].n);
    std::string right_token = std::string(symbols[right].text, symbols[right].n);
    
    int rank_found = -1;
    rank_found = vocab_find_bpe_rank(vocab, left_token, right_token);
    //printf("left: %s, right: %s, rank found: %d\n", left_token.c_str(), right_token.c_str(), rank_found);
    if (rank_found < 0) { return; }

    llm_bigram_bpe bigram;
    bigram.left = left;
    bigram.right = right;
    bigram.text = left_token + right_token;
    bigram.size = left_token.size() + right_token.size();
    bigram.rank = rank_found;
    work_queue.push(bigram);
}

// token_ids should point to a buffer with `sizeof(int32_t) * text_len` bytes because text_len
// is the maximum number of tokens.
bool moondream_tokenize(
    moondream_vocab & vocab, 
    const char * text,
    size_t text_len,
    int32_t * token_ids_output,
    size_t * n_token_ids_output
) {
    if (!text || text[0] == '\0') {
        printf("could not tokenize text because text is NULL or empty");
        return false;
    }
    if (!token_ids_output) {
        printf("could not tokenize text because pointer to output buffer for token ids is NULL\n");
        return false;
    }
    if (!n_token_ids_output) {
        printf("could not tokenize text because pointer to n_token_ids_output is NULL\n");
        return false;
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

    const size_t symbol_buf_size = sizeof(llm_symbol) * text_len;
    llm_symbol * symbols = (llm_symbol *)malloc(symbol_buf_size);
    if (!symbols) {
        printf("failed to allocate memory for symbols buffer during tokenization\n");
        return false;
    }
    llm_symbol * symbols_final = (llm_symbol *)malloc(symbol_buf_size);
    if (!symbols_final) {
        printf("failed to allocate memory for symbols_final buffer during tokenization\n");
        free(symbols);
        return false;
    }
    int n_symbols = 0;
    int n_symbols_final = 0;

    for (auto & word : word_collection) {
        llm_bigram_bpe::queue work_queue = llm_bigram_bpe::queue();
        const int symbols_cur_word_start_idx = n_symbols;
        size_t char_offset = 0;
        
        while (char_offset < word.size()) {
            llm_symbol cur_symbol;
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
            llm_symbol cur_symbol = symbols[j];
            std::string sym_text = std::string(cur_symbol.text, cur_symbol.n);
            printf("(DEBUG) symbols[%d]: %s\n", j, sym_text.c_str());
        }
#endif // MOONDREAM_EXTRA_LOGS

        for (int k = symbols_cur_word_start_idx + 1; k < n_symbols; ++k) {
            add_new_bigram(vocab, symbols, k - 1, k, work_queue);
        }

        // Build token(s).
        while (!work_queue.empty()) {
            llm_bigram_bpe bigram = work_queue.top();
            work_queue.pop();
            llm_symbol & left_symbol = symbols[bigram.left];
            llm_symbol & right_symbol = symbols[bigram.right];
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
            llm_symbol cur_symbol = symbols[cur_symbol_idx];
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
        llm_symbol f = symbols_final[k];
        std::string sym_final = std::string(f.text, f.n);
        printf("(DEBUG) symbols_final[%d] %s\n", k, sym_final.c_str());
    }
#endif // MOONDREAM_EXTRA_LOGS

    size_t token_ids_output_offset = 0;
    //token_ids_output[token_ids_output_offset++] = vocab.bos_token_id; 
    if (n_symbols_final >= 0) {
        int cur_symbol_idx = 0;
        while (cur_symbol_idx >= 0) {
            llm_symbol & cur_symbol = symbols_final[cur_symbol_idx];
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
                        return false;
                    } else if (token_ids_output_offset >= text_len) {
                        printf("exceeded maximum number of tokens in token id output buffer\n");
                        free(symbols);
                        free(symbols_final);
                        return false;
                    } else {
                        token_ids_output[token_ids_output_offset++] = (*token_multibyte).second;
                    }
                }
            } else if (token_ids_output_offset >= text_len) {
                printf("exceed maximum number of tokens in token id output buffer\n");
                free(symbols);
                free(symbols_final);
                return false;
            } else {
                token_ids_output[token_ids_output_offset++] = (*token).second;
            }
        }
    }

    free(symbols);
    free(symbols_final);
    *n_token_ids_output = token_ids_output_offset;
    return true;
}

bool moondream_load_model(const char * gguf_file_path, moondream_model & model) {
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
    moondream_hparams hparams;
    hparams.n_ctx_train = gguf_get_val_u32(meta, gguf_find_key(meta, ARCH_PREFIX("context_length")));
    hparams.n_embd = gguf_get_val_u32(meta, gguf_find_key(meta, ARCH_PREFIX("embedding_length")));
    hparams.n_rot = gguf_get_val_u32(meta, gguf_find_key(meta, ARCH_PREFIX("rope.dimension_count")));
    hparams.n_layer = gguf_get_val_u32(meta, gguf_find_key(meta, ARCH_PREFIX("block_count")));
    hparams.n_ff = gguf_get_val_u32(meta, gguf_find_key(meta, ARCH_PREFIX("feed_forward_length")));
    hparams.n_head = gguf_get_val_u32(meta, gguf_find_key(meta, ARCH_PREFIX("attention.head_count")));
    hparams.n_head_kv = gguf_get_val_u32(meta, gguf_find_key(meta, ARCH_PREFIX("attention.head_count_kv")));
    hparams.f_norm_eps = gguf_get_val_f32(
        meta, gguf_find_key(meta, ARCH_PREFIX("attention.layer_norm_epsilon"))
    );
    hparams.f_norm_rms_eps = 0.0f; // Not present in file.
    // Calculate n_head_k and n_head_v because they are not specified.
    hparams.n_embd_head_k = hparams.n_embd / hparams.n_head;
    hparams.n_embd_head_v = hparams.n_embd_head_k;
    // TODO: verify that the GQA hparams are correct. Reference llama.cpp lines 1922 and 1926.
    hparams.n_embd_k_gqa = hparams.n_embd_head_k * hparams.n_head_kv;
    hparams.n_embd_v_gqa = hparams.n_embd_head_v * hparams.n_head_kv;
    // TODO: determine this dynamically from the GGUF file instead of hardcoding it
    hparams.n_vocab = 51200;
    hparams.f_max_alibi_bias = 0.0f; // TODO: figure out what this is supposed to be.
    model.hparams = hparams;
    /* End of hparams load. */
    
    /* Start of vocab load. */
    moondream_vocab vocab;
    // NOTE: the pre-tokenizer is missing, this might degrade generation quality.
    const char * tokenizer_model_name = gguf_get_val_str(meta, gguf_find_key(meta, TOK_PREFIX("model")));
    vocab.bos_token_id = (int64_t)gguf_get_val_u32(meta, gguf_find_key(meta, TOK_PREFIX("bos_token_id")));
    vocab.eos_token_id = (int64_t)gguf_get_val_u32(meta, gguf_find_key(meta, TOK_PREFIX("eos_token_id")));
    vocab.unknown_token_id = (int64_t)gguf_get_val_u32(
        meta, gguf_find_key(meta, TOK_PREFIX("unknown_token_id"))
    );
    vocab.separator_token_id = -1;
    vocab.padding_token_id = -1;
    const int tokens_key_id = gguf_find_key(meta, TOK_PREFIX("tokens"));
    vocab.n_tokens = gguf_get_arr_n(meta, tokens_key_id);
    if (vocab.n_tokens != hparams.n_vocab) {
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
        moondream_layer cur_layer;
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

    printf("------------\nloaded %s from %s\n", model_name, gguf_file_path);
    printf("gguf_version: %d\n", gguf_version);
    printf("gguf_alignment: %ld\n", gguf_alignment);
    printf("gguf_data_offset: %ld\n", gguf_data_offset);
    printf("model_arch: %s\n", model_arch);
    printf("mem_size: %lf GiB\n", bytes_to_gib(ggml_get_mem_size(model.ctx)));
    printf("------------\nHyperparameters\n------------\n");
    printf("n_ctx_train: %d\n", hparams.n_ctx_train);
    printf("n_embd: %d\n", hparams.n_embd);
    printf("n_layer: %d\n", hparams.n_layer);
    printf("n_ff: %d\n", hparams.n_ff);
    printf("n_head: %d\n", hparams.n_head);
    printf("n_head_kv: %d\n", hparams.n_head_kv);
    printf("n_embd_head_k: %d\n", hparams.n_embd_head_k);
    printf("n_embd_head_v: %d\n", hparams.n_embd_head_v);
    printf("n_embd_k_gqa: %d\n", hparams.n_embd_k_gqa);
    printf("n_embd_v_gqa: %d\n", hparams.n_embd_v_gqa);
    printf("n_vocab: %d\n", hparams.n_vocab);
    printf("------------\nVocab\n------------\n");
    printf("tokenizer_model_name: %s\n", tokenizer_model_name);
    printf("bos_token_id: %ld\n", vocab.bos_token_id);
    printf("eos_token_id: %ld\n", vocab.eos_token_id);
    printf("unkown_token_id: %ld\n", vocab.separator_token_id);
    printf("separator_token_id: %ld\n", vocab.separator_token_id);
    printf("padding_token_id: %ld\n", vocab.padding_token_id);
    printf("n_tokens: %d\n", vocab.n_tokens);
    printf("n_merges: %d\n", vocab.n_merges);
    printf("------------\n");
    
    gguf_free(meta);
    return true;
}

bool moondream_load_mmproj_model(const char * gguf_file_path, moondream_mmproj_model & model) {
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
    moondream_mmproj_hparams hparams;
    hparams.image_size = gguf_get_val_u32(meta, gguf_find_key(meta, "clip.vision.image_size"));
    hparams.patch_size = gguf_get_val_u32(meta, gguf_find_key(meta, "clip.vision.patch_size"));
    hparams.n_embd = gguf_get_val_u32(meta, gguf_find_key(meta, "clip.vision.embedding_length"));
    hparams.n_ff = gguf_get_val_u32(meta, gguf_find_key(meta, "clip.vision.feed_forward_length"));
    hparams.n_proj = gguf_get_val_u32(meta, gguf_find_key(meta, "clip.vision.projection_dim"));
    hparams.n_head = gguf_get_val_u32(meta, gguf_find_key(meta, "clip.vision.attention.head_count"));
    hparams.n_layer = gguf_get_val_u32(meta, gguf_find_key(meta, "clip.vision.block_count"));
    hparams.f_norm_eps = gguf_get_val_f32(meta, gguf_find_key(meta, "clip.vision.attention.layer_norm_epsilon"));
    hparams.use_gelu = gguf_get_val_bool(meta, gguf_find_key(meta, "clip.use_gelu"));
    
    const char * proj_type_str = gguf_get_val_str(meta, gguf_find_key(meta, "clip.projector_type"));
    if (strncmp(proj_type_str, "mlp", 3) == 0) {
        hparams.proj_type = PROJECTOR_TYPE_MLP;
    } else {
        hparams.proj_type = PROJECTOR_TYPE_UNKNOWN;
    }

    const int image_mean_key_id = gguf_find_key(meta, "clip.vision.image_mean");
    const int n_image_mean = gguf_get_arr_n(meta, image_mean_key_id);
    if (n_image_mean  != 3) {
        printf("expected n_image_mean = 3 but got n_image_mean = %d\n", n_image_mean);
        return false;
    }
    hparams.image_mean = (float *)gguf_get_arr_data(meta, image_mean_key_id);

    const int image_std_key_id = gguf_find_key(meta, "clip.vision.image_std");
    const int n_image_std = gguf_get_arr_n(meta, image_std_key_id);
    if (n_image_std != 3) {
        printf("expected n_image_std = 3 but got n_image_std = %d\n", n_image_std);
        return false;
    }
    hparams.image_std = (float *)gguf_get_arr_data(meta, image_std_key_id);
    model.hparams = hparams;
    /* End of hparams load. */

    /* Start of tensors load. */
    // For some reason the first tensor doesn't have a name, so we skip over it.
    ggml_tensor * cur = ggml_get_first_tensor(ctx);
    if (cur == NULL) {
        return false;
    }
    // Load tensors that don't repeat for each layer.
    const int n_non_repeating_tensors = 9;
    for (int i = 0; i < n_non_repeating_tensors; ++i) {
        cur = ggml_get_next_tensor(ctx, cur);
        if (cur == NULL) {
            return false;
        }
#ifdef MOONDREAM_EXTRA_LOGS 
        printf("(DEBUG) found %s\n", cur->name);
#endif
        switch (i) {
            case 0: // mm.0.weight
                model.mm_0_w = cur;
                break;
            case 1: // mm.0.bias
                model.mm_0_b = cur;
                break;
            case 2: // mm.2.weight
                model.mm_2_w = cur;
                break;
            case 3: // mm.2.bias
                model.mm_2_b = cur;
                break;
            case 4:  // v.position_embd.weight
                model.pos_embd = cur;
                break;
            case 5: // v.patch_embd.weight
                model.patch_embd = cur;
                break;
            case 6: // v.patch_embd.bias
                model.patch_bias = cur;
                break;
            case 7: // v.post_ln.weight
                model.post_ln_w = cur;
                break;
            case 8: // v.post_ln.bias
                model.post_ln_b = cur;
                break;
            default:
                return false;
        }
    }

    // Load tensors for each layer.
    const int n_tensors_per_layer = 16;
    for (int i = 0; i < hparams.n_layer; ++i) {
        moondream_mmproj_layer cur_layer;
        for (int k = 0; k < n_tensors_per_layer; ++k) {
            cur = ggml_get_next_tensor(ctx, cur);
            if (cur == NULL) {
                return false;
            }
#ifdef MOONDREAM_EXTRA_LOGS 
            printf("(DEBUG) found %s\n", cur->name);
#endif
            switch (k) {
                case 0: // attn_q.weight
                    cur_layer.q_w = cur;
                    break;
                case 1: // attn_q.bias
                    cur_layer.q_b = cur;
                    break;
                case 2: // attn_k.weight
                    cur_layer.k_w = cur;
                    break;
                case 3: // attn_k.bias
                    cur_layer.k_b = cur;
                    break;
                case 4: // attn_v.weight
                    cur_layer.v_w = cur;
                    break;
                case 5: // attn_v.bias
                    cur_layer.v_b = cur;
                    break;
                case 6: // attn_out.weight
                    cur_layer.o_w = cur;
                    break;
                case 7: // attn_out.bias
                    cur_layer.o_b = cur;
                    break;
                case 8: // ln1.weight
                    cur_layer.ln_1_w = cur;
                    break;
                case 9: // ln1.bias
                    cur_layer.ln_1_b = cur;
                    break;
                // Are ffn_down and ffn_up reversed? Usually the first ff layer projects to a higher dim.
                case 10: // ffn_down.weight
                    cur_layer.ff_i_w = cur;
                    break;
                case 11: // ffn_down.bias
                    cur_layer.ff_i_b = cur;
                    break;
                case 12: // ffn_up.weight
                    cur_layer.ff_o_w = cur;
                    break;
                case 13: // ffn_up.bias
                    cur_layer.ff_o_b = cur;
                    break;
                case 14: // ln2.weight
                    cur_layer.ln_2_w = cur;
                    break;
                case 15: // ln2.bias
                    cur_layer.ln_2_b = cur;
                    break;
                default:
                    return false;
            }
        }
        model.layers.push_back(cur_layer);
    }
    /* End of tensors load. */
    
    model.ctx = ctx;
    
    printf("------------\nloaded %s from %s\n", model_name, gguf_file_path);
    printf("gguf_version: %d\n", gguf_version);
    printf("gguf_alignment: %ld\n", gguf_alignment);
    printf("gguf_data_offset: %ld\n", gguf_data_offset);
    printf("model_arch: %s\n", model_arch);
    printf("mem_size: %lf GiB\n", bytes_to_gib(ggml_get_mem_size(model.ctx)));
    printf("------------\nMMPROJ Hyperparameters\n------------\n");
    printf("image_size: %u\n", hparams.image_size);
    printf("patch_size: %u\n", hparams.patch_size);
    printf("n_embd: %u\n", hparams.n_embd);
    printf("n_ff: %u\n", hparams.n_ff);
    printf("n_proj: %u\n", hparams.n_proj);
    printf("n_head: %u\n", hparams.n_head);
    printf("n_layer: %u\n", hparams.n_layer);
    printf("f_norm_eps: %f\n", hparams.f_norm_eps);
    printf("n_head: %u\n", hparams.n_head);
    printf("image_mean: %f %f %f\n", hparams.image_mean[0], hparams.image_mean[1], hparams.image_mean[2]);
    printf("image_std: %f %f %f\n", hparams.image_std[0], hparams.image_std[1], hparams.image_std[2]);
    printf("proj_type_str: %s\n", proj_type_str);
    printf("------------\n");

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

int main(int argc, char * argv[]) {
    if (argc < 2) {
        printf("incorrect number of arguments\n");
        return 1;
    }
    const char * data_path = argv[1];
    const size_t data_path_length = strlen(data_path);
    if (data_path_length > DATA_PATH_MAX_LEN) {
        printf("provided data path exceeded max length");
        return 1;
    }

    // Resolve text model file path.
    const char * text_model_fname = MD_TEXT_MODEL_FNAME;
    const size_t text_model_fname_length = strlen(text_model_fname);
    // Add 1 to give space for null-terminator in concatenated string.
    const size_t text_model_path_length = data_path_length + text_model_fname_length + 1;
    char text_model_path[text_model_path_length];
    snprintf(text_model_path, text_model_path_length, "%s%s", data_path, text_model_fname); 

    // Resolve mmproj file path.
    const char * mmproj_fname = MD_MMPROJ_FNAME;
    const size_t mmproj_fname_length = strlen(mmproj_fname);
    // Add 1 to give space for null-terminator in concatenated string.
    const size_t mmproj_path_length = data_path_length + text_model_fname_length + 1;
    char mmproj_path[text_model_path_length];
    snprintf(mmproj_path, mmproj_path_length, "%s%s", data_path, mmproj_fname); 

    printf("text model path: %s\n", text_model_path);
    printf("mmproj path: %s\n", mmproj_path);
    
    // Initialize GGML.
    ggml_time_init();
    // Optional NUMA initialization for better performance on supported systems.
    /*enum ggml_numa_strategy numa_strat = GGML_NUMA_STRATEGY_DISTRIBUTE;
    if (ggml_is_numa()) {
        printf("numa node detected, initializing ggml numa\n");
        ggml_numa_init(numa_strat);
    }*/

    /* Start of moondream_model load. */
    moondream_model model;
    bool result = moondream_load_model(text_model_path, model);
    if (!result) {
        printf("could not load text model\n");
        return 1;
    }
    printf("succesfully loaded text model\n");
    /* End of moondream_model load. */

    /* Start of moondream_mmproj_model load. */
    moondream_mmproj_model mmproj_model;
    result = moondream_load_mmproj_model(mmproj_path, mmproj_model);
    if (!result) {
        printf("could not load mmproj model\n");
        return 1;
    }
    printf("succesfully loaded mmproj model\n");
    /* End of moondream_mmproj_model load. */

    /* Start of moondream_mmproj_context init. */
    /*
    moondream_mmproj_context mmproj_ctx;
    result = moondream_init_mmproj_context(mmproj_ctx, mmproj_model, 8);
    if (!result) {
        printf("failed to initialze moondream_mmproj_context\n");
        return 1;
    }
    printf("succesfully initialized moondream_context\n");
    */
    /* End of moondream_mmproj_context init. */

    /* Start of moondream_context init. */
    moondream_cparams cparams = {
        .n_ctx = 512,
        .n_batch = 1,
        .n_ubatch = 1,
        .n_seq_max = 1,
        .n_threads = 8,
        .n_threads_batch = 1,
        // TODO: figure out what these shoud be
        .rope_freq_base = 10000.0f,
        .rope_freq_scale = 1.0f,
        .n_ctx_orig_yarn = model.hparams.n_ctx_train, 
        .yarn_ext_factor = 1.0f,
        .yarn_attn_factor = 1.0f,
        .yarn_beta_fast = 32.0f,
        .yarn_beta_slow = 1.0f,
        .defrag_thold = -1.0f,
        // -----------------
        .embeddings = true,
        .causal_attn = true,
        .offload_kqv = false,
        .flash_attn = false
    };
    const ggml_type type_k = GGML_TYPE_F32;
    const ggml_type type_v = GGML_TYPE_F32;
    moondream_context mctx;
    result = moondream_init_context(mctx, model.hparams, cparams, model, type_k, type_v);
    if (!result) {
        printf("failed to initialze moondream_context\n");
        return 1;
    }
    mctx.n_outputs = 1;
    printf("succesfully initialized moondream_context\n");
    /* End of moondream_context init. */

    /* Start of prompt tokenization. */
    const char * prompt = "<image>\n\nQuestion: Describe the image.\n\nAnswer:";
    size_t prompt_len = strlen(prompt);
    printf("prompt_len: %zu\n", prompt_len);
    int32_t token_ids[prompt_len];
    size_t n_token_ids = 0;
    result = moondream_tokenize(model.vocab, prompt, prompt_len, token_ids, &n_token_ids);
    if (!result) {
        printf("failed to tokenize prompt\n");
        return 1;
    }
    printf("n_token_ids: %zu\n", n_token_ids);
    printf("token_ids: ");
    for (int i = 0; i < n_token_ids; ++i) {
        printf("%d ", token_ids[i]);
    }
    printf("\n");
    /* End of prompt tokenization. */

    /* Start of moondream_batch initialization. */
    moondream_batch batch;
    result = moondream_init_batch(batch, cparams.n_ctx, model.hparams.n_embd, false, cparams.n_seq_max);
    if (!result) {
        printf("failed to initialized moondream_batch\n");
        return 1;
    }
    batch.n_tokens = n_token_ids;
    for (int i = 0; i < n_token_ids; ++i) {
        batch.token[i] = token_ids[i];
    }
    printf("succesfully initialized moondream_batch\n");
    /* End of moondream_batch initialization. */

    printf("------------\n");
    std::string response = "";
    int n_gen = 128;
    for (int i = 0; i < n_gen; ++i) {
        ggml_cgraph * gf = build_phi2(model, batch, mctx);
        
        result = ggml_backend_sched_alloc_graph(mctx.sched, gf);
        if (!result) {
            printf("failed to allocate graph for ggml_backend_sched_t\n");
            return 1;
        }
        
        ggml_backend_tensor_set(
            mctx.inp_tokens, batch.token, 0, batch.n_tokens * ggml_element_size(mctx.inp_tokens)
        );
        
        const enum ggml_status compute_status = ggml_backend_sched_graph_compute(mctx.sched, gf);
        if (compute_status != GGML_STATUS_SUCCESS) {
            printf("graph computation failed (%s)\n", ggml_status_to_string(compute_status));
            return 1;
        }
        ggml_backend_sched_synchronize(mctx.sched);

        // Extract logits and sample.
        ggml_tensor * logits = gf->nodes[gf->n_nodes - 1];
        const int logits_n_dims = ggml_n_dims(logits);
        const int64_t logits_n_elements = ggml_nelements(logits);
        printf("logits_n_dims: %d, logits_n_elements: %ld\n", logits_n_dims, logits_n_elements);
        const size_t logits_size = logits_n_elements * sizeof(float);
        float * host_logits = (float *)malloc(logits_size);
        ggml_backend_tensor_get(logits, host_logits, 0, logits_size);
        const int sampled_token_id = sample_top_logit(host_logits, logits_n_elements);
        std::string sampled_token_str = model.vocab.id_to_token[sampled_token_id];
        response += sampled_token_str;
        printf("sampled_token_id: %d, sampled_token_str: %s\n", sampled_token_id, sampled_token_str.c_str());
        printf("response: %s\n", response.c_str());
        batch.token[batch.n_tokens] = sampled_token_id;
        
        ++mctx.n_sample;
        ++batch.n_tokens;
        ++mctx.kv_cache.head;
        // Ensure kv cache head points to a valid index.
        if (mctx.kv_cache.head >= mctx.kv_cache.size) {
            mctx.kv_cache.size = 0;
        }

        free(host_logits);
        ggml_backend_sched_reset(mctx.sched);
        printf("generation step %d\n------------\n", i);
        //break;
    }
    
    moondream_free_batch(batch);
    printf("freed moondream_batch\n");
    moondream_free_context(mctx);
    printf("freed moondream_context\n");
    ggml_free(mmproj_model.ctx);
    printf("freed mmproj model ggml_context\n");
    ggml_free(model.ctx);
    printf("freed model ggml_context\n");
    return 0;
}
