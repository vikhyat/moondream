#include <cstdio>
#include <cstring>
#include <cassert>
#include <cmath>
#include <fstream>
#include <vector>
#include "ggml/ggml.h"
#include "ggml/ggml-backend.h"

#define MD_TEXT_MODEL_FNAME "moondream2-text-model-f16.gguf"
#define MD_MMPROJ_FNAME "moondream2-mmproj-f16.gguf"
#define DATA_PATH_MAX_LEN 512
#define ARCH_PREFIX(t) ("phi2." t)
#define LLAMA_MAX_NODES   8192
// Corresponds to LLAMA_ROPE_TYPE_NEOX from llama.cpp which is what is used for phi2.
#define MOONDREAM_ROPE_TYPE 2
// Define MOONDREAM_EXTRA_LOGS if you want additional logs for debugging.
//#define MOONDREAM_EXTRA_LOGS 

/* start of llm enums */
enum llm_ffn_op_type {
    LLM_FFN_SILU,
    LLM_FFN_GELU,
    LLM_FFN_RELU,
    LLM_FFN_RELU_SQR,
};

enum llm_ffn_gate_type {
    LLM_FFN_SEQ,
    LLM_FFN_PAR, // ffn_gate is parallel to ffn_up
};

enum llm_norm_type {
    LLM_NORM,
    LLM_NORM_RMS,
};
/* end of llm enums */

struct moondream_layer {
    // Normalization
    ggml_tensor * attn_norm;
    ggml_tensor * attn_norm_b;
    ggml_tensor * attn_norm_2;
    ggml_tensor * attn_norm_2_b;
    ggml_tensor * attn_q_norm;
    ggml_tensor * attn_q_norm_b;
    ggml_tensor * attn_k_norm;
    ggml_tensor * attn_k_norm_b;
    ggml_tensor * attn_out_norm;
    ggml_tensor * attn_out_norm_b;
    ggml_tensor * attn_q_a_norm;
    ggml_tensor * attn_kv_a_norm;
    // Attention
    ggml_tensor * wq;
    ggml_tensor * wk;
    ggml_tensor * wv;
    ggml_tensor * wo;
    ggml_tensor * wqkv;
    ggml_tensor * wq_a;
    ggml_tensor * wq_b;
    ggml_tensor * wkv_a_mqa;
    ggml_tensor * wkv_b;
    // Attention bias
    ggml_tensor * bq;
    ggml_tensor * bk;
    ggml_tensor * bv;
    ggml_tensor * bo;
    ggml_tensor * bqkv;
    // Normalization
    ggml_tensor * ffn_norm;
    ggml_tensor * ffn_norm_b;
    ggml_tensor * layer_out_norm;
    ggml_tensor * layer_out_norm_b;
    ggml_tensor * ffn_norm_exps;
    // Feed forward
    ggml_tensor * ffn_gate; // w1
    ggml_tensor * ffn_down; // w2
    ggml_tensor * ffn_up;   // w3
    // Feed forward bias
    ggml_tensor * ffn_gate_b = nullptr;
    ggml_tensor * ffn_down_b = nullptr; // b2
    ggml_tensor * ffn_up_b = nullptr; // b3
    ggml_tensor * ffn_act;
};

struct moondream_hparams {
    int n_embd;
    int n_ff;
    int n_layer;
    int n_rot;
    int n_ctx_train;
    int n_head;
    int n_head_kv;
    int n_embd_head_k;
    int n_embd_k_gqa;
    int n_embd_head_v;
    int n_embd_v_gqa;
    int n_vocab;
    
    float f_norm_eps;
    float f_norm_rms_eps;

    // this doesn't seem to be present in the model
    float rope_freq_base_train;
    int rope_attn_factor;

    // max bias for attention, not sure if it's used for anything else
    float f_max_alibi_bias;
};

struct moondream_cparams {
    // Context size used during inference.
    uint32_t n_ctx;
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    // Number of threads to use for generation.
    uint32_t n_threads;
    // Number of threads to use for batch processing.
    uint32_t n_threads_batch;

    float rope_freq_base;
    float rope_freq_scale;

    uint32_t n_ctx_orig_yarn;
    // These hyperparameters are not exposed in GGUF, because all
    // existing YaRN models use the same values for them.
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;
    float defrag_thold;

    bool embeddings;
    bool causal_attn;
    bool offload_kqv;
    bool flash_attn;

    //enum llama_pooling_type pooling_type;

    //ggml_backend_sched_eval_callback cb_eval;
    //void * cb_eval_user_data;
};

struct moondream_model {
    ggml_context * ctx;
    moondream_hparams hparams;
    std::vector<moondream_layer> layers;
    ggml_tensor * tok_embd;
    ggml_tensor * output_norm;
    ggml_tensor * output_norm_b;
    ggml_tensor * output;
    ggml_tensor * output_b;
};

// Arrays must have size of n_tokens
struct moondream_batch {
    int32_t n_tokens;
    // The token ids of the input (used when embd is NULL).
    int32_t * token;
    // The token embeddings (used when token is NULL).
    float * embd;
    // The positions of the respective tokens in the sequence.
    int32_t * pos;
    // The sequence to which the respective token belongs.
    int32_t ** seq_id;
    // If zero, the logits for the respective token will not be output.
    int8_t * logits;
};

struct moondream_kv_cache {
    bool has_shift = false;
    bool do_defrag = false;
    bool do_copy = false;
    // Whether or not the value tensor is transposed.
    bool v_trans = true;

    uint32_t head = 0;
    uint32_t size = 0;
    uint32_t used = 0;

    // Computed before each graph build.
    // What does it mean though?
    uint32_t n = 0;

    ggml_type type_k = GGML_TYPE_F16;
    ggml_type type_v = GGML_TYPE_F16;
    
    // k and v caches for each layer.
    std::vector<struct ggml_tensor *> k_l;
    std::vector<struct ggml_tensor *> v_l;

    ggml_context * ctx;
    ggml_backend_buffer_t buf;
};

struct moondream_context {
    ggml_context * ctx;
    moondream_cparams cparams;
    moondream_kv_cache kv_cache;
    ggml_backend_t backend_cpu;
    int n_outputs;
     // Number of tokens sampled.
    int32_t n_sample = 0;
    // Input tensors
    ggml_tensor * inp_tokens;    // I32 [n_batch]
    ggml_tensor * inp_embd;      // F32 [n_embd, n_batch]
    ggml_tensor * inp_pos;       // I32 [n_batch]
    ggml_tensor * inp_out_ids;   // I32 [n_outputs]
    ggml_tensor * inp_KQ_mask;   // F32 [kv_size, n_batch]
    ggml_tensor * inp_K_shift;   // I32 [kv_size]
    ggml_tensor * inp_mean;      // F32 [n_batch, n_batch]
    ggml_tensor * inp_cls;       // I32 [n_batch]
    ggml_tensor * inp_s_copy;    // I32 [kv_size]
    ggml_tensor * inp_s_mask;    // F32 [1, n_kv]
    ggml_tensor * inp_s_seq;     // I32 [n_kv, n_batch]
    // Memory buffers used to evaluate the model.
    std::vector<uint8_t> compute_buffer;
};


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
// What does cur mean? Can we find a more descriptive name for it?
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
// Currently wip, compiles but not tested.
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
    batch.seq_id = (int32_t **)malloc(sizeof(int32_t *) * (n_tokens_alloc + 1));
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
    batch.seq_id[n_tokens_alloc] = nullptr;
    batch.logits = (int8_t *)malloc(sizeof(int8_t) * n_tokens_alloc);
    if (!batch.logits) {
        printf("coulld not allocate memory for moondream_batch logits\n");
        return false;
    }
    
    return true;
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
    
    bool result = moondream_init_kv_cache(mctx.kv_cache, hparams, cparams, mctx.backend_cpu, type_k, type_v);
    if (!result) {
        printf("failed to initialize moondream_kv_cache\n");
        return false;
    }
    printf("succesfully initialized moondream_kv_cache\n");

    // Buffer used to store the computation graph and the tensor meta data.
    mctx.compute_buffer.resize(
        ggml_tensor_overhead() * LLAMA_MAX_NODES + ggml_graph_overhead_custom(LLAMA_MAX_NODES, false)
    );

    // TODO: equivalent of llama_output_reserve(), see llama.cpp line 11949

    return true;
}

bool moondream_load_model(const char * gguf_file_path, moondream_model & model) {
    ggml_context * ctx;
    gguf_init_params init_params = {.no_alloc = false, .ctx = &ctx};
    gguf_context * meta = gguf_init_from_file(gguf_file_path, init_params);
    if(meta == NULL) {
        return false;
    }
    int gguf_version = gguf_get_version(meta);
    size_t gguf_alignment = gguf_get_alignment(meta);
    size_t gguf_data_offset = gguf_get_data_offset(meta);
    const char * model_arch = gguf_get_val_str(meta, gguf_find_key(meta, "general.architecture"));
    
    moondream_hparams hparams;
    const char * model_name = gguf_get_val_str(meta, gguf_find_key(meta, "general.name"));
    hparams.n_ctx_train = gguf_get_val_u32(meta, gguf_find_key(meta, ARCH_PREFIX("context_length")));
    hparams.n_embd = gguf_get_val_u32(meta, gguf_find_key(meta, ARCH_PREFIX("embedding_length")));
    hparams.n_rot = gguf_get_val_u32(meta, gguf_find_key(meta, ARCH_PREFIX("rope.dimension_count")));
    hparams.n_layer = gguf_get_val_u32(meta, gguf_find_key(meta, ARCH_PREFIX("block_count")));
    hparams.n_ff = gguf_get_val_u32(meta, gguf_find_key(meta, ARCH_PREFIX("block_count")));
    hparams.n_head = gguf_get_val_u32(meta, gguf_find_key(meta, ARCH_PREFIX("attention.head_count")));
    hparams.n_head_kv = gguf_get_val_u32(meta, gguf_find_key(meta, ARCH_PREFIX("attention.head_count_kv")));
    
    // Calculate n_head_k and n_head_v because they are not specified.
    hparams.n_embd_head_k = hparams.n_embd / hparams.n_head;
    hparams.n_embd_head_v = hparams.n_embd_head_k;
    // TODO: verify that the GQA hparams are correct. Reference llama.cpp lines 1922 and 1926.
    hparams.n_embd_k_gqa = hparams.n_embd_head_k * hparams.n_head_kv;
    hparams.n_embd_v_gqa = hparams.n_embd_head_v * hparams.n_head_kv;
    // Old GQA hparams for reference:
    // hparams.n_embd_k_gqa = hparams.n_embd_head_k;
    // hparams.n_embd_v_gqa = hparams.n_embd_v_gqa;
    // TODO: determine this dynamically from the GGUF file instead of hardcoding it
    hparams.n_vocab = 51200;

    printf("loaded %s from %s\n", model_name, gguf_file_path);
    printf("gguf_version: %d\n", gguf_version);
    printf("gguf_alignment: %ld\n", gguf_alignment);
    printf("gguf_data_offset: %ld\n", gguf_data_offset);
    printf("model_arch: %s\n", model_arch);
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
    printf("found %s\n", cur->name);
#endif
    model.tok_embd = cur; // token_embd.weight
    
    const int n_tensors_per_layer = 10;
    for (int i = 0; i < hparams.n_layer; ++i) {
        moondream_layer cur_layer;
        for (int k = 0; k < n_tensors_per_layer; ++k) {
            cur = ggml_get_next_tensor(ctx, cur);
#ifdef MOONDREAM_EXTRA_LOGS 
            printf("found %s\n", cur->name);
#endif
            if (cur == NULL) {
                return false;
            }
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
#ifdef MOONDREAM_EXTRA_LOGS 
        printf("found %s\n", cur->name);
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

    model.ctx = ctx;
    model.hparams = hparams;
    return true;
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
    
    moondream_model model;
    bool result = moondream_load_model(text_model_path, model);
    if (!result) {
        printf("could not load model\n");
        return 1;
    }
    printf("succesfully loaded model\n");

    moondream_cparams cparams = {
        .n_ctx = 512,
        .n_batch = 1,
        .n_ubatch = 1,
        .n_seq_max = 1,
        .n_threads = 1,
        .n_threads_batch = 1,
        // TODO: figure out what these shoud be
        .rope_freq_base = 0.0f,
        .rope_freq_scale = 0.0f,
        .n_ctx_orig_yarn = 0, 
        .yarn_ext_factor = 0.0f,
        .yarn_attn_factor = 0.0f,
        .yarn_beta_fast = 0.0f,
        .yarn_beta_slow = 0.0f,
        .defrag_thold = 0.0f,
        // -----------------
        .embeddings = true,
        .causal_attn = true,
        .offload_kqv = false,
        .flash_attn = false
    };
    const ggml_type type_k = GGML_TYPE_F16;
    const ggml_type type_v = GGML_TYPE_F16;
    moondream_context mctx;
    result = moondream_init_context(mctx, model.hparams, cparams, type_k, type_v);
    if (!result) {
        printf("failed to initialze moondream_context\n");
        return 1;
    }
    printf("succesfully initialized moondream_context\n");
    
    moondream_batch batch;
    result = moondream_init_batch(batch, cparams.n_ctx, model.hparams.n_embd, false, cparams.n_seq_max);
    if (!result) {
        printf("failed to initialized moondream_batch\n");
        return 1;
    }
    batch.n_tokens = 1;
    printf("succesfully initialized moondream_batch\n");
    
    // Set batch tokens to some dummy ID for testing.
    for (int i = 0; i < cparams.n_ctx; ++i) {
        batch.token[i] = (model.hparams.n_vocab / 2) + i;
    }
    printf("set batch tokens\n");

// Token generation is not working yet because tensor backend buffers need to be set during the cgraph
// build step. Uncomment the following line to compile this section and see the runtime error.
//#define MOONDREAM_WIP_GEN_STEPS
#ifdef MOONDREAM_WIP_GEN_STEPS
    int n_gen = 128;
    for (int i = 0; i < n_gen; ++i) {
        ggml_cgraph * phi2_cgraph = build_phi2(model, batch, mctx);
        printf("built graph\n");
        // ggml_backend_tensor_set() produces the following error:
        // GGML_ASSERT: ../dependencies/ggml/src/ggml-backend.c:224: buf != NULL && "tensor buffer not set"
        // Somewhere during moondream_init_context() or build_phi2(), a step was skipped where the 
        // backend buffers for the moondream_context tensors was supposed to be set.
        // I think the build callback is where the backend for each tensor is set.
        ggml_backend_tensor_set(
            mctx.inp_tokens, batch.token, 0, batch.n_tokens * ggml_element_size(mctx.inp_tokens)
        );
        /*for (int k = 0; k < batch.n_tokens; ++k) {
            ggml_set_i32_1d(mctx.inp_tokens, k, (int32_t)batch.token[i]);
            printf("%d\n", k);
        }*/
        printf("generation step %d\n", i);
    }
#endif
    
    return 0;
}
