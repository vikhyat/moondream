#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <cmath>
#include <cstdarg>
#include <climits>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <queue>
#include "unicode.h"
#include "ggml.h"
#include "ggml-backend.h"

#ifndef MOONDREAM_LIBRARY_BUILD
#include "stb_image.h" 
#endif

#define MD_TEXT_MODEL_FNAME "moondream2-text-model-f16.gguf"
#define MD_MMPROJ_FNAME "moondream2-mmproj-f16.gguf"
#define DATA_PATH_MAX_LEN 512
#define ARCH_PREFIX(t) ("phi2." t)
#define TOK_PREFIX(t) ("tokenizer.ggml." t)
#define LLAMA_MAX_NODES 8192
// Corresponds to LLAMA_ROPE_TYPE_NEOX from llama.cpp which is what is used for phi2.
#define MOONDREAM_ROPE_TYPE 2
#define MOONDREAM_N_IMAGE_CHANNELS 3
// Rope scaling type should be: LLAMA_ROPE_SCALING_TYPE_LINEAR
// Define MOONDREAM_EXTRA_LOGS if you want additional logs for debugging.
//#define MOONDREAM_EXTRA_LOGS 
// Define MOONDREAM_MULTI_MODAL if you want image embeddings to be generated and used for text model.
//#define MOONDREAM_MULTI_MODAL

/* Start of helpers. */
static size_t utf8_len(char src) {
    const size_t lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}

static double bytes_to_gib(size_t n_bytes) {
    return static_cast<double>(n_bytes) / (1024.0 * 1024.0 * 1024.0);
}

static bool size_to_int32(size_t s, int32_t * i) {
    if (s <= static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        *i = static_cast<int32_t>(s);
        return true;
    }
    return false;
}

static void moondream_set_tensor_name(ggml_tensor * cur, const char * name, int il) {
    if (il >= 0) {
        ggml_format_name(cur, "%s-%d", name, il);
    } else {
        ggml_set_name(cur, name);
    }
}
/* End of helpers. */

enum projector_type {
    PROJECTOR_TYPE_MLP,
    PROJECTOR_TYPE_UNKNOWN,
};
/* End of llm enums. */

struct moondream_lm_layer {
    // Normalization
    ggml_tensor * attn_norm = nullptr;
    ggml_tensor * attn_norm_b = nullptr;
    // Attention
    ggml_tensor * wo = nullptr;
    ggml_tensor * wqkv = nullptr;
    // Attention bias
    ggml_tensor * bo = nullptr;
    ggml_tensor * bqkv = nullptr;
    // Feed forward
    ggml_tensor * ffn_down = nullptr;  // w2
    ggml_tensor * ffn_up = nullptr;   // w3
    // Feed forward bias
    ggml_tensor * ffn_down_b = nullptr; // b2
    ggml_tensor * ffn_up_b = nullptr;   // b3
};

struct moondream_mmproj_layer {
    // Attention
    ggml_tensor * k_w = nullptr;
    ggml_tensor * k_b = nullptr;
    ggml_tensor * q_w = nullptr;
    ggml_tensor * q_b = nullptr;
    ggml_tensor * v_w = nullptr;
    ggml_tensor * v_b = nullptr;
    ggml_tensor * o_w = nullptr;
    ggml_tensor * o_b = nullptr;

    // Layernorm 1 
    ggml_tensor * ln_1_w = nullptr;
    ggml_tensor * ln_1_b = nullptr;

    // Feed forward
    ggml_tensor * ff_i_w = nullptr;
    ggml_tensor * ff_i_b = nullptr;
    ggml_tensor * ff_o_w = nullptr;
    ggml_tensor * ff_o_b = nullptr;

    // Layernorm 2
    ggml_tensor * ln_2_w = nullptr;
    ggml_tensor * ln_2_b = nullptr;
};

struct moondream_lm_hparams {
    int n_embd;
    int n_ff;
    int n_layer;
    int n_rot;
    uint32_t n_ctx_train;
    int n_head;
    int n_head_kv;
    int n_embd_head_k;
    int n_embd_head_v;
    int n_vocab;
    
    float f_norm_eps;
    float f_norm_rms_eps;

    // max bias for attention, not sure if it's used for anything else
    float f_max_alibi_bias;
};

struct moondream_mmproj_hparams {
    uint32_t image_size;
    uint32_t patch_size;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_proj;
    uint32_t n_head;
    uint32_t n_layer;
    float f_norm_eps;
    bool use_gelu;
    projector_type proj_type;
    float * image_mean;
    float * image_std;
};

struct moondream_lm_cparams {
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
};

struct moondream_vocab {
    int64_t bos_token_id;
    int64_t eos_token_id;
    int64_t unknown_token_id;
    int64_t separator_token_id;
    int64_t padding_token_id;
    int n_tokens;
    int n_merges;
    const float * scores;
    const int32_t * token_type;
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int32_t> token_to_id;
    std::map<std::pair<std::string, std::string>, int> bpe_ranks;
};

struct moondream_lm {
    ggml_context * ctx = nullptr;
    moondream_lm_hparams hparams;
    moondream_vocab vocab;
    std::vector<moondream_lm_layer> layers;
    ggml_tensor * tok_embd = nullptr;
    ggml_tensor * output_norm = nullptr;
    ggml_tensor * output_norm_b = nullptr;
    ggml_tensor * output = nullptr;
    ggml_tensor * output_b = nullptr;
};

struct moondream_mmproj {
    ggml_context * ctx = nullptr;
    moondream_mmproj_hparams hparams;
    std::vector<moondream_mmproj_layer> layers;
    ggml_tensor * mm_0_w = nullptr;
    ggml_tensor * mm_0_b = nullptr;
    ggml_tensor * mm_2_w = nullptr;
    ggml_tensor * mm_2_b = nullptr;
    ggml_tensor * pos_embd = nullptr;
    ggml_tensor * patch_embd = nullptr;
    ggml_tensor * patch_bias = nullptr;
    ggml_tensor * post_ln_w = nullptr;
    ggml_tensor * post_ln_b = nullptr;
};

// Arrays must have size of n_tokens
struct moondream_lm_batch {
    int32_t n_tokens_alloc;
    int32_t n_tokens;
    // The token ids of the input (used when embd is NULL).
    int32_t * token = nullptr;
    // The token embeddings (used when token is NULL).
    float * embd = nullptr;
    // The positions of the respective tokens in the sequence.
    int32_t * pos = nullptr;
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
    // Apparently it is meant to optimize the size of the kv_cache that is considered
    // for each step.
    uint32_t n = 0;

    ggml_type type_k = GGML_TYPE_F16;
    ggml_type type_v = GGML_TYPE_F16;
    
    // k and v caches for each layer.
    std::vector<struct ggml_tensor *> k_l;
    std::vector<struct ggml_tensor *> v_l;

    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
};

struct moondream_lm_context {
    ggml_context * ctx = nullptr;
    moondream_lm_cparams cparams;
    moondream_kv_cache kv_cache;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_buffer_type_t backend_cpu_buft = nullptr;

    // The number of tokens in the current sequence, including prompt tokens, previously generated tokens,
    // and tokens in the current batch. When a token is added to the current batch, its pos is set 
    // to n_ctx_active and then n_ctx_active is incremented.
    int32_t n_ctx_active = 0;
    int n_outputs = 0;
    // Input tensors.
    ggml_tensor * inp_tokens = nullptr;    // I32 [n_batch]
    ggml_tensor * inp_embd = nullptr;      // F32 [n_embd, n_batch]
    ggml_tensor * inp_pos = nullptr;       // I32 [n_batch]
    ggml_tensor * inp_out_ids = nullptr;   // I32 [n_outputs]
    ggml_tensor * inp_kq_mask = nullptr;   // F32 [kv_size, n_batch]
    // Memory buffers used to evaluate the model.
    std::vector<uint8_t> compute_buffer;
    ggml_backend_sched_t sched = nullptr;
};

struct moondream_mmproj_context {
    int n_patches_per_side = 0;
    int n_patches = 0;
    int n_positions = 0;
    int n_output_elements = 0;
    ggml_context * ctx = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_buffer_type_t backend_cpu_buft = nullptr;
    // Input tensors.
    ggml_tensor * inp_raw = nullptr;
    ggml_tensor * positions = nullptr;
    // Memory buffers used to evaluate the model.
    std::vector<uint8_t> compute_buffer;
    ggml_backend_sched_t sched = nullptr;
    float * output_buffer = nullptr;
};

struct moondream_image {
    int n_xy = 0;
    int n_channels = 0;
    int n_scalars = 0;
    int n_positions = 0;
    float * data = nullptr;
    int32_t * pos = nullptr;
};

struct moondream_api_state {
    bool is_init = false;
    moondream_lm model;
    moondream_mmproj mmproj_model;
    moondream_lm_context mctx;
    moondream_mmproj_context mmproj_ctx;
};

static moondream_api_state api_state;

static void log_tensor(ggml_tensor * dst, const ggml_tensor * src, int ith, int nth, void * userdata) {
    if (ith != 0) {
        // Only log from the first thread.
        return;
    }

    printf("Shape: %lld %lld %lld %lld\n", dst->ne[3], dst->ne[2], dst->ne[1], dst->ne[0]);
    switch (dst->type) {
        case GGML_TYPE_F16:
            printf("Type: f16\n");
            break;
        case GGML_TYPE_F32:
            printf("Type: f32\n");
            break;
        default:
            printf("Type: unknown\n");
            break;
    }

    // Emit last 2 dimension values.
    for (int j = 0; j < dst->ne[1]; j++) {
        for (int i = 0; i < dst->ne[0]; i++) {
            if (i > 0) {
                printf("\t");
            }
            float f = ggml_get_f32_nd(dst, i, j, 0, 0);
            printf("%.7f", (double)f);
        }
        printf("\n");
    }
}

ggml_tensor * lm_build_inp_embd(
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

ggml_tensor * lm_build_inp_pos(ggml_context * ctx, moondream_lm_context & mctx, moondream_lm_batch & batch) {
    mctx.inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.n_tokens);
    ggml_set_input(mctx.inp_pos);
    return mctx.inp_pos;
}

ggml_tensor * lm_build_inp_kq_mask(
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

ggml_tensor * lm_build_norm(
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
        moondream_set_tensor_name(cur, "norm_w", il);
        cur = ggml_add(ctx, cur, mb);
    }
    return cur;
}

// Maybe this should be renamed to lm_build_kv_cache?
void lm_build_kv_store(
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
    
    // Why use GGML_ASSERT here and the regular c assert below?
    GGML_ASSERT(kv.size == n_ctx);

    ggml_tensor * k_cache_view = ggml_view_1d(
        ctx, kv.k_l[il], n_tokens*n_embd, 
        // Why are there parentheses around ggml_row_size?
        (ggml_row_size(kv.k_l[il]->type, n_embd))*kv_head
    );
    moondream_set_tensor_name(k_cache_view, "k_cache_view", il);
    ggml_build_forward_expand(graph, ggml_cpy(ctx, k_cur, k_cache_view));

    assert(v_cur->ne[0] == n_embd && v_cur->ne[1] == n_tokens);

    ggml_tensor * v_cache_view = nullptr;
    if (cparams.flash_attn) {
        v_cache_view = ggml_view_1d(
            ctx, kv.v_l[il], n_tokens*n_embd, 
            // Why are there parantheses around kv_head?
            (kv_head)*ggml_row_size(kv.v_l[il]->type, n_embd)
        );
    } else {
        // TODO: figure out exactly what view 2d is doing under the hood
        // The v cache is transposed when not using flash attention.
        v_cache_view = ggml_view_2d(
            ctx, kv.v_l[il], n_tokens, n_embd, 
            (n_ctx)*ggml_element_size(kv.v_l[il]),
            (kv_head)*ggml_element_size(kv.v_l[il])
        );
        v_cur = ggml_transpose(ctx, v_cur);
    }
    moondream_set_tensor_name(v_cache_view, "v_cache_view", il);
    ggml_build_forward_expand(graph, ggml_cpy(ctx, v_cur, v_cache_view));
}

ggml_tensor * lm_build_kqv(
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
    // TODO: figure out exactly how ggml_view_3d works under the hood
    ggml_tensor * k = ggml_view_3d(
        ctx, kv.k_l[il],
        n_embd_head_v, n_kv, n_head_kv,
        ggml_row_size(kv.k_l[il]->type, n_embd),
        ggml_row_size(kv.k_l[il]->type, n_embd_head_k),
        0
    );
    moondream_set_tensor_name(k, "k", il);

    ggml_tensor * cur;
    if (cparams.flash_attn) {
        // llama uses GGML_UNUSED here but I'm not sure what it does
        // see llama.cpp line 6989 for more details

        // Split cached v into n_head heads (not transposed).
        ggml_tensor * v = ggml_view_3d(
            ctx, kv.v_l[il], 
            n_embd_head_v, n_kv, n_head_kv,
            ggml_row_size(kv.v_l[il]->type, n_embd),
            ggml_row_size(kv.v_l[il]->type, n_embd_head_v),
            0
        );
        moondream_set_tensor_name(v, "v", il);
        cur = ggml_flash_attn_ext(ctx, q, k, v, kq_mask, kq_scale, hparams.f_max_alibi_bias);
        // For phi2 the KQ multiplication must be done with F32 precision, otherwise we get NaNs.
        // Ref: https://github.com/ggerganov/llama.cpp/pull/4490#issuecomment-1859055847
        ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);
        cur = ggml_reshape_2d(ctx, cur, n_embd_head_v*n_head, n_tokens);
    } else {
        ggml_tensor * kq = ggml_mul_mat(ctx, k, q);
        moondream_set_tensor_name(kq, "kq", il);
        // For phi2 the KQ multiplication must be done with F32 precision, otherwise we get NaNs.
        // Ref: https://github.com/ggerganov/llama.cpp/pull/4490#issuecomment-1859055847
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
        kq = ggml_soft_max_ext(ctx, kq, kq_mask, kq_scale, hparams.f_max_alibi_bias);
        moondream_set_tensor_name(kq, "kq_soft_max_ext", il);
        GGML_ASSERT(kv.size == n_ctx);
        // Split cached v into n_head heads.
        ggml_tensor * v = ggml_view_3d(
            ctx, kv.v_l[il], 
            n_kv, n_embd_head_v, n_head_kv,
            ggml_element_size(kv.v_l[il])*n_ctx,
            ggml_element_size(kv.v_l[il])*n_ctx*n_embd_head_v,
            0
        );
        moondream_set_tensor_name(v, "v", il);
        ggml_tensor * kqv = ggml_mul_mat(ctx, v, kq);
        moondream_set_tensor_name(kqv, "kqv", il);
        ggml_tensor * kqv_merged = ggml_permute(ctx, kqv, 0, 2, 1, 3);
        moondream_set_tensor_name(kqv_merged, "kqv_merged", il);
        // Make contiguous, with new shape.
        cur = ggml_cont_2d(ctx, kqv_merged, n_embd_head_v*n_head, n_tokens);
        moondream_set_tensor_name(cur, "kqv_merged_cont", il);
    }
    
    ggml_build_forward_expand(graph, cur);
    cur = ggml_mul_mat(ctx, wo, cur);
    if (wo_b) {
        // Only set the name of the output projection if there is also a bias.
        // The bias name will be set outside the function.
        moondream_set_tensor_name(cur, "kqv_wo", il);
        cur = ggml_add(ctx, cur, wo_b);
    }
    return cur;
}

ggml_tensor * lm_build_kv(
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

    lm_build_kv_store(ctx, hparams, cparams, kv, graph, k_cur, v_cur, n_tokens, kv_head, il);
    ggml_tensor * cur;
    cur = lm_build_kqv(
        ctx, model, hparams, cparams, kv, graph, wo, wo_b, 
        q_cur, kq_mask, n_tokens, n_kv, kq_scale, il
    );
    moondream_set_tensor_name(cur, "kqv_out", il);
    return cur;
}

ggml_tensor * lm_build_inp_out_ids(ggml_context * ctx, moondream_lm_context & mctx, int n_outputs) {
    mctx.inp_out_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_outputs);
    moondream_set_tensor_name(mctx.inp_out_ids, "inp_out_ids", -1);
    ggml_set_input(mctx.inp_out_ids);
    return mctx.inp_out_ids;
}

// Modification of llama.cpp build_phi2.
// Ref: https://github.com/ggerganov/llama.cpp/blob/da799b41891e34aac86ce4e173f9c4c0afd4fab3/llama.cpp
ggml_cgraph * build_phi2(
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
        moondream_set_tensor_name(attn_norm_output, "attn_norm", il);

        // Self-attention
        {
            struct ggml_tensor * q_cur = nullptr;
            struct ggml_tensor * k_cur = nullptr;
            struct ggml_tensor * v_cur = nullptr;

            cur = ggml_mul_mat(ctx0, model.layers[il].wqkv, attn_norm_output);
            moondream_set_tensor_name(cur, "wqkv", il);
            cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
            moondream_set_tensor_name(cur, "bqkv", il);

            q_cur = ggml_cont(
                ctx0, ggml_view_2d(ctx0, cur, n_embd, n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd))
            );
            k_cur = ggml_cont(
                ctx0, ggml_view_2d(ctx0, cur, n_embd, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd))
            );
            v_cur = ggml_cont(
                ctx0, ggml_view_2d(ctx0, cur, n_embd, n_tokens, cur->nb[1], 2*sizeof(float)*(n_embd))
            );
  
            moondream_set_tensor_name(q_cur, "q_cur", il);
            moondream_set_tensor_name(k_cur, "k_cur", il);
            moondream_set_tensor_name(v_cur, "v_cur", il);
            
            q_cur = ggml_reshape_3d(ctx0, q_cur, n_embd_head, n_head, n_tokens);
            k_cur = ggml_reshape_3d(ctx0, k_cur, n_embd_head, n_head_kv, n_tokens);

            q_cur = ggml_rope_ext(
                ctx0, q_cur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
            );
            moondream_set_tensor_name(q_cur, "q_cur", il);

            // With phi2, we scale the Q to avoid precision issues.
            // Ref: https://github.com/ml-explore/mlx-examples/blob/08e862336ade809bc37d1035f94b359e7d1a5152/phi2/phi2.py#L64-L66
            q_cur = ggml_scale(ctx0, q_cur, 1.0f/sqrtf(float(n_embd_head)));
            moondream_set_tensor_name(q_cur, "q_cur", il);
            
            k_cur = ggml_rope_ext(
                ctx0, k_cur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
            );
            moondream_set_tensor_name(k_cur, "k_cur", il);

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
            
            moondream_set_tensor_name(ffn_output, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, ffn_output);
        cur = ggml_add(ctx0, cur, inpL);
        moondream_set_tensor_name(cur, "l_out", il);
        
        // Set input for next layer.
        inpL = cur;
    }

    cur = lm_build_norm(
        ctx0, inpL, hparams,
        model.output_norm,
        model.output_norm_b,
        -1
    );
    moondream_set_tensor_name(cur, "result_norm", -1);

    cur = ggml_mul_mat(ctx0, model.output, cur);
    moondream_set_tensor_name(cur, "result_output_no_bias", -1);

    cur = ggml_add(ctx0, cur, model.output_b);
    moondream_set_tensor_name(cur, "result_output", -1);
    
    ggml_build_forward_expand(gf, cur);
    ggml_free(ctx0);
    return gf;
}

// Modification of llama.cpp/examples/llava/clip.pp clip_image_build_graph.
// Ref: https://github.com/ggerganov/llama.cpp/blob/da799b41891e34aac86ce4e173f9c4c0afd4fab3/examples/llava/clip.cpp
ggml_cgraph * build_clip(
    moondream_mmproj & model,
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
    const int num_patches_per_side = mctx.n_patches_per_side;
    const int num_patches = mctx.n_patches;
    const int num_positions = mctx.n_positions;
    const int n_embd = hparams.n_embd;
    const int n_head = hparams.n_head;
    const int n_head_qkv = n_embd / n_head;
    const int n_layer = hparams.n_layer;
    const float eps = hparams.f_norm_eps; 
    
    // TODO: move this into moondream_mmproj_batch struct.
    constexpr int batch_size = 1;

    ggml_init_params build_ctx_params = {
        mctx.compute_buffer.size(),
        mctx.compute_buffer.data(),
        true
    };
    ggml_context * ctx0 = ggml_init(build_ctx_params);
    ggml_cgraph * gf = ggml_new_graph(ctx0);

    ggml_tensor * inp_raw = ggml_new_tensor_4d(
        ctx0, GGML_TYPE_F32, image_size, image_size, MOONDREAM_N_IMAGE_CHANNELS, batch_size
    );
    ggml_set_name(inp_raw, "inp_raw");
    ggml_set_input(inp_raw);
    mctx.inp_raw = inp_raw;

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
    mctx.positions = positions;
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
    if (batch.token) { free(batch.token); }
    if (batch.embd) { free(batch.embd); }
    if (batch.pos) { free(batch.pos); }
}

bool moondream_kv_cache_init(
    moondream_kv_cache & kv_cache,
    moondream_lm_hparams & hparams, 
    moondream_lm_cparams & cparams, 
    ggml_backend_t backend,
    ggml_type type_k,
    ggml_type type_v
) {
    // TODO: double check this
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
    printf("succesfully allocated memory for moondream_kv_cache tensors\n");
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
    ggml_type type_v
) {
    memcpy(&mctx.cparams, &cparams, sizeof(moondream_lm_cparams));
    
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
    
    bool result = moondream_kv_cache_init(mctx.kv_cache, hparams, cparams, mctx.backend_cpu, type_k, type_v);
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
    mctx.n_outputs = 1; // TODO: figure out what n_outputs should be during initialization.
    ggml_cgraph * gf = build_phi2(model, dummy_batch, mctx);
    if (!ggml_backend_sched_reserve(mctx.sched, gf)) {
        printf("failed to reserve buffers for compute graph\n");
        return false;
    }
    printf("succesfully reserved buffers for compute graph\n");
    moondream_lm_batch_free(dummy_batch);

    // TODO: equivalent of llama_output_reserve(), see llama.cpp line 11949

    return true;
}

// WIP implementation of mmproj context initializaton.
// Currently fails with:
// GGML_ASSERT: ggml/src/ggml-backend.c:1431: node_backend_id != -1
bool moondream_mmproj_context_init(
    moondream_mmproj_context & mctx, 
    moondream_mmproj & model, 
    int n_threads
) {
    const moondream_mmproj_hparams & hparams = model.hparams;
    mctx.n_patches_per_side = hparams.image_size / hparams.patch_size;
    mctx.n_patches = mctx.n_patches_per_side * mctx.n_patches_per_side;
    mctx.n_positions = mctx.n_patches; /* + (ctx->has_class_embedding ? 1 : 0); */
    
    mctx.n_output_elements = mctx.n_patches * hparams.n_proj;
    mctx.output_buffer = (float *)malloc(sizeof(float) * mctx.n_output_elements);
    if (!mctx.output_buffer) {
        printf("failed to allocate memory for mmproj output buffer\n");
        return false;
    }

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
    ggml_cgraph * gf = build_clip(model, mctx);
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

void moondream_lm_context_free(moondream_lm_context & mctx) {
    if (mctx.backend_cpu) { ggml_backend_free(mctx.backend_cpu); }
    if (mctx.sched) { ggml_backend_sched_free(mctx.sched); }
    if (mctx.kv_cache.buf) { ggml_backend_buffer_free(mctx.kv_cache.buf); }
    if (mctx.kv_cache.ctx) { ggml_free(mctx.kv_cache.ctx); }
    if (mctx.ctx) { ggml_free(mctx.ctx); }
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
    //printf("left: %s, right: %s, rank found: %d\n", left_token.c_str(), right_token.c_str(), rank_found);
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
int32_t moondream_tokenize(
    moondream_vocab & vocab, 
    const char * text,
    int32_t text_len,
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

    int32_t n_token_ids = 0;
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

bool moondream_load_lm_from_file(const char * gguf_file_path, moondream_lm & model) {
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
    // TODO: determine this dynamically from the GGUF file instead of hardcoding it
    hparams.n_vocab = 51200;
    hparams.f_max_alibi_bias = 0.0f;
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
    printf("bos_token_id: %ld\n", vocab.bos_token_id);
    printf("eos_token_id: %ld\n", vocab.eos_token_id);
    printf("unknown_token_id: %ld\n", vocab.separator_token_id);
    printf("separator_token_id: %ld\n", vocab.separator_token_id);
    printf("padding_token_id: %ld\n", vocab.padding_token_id);
    printf("n_tokens: %d\n", vocab.n_tokens);
    printf("n_merges: %d\n", vocab.n_merges);
    printf("------------\n");
    
    gguf_free(meta);
    return true;
}

bool moondream_load_mmproj_from_file(const char * gguf_file_path, moondream_mmproj & model) {
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
    printf("gguf_alignment: %zu\n", gguf_alignment);
    printf("gguf_data_offset: %zu\n", gguf_data_offset);
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

bool moondream_lm_set_inputs(moondream_lm_context & mctx, moondream_lm_batch & batch) {
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

static std::string moondream_decode_token_str(const std::string & text) {
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

static bool moondream_lm_prefill_embeddings(
    moondream_lm_context & mctx,
    moondream_lm & model,
    moondream_lm_batch & batch,
    int embd_dim
) {
    // TODO: set n_output to zero. We don't want any logits when pre-filling the embeddings.

    ggml_cgraph * gf = build_phi2(model, batch, mctx);
    
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

static int moondream_lm_decode_inner(
    moondream_lm_context & mctx,
    moondream_lm & model,
    moondream_lm_batch & batch
) {
    ggml_cgraph * gf = build_phi2(model, batch, mctx);
        
    if (!ggml_backend_sched_alloc_graph(mctx.sched, gf)) {
        printf("failed to allocate graph for ggml_backend_sched_t\n");
        return -1;
    }
    
    if (!moondream_lm_set_inputs(mctx, batch)) {
        printf("failed to set model inputs\n");
        return -1;
    }

    const enum ggml_status compute_status = ggml_backend_sched_graph_compute(mctx.sched, gf);
    if (compute_status != GGML_STATUS_SUCCESS) {
        printf("graph computation failed (%s)\n", ggml_status_to_string(compute_status));
        return -1;
    }
    ggml_backend_sched_synchronize(mctx.sched);

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
    
    mctx.n_ctx_active = 0;
    int sampled_token_id = -1;
    std::string local_response = "";

    if (mmproj_embd) {
        // TODO: cleanup this interface. moondream_batch is too entangled with everything.
        
        // Temporarily replace token ID batch with embd batch while re-using position buffer.
        assert(batch.n_token_alloc >= n_embd);
        int32_t * temp_token = batch.token;
        batch.token = nullptr;
        batch.embd = mmproj_embd;
        batch.n_tokens = n_embd;
        for (int i = 0; i < batch.n_tokens; ++i) {
            batch.pos[i] = mctx.n_ctx_active;
            ++mctx.n_ctx_active;
        }
        moondream_lm_prefill_embeddings(mctx, model, batch, embd_dim);
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
    sampled_token_id = moondream_lm_decode_inner(mctx, model, batch);
    if (sampled_token_id < 0) {
        // Negative token IDs are invalid, so something went wrong.
        return false;
    }
    local_response += moondream_decode_token_str(model.vocab.id_to_token[sampled_token_id]);
    if (log_response_stream) {
        printf("%s\n", local_response.c_str());
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
        
        sampled_token_id = moondream_lm_decode_inner(mctx, model, batch);
        if (sampled_token_id < 0) {
            // Negative token IDs are invalid, so something went wrong.
            return false;
        }
        local_response += moondream_decode_token_str(model.vocab.id_to_token[sampled_token_id]);
        if (log_response_stream) {
            printf("%s\n", local_response.c_str());
        }
        if (sampled_token_id == model.vocab.eos_token_id) {
            break;
        }
    }
    printf("\n");
    response = local_response;
    return true;
}

bool moondream_mmproj_embed(
    moondream_mmproj_context & mctx, 
    moondream_mmproj & model,
    moondream_image & image
) {
    ggml_cgraph * gf = build_clip(model, mctx);
    if (!gf) {
        printf("failed to build mmproj compute graph\n");
        return false;
    }
    if (!ggml_backend_sched_alloc_graph(mctx.sched, gf)) {
        printf("failed to allocate graph for ggml_backend_sched_t\n");
        return false;
    }

    ggml_backend_tensor_set(
        mctx.inp_raw, image.data, 0, image.n_scalars * ggml_element_size(mctx.inp_raw)
    );
    ggml_backend_tensor_set(
        mctx.positions, image.pos, 0, image.n_positions * ggml_element_size(mctx.positions)
    );
    
    const enum ggml_status compute_status = ggml_backend_sched_graph_compute(mctx.sched, gf);
    if (compute_status != GGML_STATUS_SUCCESS) {
        printf("graph computation failed (%s)\n", ggml_status_to_string(compute_status));
        return false;
    }
    ggml_backend_sched_synchronize(mctx.sched);
    
    ggml_tensor * embeddings = gf->nodes[gf->n_nodes - 1];
    assert(mctx.n_output_elements == ggml_nelements(embeddings));
    memcpy(mctx.output_buffer, embeddings->data, sizeof(float) * mctx.n_output_elements); 
    // NOTE: embeddings may need to be transposed.
    /*printf(
        "embeddings shape: %d %d %d %d\n", 
        embeddings->ne[0], embeddings->ne[1], embeddings->ne[2], embeddings->ne[3]
    );*/

    ggml_backend_sched_reset(mctx.sched);
    return true;
}

bool moondream_api_state_init(const char * text_model_path, const char * mmproj_path, uint32_t n_threads) {
    if (api_state.is_init) {
        printf("API has already been initialized\n");
        return false;
    }

    /* Start of moondream_lm load. */
    bool result = moondream_load_lm_from_file(text_model_path, api_state.model);
    if (!result) {
        printf("could not load text model\n");
        return false;
    }
    printf("succesfully loaded text model\n");
    /* End of moondream_lm load. */

    /* Start of moondream_mmproj load. */
    result = moondream_load_mmproj_from_file(mmproj_path, api_state.mmproj_model);
    if (!result) {
        printf("could not load mmproj model\n");
        return false;
    }
    printf("succesfully loaded mmproj model\n");
    /* End of moondream_mmproj load. */

    /* Start of moondream_mmproj_context init. */
    result = moondream_mmproj_context_init(api_state.mmproj_ctx, api_state.mmproj_model, n_threads);
    if (!result) {
        printf("failed to initialze moondream_mmproj_context\n");
        return 1;
    }
    printf("succesfully initialized moondream_lm_context\n");
    /* End of moondream_mmproj_context init. */

    /* Start of moondream_lm_context init. */
    moondream_lm_cparams cparams = {
        .n_ctx = 2048,/*api_state.model.hparams.n_ctx_train,*/
        .n_batch = 2048,
        .n_ubatch = 512,
        .n_seq_max = 1,
        .n_threads = n_threads,
        .n_threads_batch = n_threads,
        // TODO: figure out what these shoud be
        .rope_freq_base = 10000.0f,
        .rope_freq_scale = 1.0f,
        .n_ctx_orig_yarn = api_state.model.hparams.n_ctx_train, 
        .yarn_ext_factor = 0.0f,
        .yarn_attn_factor = 1.0f,
        .yarn_beta_fast = 32.0f,
        .yarn_beta_slow = 1.0f,
        .defrag_thold = -1.0f,
        // -----------------
        .embeddings = false,
        .causal_attn = true,
        .offload_kqv = false,
        .flash_attn = false
    };
    const ggml_type type_k = GGML_TYPE_F16;
    const ggml_type type_v = GGML_TYPE_F16;
    result = moondream_lm_context_init(
        api_state.mctx, api_state.model.hparams, cparams, api_state.model, type_k, type_v
    );
    if (!result) {
        printf("failed to initialze moondream_lm_context\n");
        return false;
    }
    api_state.mctx.n_outputs = 1;
    printf("succesfully initialized moondream_lm_context\n");
    /* End of moondream_lm_context init. */

    api_state.is_init = true;
    return true;
}

void moondream_api_state_cleanup(void) {
    printf("cleaning up API state...\n");
    moondream_lm_context_free(api_state.mctx);
    printf("freed moondream_lm_context\n");
    ggml_free(api_state.mmproj_model.ctx);
    printf("freed mmproj model ggml_context\n");
    ggml_free(api_state.model.ctx);
    printf("freed model ggml_context\n");
}

bool moondream_api_prompt(
    const char * prompt, std::string & response, 
    int n_max_gen, bool log_response_stream
) {
    moondream_lm & model = api_state.model;
    moondream_lm_hparams & hparams = model.hparams;
    moondream_lm_context & mctx = api_state.mctx;
    moondream_lm_cparams & cparams = mctx.cparams;
    moondream_mmproj_context & mmproj_ctx = api_state.mmproj_ctx;
    moondream_mmproj & mmproj = api_state.mmproj_model;

    moondream_lm_batch batch;
    if (!moondream_lm_batch_init(batch, cparams.n_ctx, model.hparams.n_embd, false)) {
        printf("failed to initialized moondream_lm_batch\n");
        return false;
    }

    int32_t prompt_len = 0;
    if (!size_to_int32(strlen(prompt), &prompt_len)) {
        printf("prompt was too big (length greater than max 32 bit integer\n");
        return false;
    }
    int32_t prompt_token_ids[prompt_len];
    int32_t n_prompt_tokens = moondream_tokenize(model.vocab, prompt, prompt_len, prompt_token_ids);
    if (n_prompt_tokens < 0) {
        printf("failed to tokenize prompt\n");
        return 1;
    }
    printf("n_prompt_tokens: %d\n", n_prompt_tokens);
    printf("prompt_token_ids: ");
    for (int i = 0; i < n_prompt_tokens; ++i) {
        printf("%d ", prompt_token_ids[i]);
    }
    printf("\n");

    if (log_response_stream) {
        printf("------------\n");
    }

#ifdef MOONDREAM_MULTI_MODAL
    const bool decode_success = moondream_lm_decode(
        mctx, model, batch, response,  
        n_prompt_tokens, prompt_token_ids,
        n_max_gen, log_response_stream, 
        mmproj_ctx.output_buffer, mmproj_ctx.n_patches, mmproj.hparams.n_proj
    );
#else
    // Don't pass any mmproj embeddings.
    const bool decode_success = moondream_lm_decode(
        mctx, model, batch, response,  
        n_prompt_tokens, prompt_token_ids,
        n_max_gen, log_response_stream, 
        nullptr, 0, 0
    );

#endif
    if (!decode_success) {
        printf("moondream decode failed\n");
        return false;
    }

    if (log_response_stream) {
        printf("------------\n");
    }
     
    moondream_lm_batch_free(batch);
    return true;
}

bool moondream_image_init(moondream_image & image, int n_xy, int n_positions) {
    assert(image.data == nullptr);
    image.n_xy = n_xy;
    image.n_channels = MOONDREAM_N_IMAGE_CHANNELS;
    image.n_scalars = image.n_xy * image.n_xy * image.n_channels;
    image.n_positions = n_positions;
    image.data = (float *)calloc(image.n_scalars, sizeof(float));
    if (!image.data) {
        printf("could not allocate memory for moondream_image data\n");
        return false;
    }
    image.pos = (int32_t *)malloc(sizeof(int32_t) * image.n_positions);
    if (!image.pos) {
        printf("could not allocate memory for moondream_image pos\n");
        return false;
    }
    for (int i = 0; i < image.n_positions; ++i) {
        image.pos[i] = i;
    }
    return true;
}

#ifndef MOONDREAM_LIBRARY_BUILD
bool moondream_image_load_and_set(const char * path, moondream_image & image) {
    assert(image.data != nullptr);
    assert(image.n_channels = MOONDREAM_N_IMAGE_CHANNELS);
    int base_width = -1, base_height = -1, base_channels = -1;
    unsigned char * base_stbi_data = stbi_load(
        path, &base_width, &base_height, &base_channels, MOONDREAM_N_IMAGE_CHANNELS
    );
    if (!base_stbi_data) {
        printf("stb could not load %s\n", path);
        return false;
    }
    
    assert(base_width > 0);
    assert(base_height > 0);
    int n_base_scalars = base_width * base_height * MOONDREAM_N_IMAGE_CHANNELS;
    float * base_float_data = (float *)malloc(sizeof(float) * n_base_scalars);
    if (!base_float_data) {
        printf("could not allocate memory for image floating point data\n");
        stbi_image_free(base_stbi_data);
        return false;
    }
    for (int i = 0; i < n_base_scalars; ++i) {
        base_float_data[i] = static_cast<float>(base_stbi_data[i]) / 255.0f;
    }
    stbi_image_free(base_stbi_data);

    if (base_width == image.n_xy && base_height == image.n_xy) {
        memcpy(image.data, base_float_data, sizeof(float) * n_base_scalars);
    } else {
        // Resize image from (base_width * base_height) to (n_xy * n_xy).
        float scale_x = static_cast<float>(base_width) / image.n_xy;
        float scale_y = static_cast<float>(base_height) / image.n_xy;

        for (int y = 0; y < image.n_xy; ++y) {
            for (int x = 0; x < image.n_xy; ++x) {
                int src_x = static_cast<int>(x * scale_x);
                int src_y = static_cast<int>(y * scale_y);

                // Clamp source coordinates.
                src_x = std::min(src_x, base_width - 1);
                src_y = std::min(src_y, base_height - 1);

                for (int c = 0; c < MOONDREAM_N_IMAGE_CHANNELS; ++c) {
                    int dst_idx = (y * image.n_xy + x) * MOONDREAM_N_IMAGE_CHANNELS + c;
                    int src_idx = (src_y * base_width + src_x) * MOONDREAM_N_IMAGE_CHANNELS + c;
                    image.data[dst_idx] = base_float_data[src_idx];
                }
            }
        }
    }
    free(base_float_data);
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
    
    // Initialize GGML.
    ggml_time_init();
    // Optional NUMA initialization for better performance on supported systems.
    /*enum ggml_numa_strategy numa_strat = GGML_NUMA_STRATEGY_DISTRIBUTE;
    if (ggml_is_numa()) {
        printf("numa node detected, initializing ggml numa\n");
        ggml_numa_init(numa_strat);
    }*/

    if (!moondream_api_state_init(text_model_path, mmproj_path, 8)) {
        printf("failed to initialize api state\n");
        return 1;
    }

#ifdef MOONDREAM_MULTI_MODAL
    const int image_size = api_state.mmproj_model.hparams.image_size;
    const int n_positions_mm = api_state.mmproj_ctx.n_positions;
    moondream_image image;
    if (!moondream_image_init(image, image_size, n_positions_mm)) {
        printf("failed to initialize moondream_image\n");
        return 1;
    }
    // Assuming the binary will be run from ../build/
    const char * image_path = "../../../assets/demo-1.jpg";
    if (!moondream_image_load_and_set(image_path, image)) {
        printf("failed to load and set moondream_image\n");
        return 1;
    }
    printf("succesfully loaded %s\n", image_path);
    
    if (!moondream_mmproj_embed(api_state.mmproj_ctx, api_state.mmproj_model, image)) {
        printf("failed to create image embeddings\n");
        return 1;
    }
    printf("succesfully created image embeddings\n");
    const float * image_embeddings = api_state.mmproj_ctx.output_buffer;
#endif

    const char * prompt = "<image>\n\nQuestion: Describe the image.\n\nAnswer:";
    std::string response = "";
    if (!moondream_api_prompt(prompt, response, 128, true)) {
        printf("prompt failed\n");
        return 1;
    }
    moondream_api_state_cleanup();
    return 0;
}
#endif // MOONDREAM_LIBRARY_BUILD
