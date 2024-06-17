#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>
#include "ggml/ggml.h"

#define MD_TEXT_MODEL_FNAME "moondream2-text-model-f16.gguf"
#define MD_MMPROJ_FNAME "moondream2-mmproj-f16.gguf"
#define DATA_PATH_MAX_LEN 512
#define ARCH_PREFIX(t) ("phi2." t)

struct moondream_layer {
    // normalization
    struct ggml_tensor * attn_norm;
    struct ggml_tensor * attn_norm_b;
    struct ggml_tensor * attn_norm_2;
    struct ggml_tensor * attn_norm_2_b;
    struct ggml_tensor * attn_q_norm;
    struct ggml_tensor * attn_q_norm_b;
    struct ggml_tensor * attn_k_norm;
    struct ggml_tensor * attn_k_norm_b;
    struct ggml_tensor * attn_out_norm;
    struct ggml_tensor * attn_out_norm_b;
    struct ggml_tensor * attn_q_a_norm;
    struct ggml_tensor * attn_kv_a_norm;

    // attention
    struct ggml_tensor * wq;
    struct ggml_tensor * wk;
    struct ggml_tensor * wv;
    struct ggml_tensor * wo;
    struct ggml_tensor * wqkv;
    struct ggml_tensor * wq_a;
    struct ggml_tensor * wq_b;
    struct ggml_tensor * wkv_a_mqa;
    struct ggml_tensor * wkv_b;

    // attention bias
    struct ggml_tensor * bq;
    struct ggml_tensor * bk;
    struct ggml_tensor * bv;
    struct ggml_tensor * bo;
    struct ggml_tensor * bqkv;

    // normalization
    struct ggml_tensor * ffn_norm;
    struct ggml_tensor * ffn_norm_b;
    struct ggml_tensor * layer_out_norm;
    struct ggml_tensor * layer_out_norm_b;
    struct ggml_tensor * ffn_norm_exps;

    // ff
    struct ggml_tensor * ffn_gate; // w1
    struct ggml_tensor * ffn_down; // w2
    struct ggml_tensor * ffn_up;   // w3

    // ff bias
    struct ggml_tensor * ffn_gate_b = nullptr;
    struct ggml_tensor * ffn_down_b = nullptr; // b2
    struct ggml_tensor * ffn_up_b = nullptr; // b3
    struct ggml_tensor * ffn_act;
};

struct moondream_hparams {
    int n_embd;
    int n_ff;
    int n_layer; // I think this is the same as n_block
    int n_rot;
    int n_ctx_train;
    int n_head;
    int n_head_kv;
    int n_embd_head_k;
    int n_embd_k_gqa;
    int n_embd_head_v;
    int n_embd_v_gqa;

    // this doesn't seem to be present in the model
    float rope_freq_base_train;
    int rope_attn_factor;
};

struct moondream_cparams {
    uint32_t n_ctx; // context size used during inference
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    uint32_t n_threads; // number of threads to use for generation
    uint32_t n_threads_batch; // number of threads to use for batch processing

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
    moondream_hparams hparams;
    moondream_cparams cparams;
    std::vector<moondream_layer> layers;
    ggml_context * ctx;
};

/* 
NOTE: llama.cpp has an llm_build_context struct which encapsulates all the cgraph build functions,
we probably don't need that but we will need some of the member variables.

Reference for convenience: 

struct llm_build_context {
    const llama_model    & model;
          llama_context  & lctx;
    const llama_hparams  & hparams;
    const llama_cparams  & cparams;
    const llama_batch    & batch;
    const llama_kv_cache & kv_self;

    const int64_t n_embd;
    const int64_t n_layer;
    const int64_t n_rot;
    const int64_t n_ctx;       // user-specified context size (can be different from n_ctx_train)
    const int64_t n_head;
    const int64_t n_head_kv;
    const int64_t n_embd_head_k;
    const int64_t n_embd_k_gqa;
    const int64_t n_embd_head_v;
    const int64_t n_embd_v_gqa;
    const int64_t n_expert;
    const int64_t n_expert_used;

    const float freq_base;
    const float freq_scale;
    const float ext_factor;
    const float attn_factor;
    const float beta_fast;
    const float beta_slow;
    const float norm_eps;
    const float norm_rms_eps;

    const int32_t n_tokens;
    const int32_t n_kv;     // size of KV cache to consider (n_kv <= kv_self.size)
    const int32_t n_outputs;
    const int32_t kv_head;  // index of where we store new KV data in the cache
    const int32_t n_ctx_orig;

    const bool flash_attn;

    const enum llama_pooling_type pooling_type;
    const enum llama_rope_type    rope_type;

    const llm_build_cb & cb;

    std::vector<uint8_t> & buf_compute_meta;

    struct ggml_context * ctx0 = nullptr;

    // TODO: consider making the entire interface noexcept
    llm_build_context(
        llama_context  & lctx,
    const llama_batch  & batch,
    const llm_build_cb & cb,
                  bool   worst_case) :
        model            (lctx.model),
        lctx             (lctx),
        hparams          (model.hparams),
        cparams          (lctx.cparams),
        batch            (batch),
        kv_self          (lctx.kv_self),
        n_embd           (hparams.n_embd),
        n_layer          (hparams.n_layer),
        n_rot            (hparams.n_rot),
        n_ctx            (cparams.n_ctx),
        n_head           (hparams.n_head),
        n_head_kv        (hparams.n_head_kv),
        n_embd_head_k    (hparams.n_embd_head_k),
        n_embd_k_gqa     (hparams.n_embd_k_gqa()),
        n_embd_head_v    (hparams.n_embd_head_v),
        n_embd_v_gqa     (hparams.n_embd_v_gqa()),
        n_expert         (hparams.n_expert),
        n_expert_used    (hparams.n_expert_used),
        freq_base        (cparams.rope_freq_base),
        freq_scale       (cparams.rope_freq_scale),
        ext_factor       (cparams.yarn_ext_factor),
        attn_factor      (cparams.yarn_attn_factor),
        beta_fast        (cparams.yarn_beta_fast),
        beta_slow        (cparams.yarn_beta_slow),
        norm_eps         (hparams.f_norm_eps),
        norm_rms_eps     (hparams.f_norm_rms_eps),
        n_tokens         (batch.n_tokens),
        n_kv             (worst_case ? kv_self.size : kv_self.n),
        n_outputs        (worst_case ? n_tokens : lctx.n_outputs),
        kv_head          (worst_case ? (kv_self.recurrent ? 0 : kv_self.size - n_tokens) : kv_self.head),
        n_ctx_orig       (cparams.n_ctx_orig_yarn),
        flash_attn       (cparams.flash_attn),
        pooling_type     (cparams.pooling_type),
        rope_type        (hparams.rope_type),
        cb               (cb),
        buf_compute_meta (lctx.buf_compute_meta) {
            // all initializations should be done in init()
        }
*/

// modification of llama.cpp build_phi2
// ref: https://github.com/ggerganov/llama.cpp/blob/da799b41891e34aac86ce4e173f9c4c0afd4fab3/llama.cpp
// currently wip, uncomment following line to get compiler errors
//#define MOONDREAM_BUILD_CGRAPH_WIP
#ifdef MOONDREAM_BUILD_CGRAPH_WIP
struct ggml_cgraph * build_phi2(ggml_context & ctx0, moondream_hparams & hparams, moondream_cparams & cparams) {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, LLAMA_MAX_NODES, false);

    const int64_t n_embd_head = hparams.n_embd_head_v;
    //const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();
    const int64_t n_embd_gqa = n_embd_head;
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

    struct ggml_tensor * cur;
    struct ggml_tensor * attn_norm_output;
    struct ggml_tensor * ffn_output;
    struct ggml_tensor * inpL;

    // TODO: implement llm_build_inp_embd (see llama.cpp line 6654)
    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // TODO: implement build_inp_pos (see llama.cpp line 7346)
    // inp_pos - contains the positions
    struct ggml_tensor * inp_pos = build_inp_pos();

    // TODO: implement build_inp_KQ_mask (see llama.cpp line 7371)
    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    for (int il = 0; il < n_layer; ++il) {
        // TODO: implement llm_build_norm (see llama.cpp line 6728)
        attn_norm_output = llm_build_norm(
            ctx0, inpL, hparams,
            model.layers[il].attn_norm,
            model.layers[il].attn_norm_b,
            LLM_NORM, cb, il
        );
        cb(attn_norm_output, "attn_norm", il);

        // self-attention
        {
            struct ggml_tensor * Qcur = nullptr;
            struct ggml_tensor * Kcur = nullptr;
            struct ggml_tensor * Vcur = nullptr;

            if (model.layers[il].wqkv) {
                cur = ggml_mul_mat(ctx0, model.layers[il].wqkv, attn_norm_output);
                cb(cur, "wqkv", il);

                cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                cb(cur, "bqkv", il);

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

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);

            Qcur = ggml_rope_ext(
                ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
            );
            cb(Qcur, "Qcur", il);

            // with phi2, we scale the Q to avoid precision issues
            // ref: https://github.com/ml-explore/mlx-examples/blob/08e862336ade809bc37d1035f94b359e7d1a5152/phi2/phi2.py#L64-L66
            Qcur = ggml_scale(ctx0, Qcur, 1.0f/sqrtf(float(n_embd_head)));
            cb(Qcur, "Qcur", il);

            Kcur = ggml_rope_ext(
                ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
            );
            cb(Kcur, "Kcur", il);

            cur = llm_build_kv(
                ctx0, model, hparams, cparams, kv_self, gf,
                model.layers[il].wo, model.layers[il].bo,
                Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f, cb, il
            );
        }

        if (il == n_layer - 1) {
            // TODO: implement build_inp_out_ids (see llama.cpp line 7464)
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            cur = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            attn_norm_output = ggml_get_rows(ctx0, attn_norm_output, inp_out_ids);
        }

        // FF
        {
            // TODO: implement llm_build_ffn (see llam.cpp line 6760)
            ffn_output = llm_build_ffn(
                ctx0, attn_norm_output,
                model.layers[il].ffn_up, model.layers[il].ffn_up_b,
                NULL, NULL,
                model.layers[il].ffn_down, model.layers[il].ffn_down_b,
                NULL,
                LLM_FFN_GELU, LLM_FFN_SEQ, cb, il
            );
            cb(ffn_output, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, ffn_output);
        cb(cur, "l_out", il);

        cur = ggml_add(ctx0, cur, inpL);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    // TODO: implement llm_build_norm (see llama.cpp line 6728)
    cur = llm_build_norm(
        ctx0, inpL, hparams,
        model.output_norm,
        model.output_norm_b,
        LLM_NORM, cb, -1
    );
    cb(cur, "result_norm", -1);

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output_no_bias", -1);

    cur = ggml_add(ctx0, cur, model.output_b);
    cb(cur, "result_output", -1);
    ggml_build_forward_expand(gf, cur);
    return gf;
}
#endif // MOONDREAM_BUILD_CGRAPH_WIP

/*
TODO: remove this later
REFERENCE: phi2 layer names from llama.cpp:

        LLM_ARCH_PHI3,
        {
            { LLM_TENSOR_TOKEN_EMBD,         "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,        "output_norm" },
            { LLM_TENSOR_OUTPUT,             "output" },
            { LLM_TENSOR_ROPE_FACTORS_LONG,  "rope_factors_long" },
            { LLM_TENSOR_ROPE_FACTORS_SHORT, "rope_factors_short" },
            { LLM_TENSOR_ATTN_NORM,          "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_QKV,           "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_Q,             "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,             "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,             "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,           "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,           "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_DOWN,           "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,             "blk.%d.ffn_up" },
*/
bool moondream_load_model(const char * gguf_file_path, moondream_model * model) {
    gguf_init_params init_params = {.no_alloc = true, .ctx = nullptr};
    gguf_context * ctx = gguf_init_from_file(gguf_file_path, init_params);
    int gguf_version = gguf_get_version(ctx);
    size_t gguf_alignment = gguf_get_alignment(ctx);
    size_t gguf_data_offset = gguf_get_data_offset(ctx);
    
    const char * model_arch = gguf_get_val_str(ctx, gguf_find_key(ctx, "general.architecture"));
    
    moondream_hparams hparams;
    const char * model_name = gguf_get_val_str(ctx, gguf_find_key(ctx, "general.name"));
    hparams.n_ctx_train = gguf_get_val_u32(ctx, gguf_find_key(ctx, ARCH_PREFIX("context_length")));
    hparams.n_embd = gguf_get_val_u32(ctx, gguf_find_key(ctx, ARCH_PREFIX("embedding_length")));
    hparams.n_rot = gguf_get_val_u32(ctx, gguf_find_key(ctx, ARCH_PREFIX("rope.dimension_count")));
    hparams.n_layer = gguf_get_val_u32(ctx, gguf_find_key(ctx, ARCH_PREFIX("block_count")));
    hparams.n_ff = gguf_get_val_u32(ctx, gguf_find_key(ctx, ARCH_PREFIX("block_count")));
    hparams.n_head = gguf_get_val_u32(ctx, gguf_find_key(ctx, ARCH_PREFIX("attention.head_count")));
    hparams.n_head_kv = gguf_get_val_u32(ctx, gguf_find_key(ctx, ARCH_PREFIX("attention.head_count_kv")));
    
    // n_head_k and n_head_v are not specified, so calculate them according to the gguf documentation
    // "If not specified, it will be `n_embd / n_head`"
    // TODO: remove these commented lines later, just keeping them as a reference for now
    //hparams.n_embd_head_k = gguf_get_val_u32(ctx, gguf_find_key(ctx, ARCH_PREFIX("attention.value_length")));
    //hparams.n_embd_head_k = gguf_get_val_u32(ctx, gguf_find_key(ctx, ARCH_PREFIX("attention.key_length")));
    hparams.n_embd_head_k = hparams.n_embd / hparams.n_head;
    hparams.n_embd_head_v = hparams.n_embd_head_k;

    printf("loaded %s from %s\n", model_name, gguf_file_path);
    printf("gguf version: %d\n", gguf_version);
    printf("gguf alignment: %ld\n", gguf_alignment);
    printf("gguf data offset: %ld\n", gguf_data_offset);
    printf("model architecture: %s\n", model_arch);
    printf("context length: %d\n", hparams.n_ctx_train);
    printf("embedding length: %d\n", hparams.n_embd);
    printf("block count: %d\n", hparams.n_layer);
    printf("feed forward length: %d\n", hparams.n_ff);
    printf("head count: %d\n", hparams.n_head);
    printf("head count kv: %d\n", hparams.n_head_kv);
    printf("n_embd_head_k: %d\n", hparams.n_embd_head_k);
    printf("n_embd_head_v: %d\n", hparams.n_embd_head_v);
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

    // resolve text model file path
    const char * text_model_fname = MD_TEXT_MODEL_FNAME;
    const size_t text_model_fname_length = strlen(text_model_fname);
    // add 1 to give space for null-terminator in concatenated string
    const size_t text_model_path_length = data_path_length + text_model_fname_length + 1;
    char text_model_path[text_model_path_length];
    snprintf(text_model_path, text_model_path_length, "%s%s", data_path, text_model_fname); 

    // resolve mmproj file path
    const char * mmproj_fname = MD_MMPROJ_FNAME;
    const size_t mmproj_fname_length = strlen(mmproj_fname);
    // add 1 to give space for null-terminator in concatenated string
    const size_t mmproj_path_length = data_path_length + text_model_fname_length + 1;
    char mmproj_path[text_model_path_length];
    snprintf(mmproj_path, mmproj_path_length, "%s%s", data_path, mmproj_fname); 

    printf("text model path: %s\n", text_model_path);
    printf("mmproj path: %s\n", mmproj_path);
    
    moondream_model model;
    bool result = moondream_load_model(text_model_path, &model);
    if (result == false) {
        printf("could not load model\n");
    } else {
        printf("succesfully loaded model\n");
    }
    return 0;
}
