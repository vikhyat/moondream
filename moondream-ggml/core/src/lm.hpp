#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <map>
#include <unordered_map>

#include "ggml.h"
#include "ggml-backend.h"

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

struct moondream_lm_hparams {
    int n_embd;
    int n_ff;
    int n_layer;
    int n_rot;
    int n_ctx_train;
    int n_head;
    int n_head_kv;
    int n_embd_head_k;
    int n_embd_head_v;
    int n_vocab;

    float f_norm_eps;
    float f_norm_rms_eps;

    float f_max_alibi_bias;
};

struct moondream_lm_cparams {
    // Context size used during inference.
    int n_ctx;
    int n_batch;
    int n_ubatch;
    int n_seq_max;
    // Number of threads to use for generation.
    int n_threads;
    // Number of threads to use for batch processing.
    int n_threads_batch;

    float rope_freq_base;
    float rope_freq_scale;

    int n_ctx_orig_yarn;
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
    int32_t bos_token_id;
    int32_t eos_token_id;
    int32_t unknown_token_id;
    int32_t separator_token_id;
    int32_t padding_token_id;
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

// Arrays must have size of n_tokens
struct moondream_lm_batch {
    int n_tokens_alloc;
    int n_tokens;
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
    int n_ctx_active = 0;
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

bool moondream_lm_batch_init(
    moondream_lm_batch & batch,
    int n_tokens_alloc,
    int n_embd,
    bool alloc_embd
);
void moondream_lm_batch_free(moondream_lm_batch & batch);
bool moondream_kv_cache_init(
    moondream_kv_cache & kv_cache,
    moondream_lm_hparams & hparams,
    moondream_lm_cparams & cparams,
    ggml_backend_t backend,
    ggml_type type_k,
    ggml_type type_v
);
bool moondream_lm_context_init(
    moondream_lm_context & mctx,
    moondream_lm_hparams & hparams,
    moondream_lm_cparams & cparams,
    moondream_lm & model,
    ggml_type type_k,
    ggml_type type_v,
    bool normal_logs_enabled
);
void moondream_lm_context_free(moondream_lm_context & mctx);
int32_t moondream_lm_tokenize(
    moondream_vocab & vocab,
    const char * text,
    int text_len,
    int32_t * token_ids_output
);
bool moondream_lm_load_from_file(
    const char * gguf_file_path, moondream_lm & model, bool normal_logs_enabled
);
bool moondream_lm_decode(
    moondream_lm_context & mctx,
    moondream_lm & model,
    moondream_lm_batch & batch,
    std::string & response,
    int n_prompt_tokens,
    int32_t * prompt_token_ids,
    int n_max_gen,
    bool log_response_stream,
    float * mmproj_embd,
    int n_embd,
    int embd_dim
);
