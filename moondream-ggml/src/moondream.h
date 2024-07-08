#include <cstdint>
#include <vector>
#include <unordered_map>
#include <map>
#include "ggml/ggml.h"
#include "ggml/ggml-backend.h"

enum projector_type {
    PROJECTOR_TYPE_MLP,
    PROJECTOR_TYPE_MLP_NORM,
    PROJECTOR_TYPE_LDP,
    PROJECTOR_TYPE_LDPV2,
    PROJECTOR_TYPE_UNKNOWN,
};

/* Start of llm enums. */
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
/* End of llm enums. */

struct moondream_layer {
    // Normalization
    ggml_tensor * attn_norm = nullptr;
    ggml_tensor * attn_norm_b = nullptr;
    ggml_tensor * attn_norm_2 = nullptr;
    ggml_tensor * attn_norm_2_b = nullptr;
    ggml_tensor * attn_q_norm = nullptr;
    ggml_tensor * attn_q_norm_b = nullptr;
    ggml_tensor * attn_k_norm = nullptr;
    ggml_tensor * attn_k_norm_b = nullptr;
    ggml_tensor * attn_out_norm = nullptr;
    ggml_tensor * attn_out_norm_b = nullptr;
    ggml_tensor * attn_q_a_norm = nullptr;
    ggml_tensor * attn_kv_a_norm = nullptr;
    // Attention
    ggml_tensor * wq = nullptr;
    ggml_tensor * wk = nullptr;
    ggml_tensor * wv = nullptr;
    ggml_tensor * wo = nullptr;
    ggml_tensor * wqkv = nullptr;
    ggml_tensor * wq_a = nullptr;
    ggml_tensor * wq_b = nullptr;
    ggml_tensor * wkv_a_mqa = nullptr;
    ggml_tensor * wkv_b = nullptr;
    // Attention bias
    ggml_tensor * bq = nullptr;
    ggml_tensor * bk = nullptr;
    ggml_tensor * bv = nullptr;
    ggml_tensor * bo = nullptr;
    ggml_tensor * bqkv = nullptr;
    // Normalization
    ggml_tensor * ffn_norm = nullptr;
    ggml_tensor * ffn_norm_b = nullptr;
    ggml_tensor * layer_out_norm = nullptr;
    ggml_tensor * layer_out_norm_b = nullptr;
    ggml_tensor * ffn_norm_exps = nullptr;
    // Feed forward
    ggml_tensor * ffn_gate = nullptr; // w1
    ggml_tensor * ffn_down = nullptr;  // w2
    ggml_tensor * ffn_up = nullptr;   // w3
    // Feed forward bias
    ggml_tensor * ffn_gate_b = nullptr;
    ggml_tensor * ffn_down_b = nullptr; // b2
    ggml_tensor * ffn_up_b = nullptr;   // b3
    ggml_tensor * ffn_act = nullptr;
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

struct moondream_model {
    ggml_context * ctx = nullptr;
    moondream_hparams hparams;
    moondream_vocab vocab;
    std::vector<moondream_layer> layers;
    ggml_tensor * tok_embd = nullptr;
    ggml_tensor * output_norm = nullptr;
    ggml_tensor * output_norm_b = nullptr;
    ggml_tensor * output = nullptr;
    ggml_tensor * output_b = nullptr;
};

struct moondream_mmproj_model {
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
struct moondream_batch {
    int32_t n_tokens;
    // The token ids of the input (used when embd is NULL).
    int32_t * token = nullptr;
    // The token embeddings (used when token is NULL).
    float * embd = nullptr;
    // The positions of the respective tokens in the sequence.
    int32_t * pos = nullptr;
    // The sequence to which the respective token belongs.
    int32_t ** seq_id = nullptr;
    // If zero, the logits for the respective token will not be output.
    int8_t * logits = nullptr;
};

// Batch for mmproj/clip input.
struct moondream_mmproj_batch{
    float * images = nullptr;
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

struct moondream_context {
    ggml_context * ctx = nullptr;
    moondream_cparams cparams;
    moondream_kv_cache kv_cache;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_buffer_type_t backend_cpu_buft = nullptr;

    int n_outputs = 0;
     // Number of tokens sampled.
    int32_t n_sample = 0;
    // Input tensors
    ggml_tensor * inp_tokens = nullptr;    // I32 [n_batch]
    ggml_tensor * inp_embd = nullptr;      // F32 [n_embd, n_batch]
    ggml_tensor * inp_pos = nullptr;       // I32 [n_batch]
    ggml_tensor * inp_out_ids = nullptr;   // I32 [n_outputs]
    ggml_tensor * inp_KQ_mask = nullptr;   // F32 [kv_size, n_batch]
    ggml_tensor * inp_K_shift = nullptr;   // I32 [kv_size]
    ggml_tensor * inp_mean = nullptr;      // F32 [n_batch, n_batch]
    ggml_tensor * inp_cls = nullptr;       // I32 [n_batch]
    ggml_tensor * inp_s_copy = nullptr;    // I32 [kv_size]
    ggml_tensor * inp_s_mask = nullptr;    // F32 [1, n_kv]
    ggml_tensor * inp_s_seq = nullptr;     // I32 [n_kv, n_batch]
    // Memory buffers used to evaluate the model.
    std::vector<uint8_t> compute_buffer;
    ggml_backend_sched_t sched = nullptr;
};

struct moondream_mmproj_context {
    ggml_context * ctx = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_buffer_type_t backend_cpu_buft = nullptr;
    std::vector<uint8_t> compute_buffer;
    ggml_backend_sched_t sched = nullptr;
};
