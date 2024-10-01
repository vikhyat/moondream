#pragma once

#include <cstdint>
#include <vector>

#include "image.hpp"
#include "ggml.h"
#include "ggml-backend.h"

void test_bilinear_downsample(void);

enum projector_type {
    PROJECTOR_TYPE_MLP,
    PROJECTOR_TYPE_UNKNOWN,
};

struct moondream_mmproj_layer {
    // Attention.
    ggml_tensor * k_w = nullptr;
    ggml_tensor * k_b = nullptr;
    ggml_tensor * q_w = nullptr;
    ggml_tensor * q_b = nullptr;
    ggml_tensor * v_w = nullptr;
    ggml_tensor * v_b = nullptr;
    ggml_tensor * o_w = nullptr;
    ggml_tensor * o_b = nullptr;
    // Layernorm 1.
    ggml_tensor * ln_1_w = nullptr;
    ggml_tensor * ln_1_b = nullptr;
    // Feed forward.
    ggml_tensor * ff_i_w = nullptr;
    ggml_tensor * ff_i_b = nullptr;
    ggml_tensor * ff_o_w = nullptr;
    ggml_tensor * ff_o_b = nullptr;
    // Layernorm 2.
    ggml_tensor * ln_2_w = nullptr;
    ggml_tensor * ln_2_b = nullptr;
};

struct moondream_mmproj_hparams {
    int image_size;
    int patch_size;
    int n_embd;
    int n_ff;
    int n_proj;
    int n_head;
    int n_layer;
    float f_norm_eps;
    bool use_gelu;
    projector_type proj_type;
    float image_mean[3];
    float image_std[3];
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
    ggml_tensor * inp_patches = nullptr;
    ggml_tensor * positions = nullptr;
    // Memory buffers used to evaluate the model.
    std::vector<uint8_t> compute_buffer;
    ggml_backend_sched_t sched = nullptr;
    float * output_buffer = nullptr;
};

struct moondream_mmproj_batch {
    int n_batch; // 1 (full image) + n (image patches), n is in the range 0 to 4.
    int n_scalars; // height * width * channels * batch
    int image_side_length;
    float * patch_data = nullptr; // Images of the form HWCN.
    void * pos; // TODO: not implemented
    int n_positions; // TODO: not implemented
};

bool moondream_mmproj_context_init(
    moondream_mmproj_context & mctx,
    moondream_mmproj & model,
    int n_threads,
    bool normal_logs_enabled);
void moondream_mmproj_context_free(moondream_mmproj_context & mctx);
bool moondream_mmproj_load_from_file(
    const char * gguf_file_path, moondream_mmproj & model, bool normal_logs_enabled);
bool moondream_mmproj_embed(
    moondream_mmproj_context & mctx,
    moondream_mmproj & model,
    moondream_mmproj_batch & batch);

bool moondream_mmproj_load_image_to_batch(const char * img_path, moondream_mmproj_batch & batch);
bool moondream_mmproj_batch_save_to_pngs(moondream_mmproj_batch & batch);
bool moondream_mmproj_batch_init(moondream_mmproj_batch & batch);
void moondream_mmproj_batch_free(moondream_mmproj_batch & batch);
