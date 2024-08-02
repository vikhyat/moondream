#include <cassert>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <limits>

#include "stb_image.h"
#include "stb_image_write.h"
#include "helpers.hpp"
#include "mmproj.hpp"

// Modification of llama.cpp/examples/llava/clip.pp clip_image_build_graph.
// Ref: https://github.com/ggerganov/llama.cpp/blob/da799b41891e34aac86ce4e173f9c4c0afd4fab3/examples/llava/clip.cpp
static ggml_cgraph * mmproj_build_clip(
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
        ggml_tensor * q = ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].q_w, cur), model.layers[il].q_b);
        q = ggml_scale_inplace(ctx0, q, 1.0f / sqrtf((float)n_head_qkv));
        q = ggml_reshape_4d(ctx0, q, n_head_qkv, n_head, num_positions, batch_size);
        q = ggml_cont(ctx0, ggml_permute(ctx0, q, 0, 2, 1, 3));
        q = ggml_reshape_3d(ctx0, q, n_head_qkv, num_positions, n_head * batch_size);

        ggml_tensor * k = ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].k_w, cur), model.layers[il].k_b);
        k = ggml_reshape_4d(ctx0, k, n_head_qkv, n_head, num_positions, batch_size);
        k = ggml_cont(ctx0, ggml_permute(ctx0, k, 0, 2, 1, 3));
        k = ggml_reshape_3d(ctx0, k, n_head_qkv, num_positions, n_head * batch_size);

        ggml_tensor * v = ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].v_w, cur), model.layers[il].v_b);
        v = ggml_reshape_4d(ctx0, v, n_head_qkv, n_head, num_positions, batch_size);
        v = ggml_cont(ctx0, ggml_permute(ctx0, v, 1, 2, 0, 3));
        v = ggml_reshape_3d(ctx0, v, num_positions, n_head_qkv, n_head * batch_size);

        ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
        kq = ggml_soft_max_inplace(ctx0, kq);
        ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
        kqv = ggml_reshape_4d(ctx0, kqv, n_head_qkv, num_positions, n_head, batch_size);
        kqv = ggml_permute(ctx0, kqv, 0, 2, 1, 3);

        cur = ggml_cont_3d(ctx0, kqv, n_embd, num_positions, batch_size);
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
    mctx.inp_patches = patches;
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
    ggml_cgraph * gf = mmproj_build_clip(model, mctx);
    if (!gf) {
        printf("failed to build mmproj compute graph\n");
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

void moondream_mmproj_context_free(moondream_mmproj_context & mctx) {
    if (mctx.backend_cpu) {
        ggml_backend_free(mctx.backend_cpu);
        mctx.backend_cpu = nullptr;
    }
    if (mctx.output_buffer) {
        free(mctx.output_buffer);
        mctx.output_buffer = nullptr;
    }
    if (mctx.sched) {
        ggml_backend_sched_free(mctx.sched);
        mctx.sched = nullptr;
    }
    if (mctx.ctx) {
        ggml_free(mctx.ctx);
        mctx.ctx = nullptr;
    }
}

bool moondream_mmproj_load_from_file(const char * gguf_file_path, moondream_mmproj & model) {
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
    memcpy(hparams.image_mean, gguf_get_arr_data(meta, image_mean_key_id), sizeof(float) * 3);

    const int image_std_key_id = gguf_find_key(meta, "clip.vision.image_std");
    const int n_image_std = gguf_get_arr_n(meta, image_std_key_id);
    if (n_image_std != 3) {
        printf("expected n_image_std = 3 but got n_image_std = %d\n", n_image_std);
        return false;
    }
    memcpy(hparams.image_std, gguf_get_arr_data(meta, image_std_key_id), sizeof(float) * 3);
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

bool moondream_mmproj_embed(
    moondream_mmproj_context & mctx,
    moondream_mmproj & model,
    moondream_image & image
) {
    ggml_cgraph * gf = mmproj_build_clip(model, mctx);
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

void moondream_image_free(moondream_image & image) {
    if (image.data) {
        free(image.data);
        image.data = nullptr;
    }
    if (image.pos) {
        free(image.pos);
        image.pos = nullptr;
    }
}

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

bool moondream_mmproj_batch_init(moondream_mmproj_batch & batch) {
    constexpr int n_elements_per_patch =
        MOONDREAM_IMAGE_PATCH_SIDE_LENGTH * MOONDREAM_IMAGE_PATCH_SIDE_LENGTH * MOONDREAM_N_IMAGE_CHANNELS;
    constexpr int n_elements_total = (1 + MOONDREAM_MAX_IMAGE_PATCHES) * n_elements_per_patch;
    batch.patches = (float *)malloc(sizeof(float) * n_elements_total);
    if (!batch.patches) {
        printf("failed to allocate memory for moondream_mmproj_batch.patches\n");
        return false;
    }
    return true;
}

void moondream_mmproj_batch_free(moondream_mmproj_batch & batch) {
    if (batch.patches) {
        free(batch.patches);
        batch.patches = nullptr;
    }
}

void moondream_mmproj_image_preprocess(moondream_image_alt & image, moondream_mmproj_batch & batch) {
    int target_width, target_height;
    static const int n_supported_sizes = 4;
    static const int supported_sizes[][2] = {{378, 378}, {378, 756}, {756, 378}, {756, 756}};
    const int max_dim = image.width > image.height ? image.width : image.height;
    if (max_dim < 512) {
        target_width = 378;
        target_height = 378;
    } else {
        float min_aspect_ratio_diff = std::numeric_limits<float>::max();
        int min_side_length_diff = std::numeric_limits<int>::max();
        const float aspect_ratio = (float)image.width / (float)image.height;
        for (int i = 0; i < n_supported_sizes; ++i) {
            const int cur_width = supported_sizes[i][0];
            const int cur_height = supported_sizes[i][1];
            const float cur_aspect_ratio = (float)cur_width / (float)cur_height;
            const float cur_aspect_ratio_diff = fabsf(cur_aspect_ratio - aspect_ratio);
            const int cur_side_length_diff = abs(cur_width - image.width) + abs(cur_height - image.height);
            // TODO: figure out how python resolves the two values in the min key and implement it here.
            // Ref:
            /*
                im_size = min(
                    self.supported_sizes,
                    key=lambda size: (
                        abs((size[1] / size[0]) - aspect_ratio),
                        abs(size[0] - width) + abs(size[1] - height),
                    ),
                )
            */
            if (cur_side_length_diff < min_side_length_diff) {
                min_side_length_diff = cur_side_length_diff;
                target_width = cur_width;
                target_height = cur_height;
            }
        }
    }

    // TODO:
    // - Resize image to (target_width, target_height)
    // - Normalize image
    // - Split image into patches
    // - Resize image to (378, 378) if it isn't already that size
    // - Combined resized image with patches
    // - Copy combined image/patches into the batch buffer
}

struct mmproj_image_downsample_buffer {
    int n_channels = 0;
    int n_data_elements = 0;
    float * inter_image = nullptr;
    float * final_image = nullptr;
};

bool mmproj_image_downsample_buffer_init(mmproj_image_downsample_buffer & buf, int n_channels) {
    buf.n_channels = n_channels;
    constexpr int n_spatial_elements = MOONDREAM_MAX_IMAGE_SIDE_LENGTH * MOONDREAM_MAX_IMAGE_SIDE_LENGTH;
    buf.n_data_elements = n_spatial_elements * buf.n_channels;
    buf.inter_image = (float *)calloc(buf.n_data_elements, sizeof(float));
    if (!buf.inter_image) {
        printf("failed to allocate memory for mmproj_image_downsample_buffer.inter_image\n");
        return false;
    }
    buf.final_image = (float *)calloc(buf.n_data_elements, sizeof(float));
    if (!buf.final_image) {
        printf("failed to allocate memory for mmproj_image_downsample_buffer.final_image\n");
        return false;
    }
    return true;
}

void mmproj_image_downsample_buffer_free(mmproj_image_downsample_buffer & buf) {
    if (buf.inter_image) {
        free(buf.inter_image);
        buf.inter_image = nullptr;
    }
    if (buf.final_image) {
        free(buf.final_image);
        buf.final_image = nullptr;
    }
}

void mmproj_bilinear_downsample(
    const float * base_image,
    mmproj_image_downsample_buffer & buf,
    int base_width, int base_height,
    int new_width, int new_height,
    int channels
) {
    printf("start downsample\n");
    printf("channels: %d\n", channels);
    printf("base_width: %d, base_height: %d\n", base_width, base_height);
    printf("new_width: %d, new_height: %d\n", new_width, new_height);

    const int x_gaps = new_width - 1;
    const float x_gap_size = (float)base_width / (float)(x_gaps);

    const int y_gaps = new_height - 1;
    const float y_gap_size = (float)base_height / (float)(y_gaps);

    printf("x_gaps: %d, x_gap_size: %f\n", x_gaps, x_gap_size);
    printf("y_gaps: %d, y_gap_size: %f\n", y_gaps, y_gap_size);

    // TODO: not quite right. Base values need to be weighted based on distance from the sample point.

    for (int x = 0; x < x_gaps; ++x) {
        const int base_left_col_offset = (int)floorf(x * x_gap_size);
        for (int y = 0; y < base_height; ++y) {
            const int base_left_offset = (y * base_width + base_left_col_offset) * channels;
            const int base_right_offset = base_left_offset + channels;
            const int inter_offset = (y * new_width + x) * channels;
            for (int k = 0; k < channels; ++k) {
                const float base_left_val = base_image[base_left_offset + k];
                const float base_right_val = base_image[base_right_offset + k];
                buf.inter_image[inter_offset + k] = (base_left_val + base_right_val) / 2.0f;
            }
        }
    }

    for (int y = 0; y < y_gaps; ++y) {
        const int inter_up_row_offset = (int)floorf(y * y_gap_size) * new_width;
        const int inter_down_row_offset = inter_up_row_offset + new_width;
        for (int x = 0; x < new_width; ++x) {
            const int inter_up_offset = (inter_up_row_offset + x) * channels;
            const int inter_down_offset = (inter_up_row_offset + x) * channels;
            const int new_offset = (y * new_width + x) * channels;
            for (int k = 0; k < channels; ++k) {
                const float inter_up_val = buf.inter_image[inter_up_offset + k];
                const float inter_down_val = buf.inter_image[inter_down_offset + k];
                buf.final_image[new_offset + k] = (inter_up_val + inter_down_val) / 2.0f;
            }
        }
    }
}

void test_bilinear_downsample(void) {
    int base_width = -1, base_height = -1, base_channels = -1;
    unsigned char * base_stbi_data = stbi_load(
        "../../../assets/demo-1.jpg",
        &base_width, &base_height, &base_channels,
        MOONDREAM_N_IMAGE_CHANNELS
    );
    if (!base_stbi_data) {
        printf("stb could not load image\n");
        return;
    }
    base_channels = MOONDREAM_N_IMAGE_CHANNELS;

    mmproj_image_downsample_buffer buf;
    mmproj_image_downsample_buffer_init(buf, base_channels);

    int new_width = 378;
    int new_height = 378;

    int image_elements = base_width * base_height * base_channels;
    size_t image_size_bytes = sizeof(float) * image_elements;
    float * base_image = (float *)malloc(image_size_bytes);
    for (int i = 0; i < image_elements; ++i) {
        const uint8_t int_element = static_cast<uint8_t>(base_stbi_data[i]);
        base_image[i] = static_cast<float>(int_element) / 255.0f;
    }
    stbi_image_free(base_stbi_data);

    mmproj_bilinear_downsample(
        base_image, buf,
        base_width, base_height,
        new_width, new_height,
        base_channels
    );
    float * new_image = buf.final_image;

    const int new_image_elements = new_width * new_height * base_channels;
    uint8_t * pre_save_image = (uint8_t *)malloc(sizeof(uint8_t) * new_image_elements);
    for (int i = 0; i < new_image_elements; ++i) {
        pre_save_image[i] = static_cast<uint8_t>(new_image[i] * 255.0f);
    }
    stbi_write_png(
        "../../../data/bilinear_downsample_preview.png",
        new_width, new_height, base_channels,
        pre_save_image,
        new_width * base_channels
    );
    free(pre_save_image);
    free(base_image);
    mmproj_image_downsample_buffer_free(buf);
}
