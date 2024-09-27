#include <cassert>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <limits>

#include "stb_image.h"
#include "stb_image_write.h"
#include "stb_image_resize2.h"
#include "helpers.hpp"
#include "mmproj.hpp"
#include "patch.hpp"

ggml_tensor * tensor_split_3d(ggml_context * ctx, ggml_tensor * input, size_t start_idx, size_t end_idx) {
    assert(input->type == GGML_TYPE_F32);

    size_t D1 = input->ne[0];
    size_t D2 = input->ne[1];
    size_t D3 = input->ne[2];

    size_t slice_size = end_idx - start_idx; // Size of the slice along the 3rd dimension

    // Create a new tensor for the slice with dimensions [D1, D2, slice_size]
    ggml_tensor * sliced_tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D1, D2, slice_size);

    // Manually copy the slice
    float * dest = (float *)sliced_tensor->data;
    float * src = (float *)input->data;

    for (size_t i = 0; i < D1; ++i) {
        for (size_t j = 0; j < D2; ++j) {
            // Can't memcpy here because input and sliced_tensor don't have memory allocated for data.
            // Need to use a ggml copy function to build data movement into the graph.
            std::memcpy(
                &dest[(i * D2 + j) * slice_size],
                &src[(i * D2 + j) * D3 + start_idx],
                slice_size * sizeof(float) // Assuming float32 type
            );
        }
    }
    return sliced_tensor;
}

// Modification of llama.cpp/examples/llava/clip.pp clip_image_build_graph.
// Ref: https://github.com/ggerganov/llama.cpp/blob/da799b41891e34aac86ce4e173f9c4   c0afd4fab3/examples/llava/clip.cpp
static ggml_cgraph * mmproj_build_clip(
    moondream_mmproj & model,
    moondream_mmproj_context & mctx
) {
    moondream_mmproj_hparams & hparams = model.hparams;
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

    constexpr int batch_size = 1;

    ggml_init_params build_ctx_params = {
        mctx.compute_buffer.size(),
        mctx.compute_buffer.data(),
        true};
    ggml_context * ctx0 = ggml_init(build_ctx_params);
    ggml_cgraph * gf = ggml_new_graph(ctx0);

    // Shape should be (378, 378, 3, num_preprocess_patches + 1)
    ggml_tensor * inp_raw = ggml_new_tensor_4d(
        ctx0, GGML_TYPE_F32, image_size, image_size, MOONDREAM_N_IMAGE_CHANNELS, batch_size);
    ggml_set_name(inp_raw, "inp_raw");
    ggml_set_input(inp_raw);
    mctx.inp_raw = inp_raw;

    // Reference python code
    //   def __init__(self):
    //       super().__init__()
    //       self.linear = nn.Linear(588, 1152)
    //   def forward(self, x):
    //       b, c, hp1, wp2 = x.shape
    //       p1, p2 = 14, 14
    //       h, w = hp1 // p1, wp2 // p2
    //       x = x.reshape(b, c, h, p1, w, p2)
    //       x = x.permute(0, 2, 4, 1, 3, 5)
    //       x = x.reshape(b, h * w, c * p1 * p2)
    //       return self.linear(x)

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
        if (hparams.use_gelu)
        {
            cur = ggml_gelu_inplace(ctx0, cur);
        }
        else
        {
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

    // NOTE: commented this out because tensor_split_3d() causes segfault on graph build.
    /*ggml_tensor *full_img_features = tensor_split_3d(ctx0, embeddings, 0, 1);
    ggml_tensor *patch_features = nullptr;
    if (num_patches > 0)
    {
        patch_features = tensor_split_3d(ctx0, embeddings, 1, num_patches);
    }*/

    // TODO: merge patch features and concatenate with image features.
    // The concatenated features will then be passed as input to the projector.
    // REF:
    /*

        combined_features = self.encoder(combined_images)

        full_img_features = combined_features[: len(im_list)]
        patch_features = (
            combined_features[len(im_list) :].transpose(1, 2).view(-1, 1152, 27, 27)
        )

        reshaped_patch_features = []
        patch_idx = 0
        for i, patch_set in enumerate(patches): ### THIS IS FOR BATCHES - IGNORE FOR HERE
            if len(patch_set) == 0:
                reshaped_patch_features.append(
                    full_img_features[i].transpose(0, 1).view(1152, 27, 27)
                )
            else:
                sample_features = []
                for row_patches in patch_set:
                    row_len = len(row_patches)
                    row_features = patch_features[
                        patch_idx : patch_idx + row_len
                    ]  # row_len, T, C
                    row_features = torch.cat(
                        list(row_features), dim=2
                    )  # T, C * row_len
                    patch_idx += row_len
                    sample_features.append(row_features)
                sample_features = torch.cat(sample_features, dim=1)
                sample_features = F.interpolate( ### CHANGED TO ADAPTIVE AVG POOL
                    sample_features.unsqueeze(0), size=(27, 27), mode="bilinear"
                ).squeeze(0)
                reshaped_patch_features.append(sample_features)
        reshaped_patch_features = (
            torch.stack(reshaped_patch_features).view(-1, 1152, 729).transpose(1, 2)
        )

        final_features = torch.cat([full_img_features, reshaped_patch_features], dim=2)
    */
    //
    // JPA TODO: implement custom GGML function to merge/reassemble patch embeddings from original image
    //
    // 3x729x1152 tensor (729 => 27x27). 27 = sqrt(729)
    //
    // tensor is either 1 dim, 3 dim or 5 dim.
    // 1 dim tensor => just use that
    // 3 dim tensor => 1st is image, next two are either a row or a col
    // 5 dim tensor => 1st is image, the next 4 are topLeft, topRight, botLeft, BotRight patches
    //
    // create new tensor made up of:
    // 1. The first patch (which is the of the entire image)
    // 2. The adaptive average pool of the rest of the patches
    //
    // shape will be (27x27)x(1152+2), or 729x2304

    embeddings = ggml_reshape_2d(ctx0, embeddings, embeddings->ne[0], embeddings->ne[1]);

    // NOTE: the `patches` tensor can be used to select the patch features corresponding
    // to an image feature, but this requires the input to be batched with the image and patches
    // concatenated along the batch dimension.
    ggml_tensor *patches = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, num_patches);
    ggml_set_name(patches, "patches");
    ggml_set_input(patches);
    // TODO: this is an input tensor so its values have to be set after sched alloc,
    // but there's no point in doing that until batching and image/patch feature
    // merging is implemented because the output will still likely be nonsense to the LM.
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
    int n_threads,
    bool normal_logs_enabled
) {
    const moondream_mmproj_hparams &hparams = model.hparams;
    mctx.n_patches_per_side = hparams.image_size / hparams.patch_size;
    mctx.n_patches = mctx.n_patches_per_side * mctx.n_patches_per_side;
    // NOTE: n_positions should be (mctx.n_patches + 1) if there is a class embedding.
    mctx.n_positions = mctx.n_patches;
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
    ggml_backend_cpu_set_n_threads(mctx.backend_cpu, n_threads);
    mctx.backend_cpu_buft = ggml_backend_get_default_buffer_type(mctx.backend_cpu);
    // TODO: figure out a way to dynamically determine the number of required nodes because LLAMA_MAX_NODES
    // is probably overkill for the clip graph.
    const size_t compute_buf_size =
        ggml_tensor_overhead() * LLAMA_MAX_NODES + ggml_graph_overhead_custom(LLAMA_MAX_NODES, false);
    if (normal_logs_enabled) {
        const double compute_buf_size_gib = bytes_to_gib(compute_buf_size);
        printf("new mmproj compute_buf_size is %zu B, %lf GiB\n", compute_buf_size, compute_buf_size_gib);
    }
    mctx.compute_buffer.resize(compute_buf_size);

    // Initialize scheduler.
    ggml_cgraph * gf = mmproj_build_clip(model, mctx);
    if (!gf) {
        printf("failed to build mmproj compute graph\n");
        return false;
    }
    if (normal_logs_enabled) {
        printf("n_nodes: %d\n", gf->n_nodes);
        printf("built mmproj graph\n");
    }
    mctx.sched = ggml_backend_sched_new(&mctx.backend_cpu, &mctx.backend_cpu_buft, 1, gf->n_nodes, false);
    if (!ggml_backend_sched_reserve(mctx.sched, gf)) {
        printf("failed to reserve buffers for mmproj compute graph\n");
        return false;
    }
    return true;
}

void moondream_mmproj_context_free(moondream_mmproj_context &mctx)
{
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

bool moondream_mmproj_load_from_file(
    const char * gguf_file_path, moondream_mmproj & model, bool normal_logs_enabled
) {
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
    hparams.image_size = (int)gguf_get_val_u32(meta, gguf_find_key(meta, "clip.vision.image_size"));
    hparams.patch_size = (int)gguf_get_val_u32(meta, gguf_find_key(meta, "clip.vision.patch_size"));
    hparams.n_embd = (int)gguf_get_val_u32(meta, gguf_find_key(meta, "clip.vision.embedding_length"));
    hparams.n_ff = (int)gguf_get_val_u32(meta, gguf_find_key(meta, "clip.vision.feed_forward_length"));
    hparams.n_proj = (int)gguf_get_val_u32(meta, gguf_find_key(meta, "clip.vision.projection_dim"));
    hparams.n_head = (int)gguf_get_val_u32(meta, gguf_find_key(meta, "clip.vision.attention.head_count"));
    hparams.n_layer = (int)gguf_get_val_u32(meta, gguf_find_key(meta, "clip.vision.block_count"));
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
    if (n_image_mean != 3) {
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
        switch (i)
        {
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
            case 4: // v.position_embd.weight
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

    if (normal_logs_enabled) {
        printf("------------\nloaded %s from %s\n", model_name, gguf_file_path);
        printf("gguf_version: %d\n", gguf_version);
        printf("gguf_alignment: %zu\n", gguf_alignment);
        printf("gguf_data_offset: %zu\n", gguf_data_offset);
        printf("model_arch: %s\n", model_arch);
        printf("mem_size: %lf GiB\n", bytes_to_gib(ggml_get_mem_size(model.ctx)));
        printf("------------\nMMPROJ Hyperparameters\n------------\n");
        printf("image_size: %d\n", hparams.image_size);
        printf("patch_size: %d\n", hparams.patch_size);
        printf("n_embd: %d\n", hparams.n_embd);
        printf("n_ff: %d\n", hparams.n_ff);
        printf("n_proj: %d\n", hparams.n_proj);
        printf("n_head: %d\n", hparams.n_head);
        printf("n_layer: %d\n", hparams.n_layer);
        printf("f_norm_eps: %f\n", hparams.f_norm_eps);
        printf("n_head: %d\n", hparams.n_head);
        printf("image_mean: %f %f %f\n", hparams.image_mean[0], hparams.image_mean[1], hparams.image_mean[2]);
        printf("image_std: %f %f %f\n", hparams.image_std[0], hparams.image_std[1], hparams.image_std[2]);
        printf("proj_type_str: %s\n", proj_type_str);
        printf("------------\n");
    }
    gguf_free(meta);
    return true;
}

bool moondream_mmproj_embed(
    moondream_mmproj_context & mctx,
    moondream_mmproj & model,
    moondream_image & image // use the batch instead
)  {
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
        mctx.inp_raw, image.data, 0, image.n_scalars * ggml_element_size(mctx.inp_raw));
    ggml_backend_tensor_set(
        mctx.positions, image.pos, 0, image.n_positions * ggml_element_size(mctx.positions));
    printf("WARNING: moondream_mmproj_context.positions values were not initialized!\n");

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
    if (!image.data)
    {
        printf("could not allocate memory for moondream_image data\n");
        return false;
    }
    image.pos = (int32_t *)malloc(sizeof(int32_t) * image.n_positions);
    if (!image.pos)
    {
        printf("could not allocate memory for moondream_image pos\n");
        return false;
    }
    for (int i = 0; i < image.n_positions; ++i)
    {
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
        path, &base_width, &base_height, &base_channels, MOONDREAM_N_IMAGE_CHANNELS);
    if (!base_stbi_data) {
        printf(
            "could not load \"%s\", stbi_failure_reason \"%s\"\n",
            path, stbi_failure_reason());
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

// bool moondream_mmproj_batch_init(moondream_mmproj_batch &batch)
// {
//     constexpr int n_elements_per_patch =
//         MOONDREAM_IMAGE_PATCH_SIDE_LENGTH * MOONDREAM_IMAGE_PATCH_SIDE_LENGTH * MOONDREAM_N_IMAGE_CHANNELS;
//     constexpr int n_elements_total = (1 + MOONDREAM_MAX_IMAGE_PATCHES) * n_elements_per_patch;
//     for (int i = 0; i < MAX_PATCHES; ++i)
//     {
//         init_patch(batch.patches[i]);
//     }
//     batch.patch_data = nullptr;
//     return true;
// }

// void moondream_mmproj_batch_free(moondream_mmproj_batch &batch)
// {
//     for (int i = 0; i < MAX_PATCHES; ++i)
//     {
//         free_patch(batch.patches[i]);
//         batch.patches[i]
//             .data = nullptr;
//     }
//     free(batch.patch_data);
// }

void find_target_image_size(moondream_image_alt_u8 & image, int & target_width, int & target_height) {

    // Reference python code:
    //
    // width, height = image.size
    // max_dim = max(width, height)
    // if max_dim < 512:
    //     im_size = (378, 378)
    // else:
    //     aspect_ratio = width / height
    //     im_size = min(
    //         self.supported_sizes,
    //         key=lambda size: (
    //             abs((size[1] / size[0]) - aspect_ratio),
    //             abs(size[0] - width) + abs(size[1] - height),
    //         ),
    //     )

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
            const float cur_inv_aspect_ratio = (float)cur_height / (float)cur_width;
            const float cur_aspect_ratio_diff = fabsf(cur_inv_aspect_ratio - aspect_ratio);
            const int cur_side_length_diff = abs(cur_width - image.width) + abs(cur_height - image.height);
            if (cur_aspect_ratio_diff < min_aspect_ratio_diff) {
                min_aspect_ratio_diff = cur_aspect_ratio_diff;
                target_width = cur_width;
                target_height = cur_height;
            } else if (
                cur_aspect_ratio_diff == min_aspect_ratio_diff
                && cur_side_length_diff < min_side_length_diff
            ) {
                min_side_length_diff = cur_side_length_diff;
                target_width = cur_width;
                target_height = cur_height;
            }
        }
    }
}

// allocates and initializes the batch.patch_data with an N-sized array of 378x378x3 f32
bool patches_to_batch(
    moondream_image_alt_f32 & image, moondream_patch_set patch_set, moondream_mmproj_batch & batch
) {
    constexpr size_t n_patch_elements =
        MOONDREAM_IMAGE_PATCH_SIDE_LENGTH * MOONDREAM_IMAGE_PATCH_SIDE_LENGTH * MOONDREAM_N_IMAGE_CHANNELS;
    size_t patch_byte_size = n_patch_elements * sizeof(float);
    batch.n_batch = patch_set.count + 1;
    batch.patch_data = (float *)malloc(batch.n_batch * patch_byte_size);
    if (batch.patch_data == nullptr) {
        printf("failed to allocate memory for moondream_mmproj_batch patch_data\n");
        return false;
    }
    memcpy(batch.patch_data, image.data, patch_byte_size);

    for (int i = 0; i < patch_set.count; ++i) {
        // offset should be in number of floats
        size_t offset = (i + 1) * n_patch_elements;
        memcpy((void *)&batch.patch_data[offset], patch_set.patches[i].data, patch_byte_size);
    }
    return true;
}

bool save_image_batch_to_pngs(moondream_mmproj_batch & batch) {
    printf("n_batch %d\n", batch.n_batch);
    constexpr size_t side_length = MOONDREAM_IMAGE_PATCH_SIDE_LENGTH;
    constexpr size_t n_channels = MOONDREAM_N_IMAGE_CHANNELS;
    constexpr size_t n_patch_elements = side_length * side_length * n_channels;
    uint8_t * temp_image = (uint8_t *)malloc(sizeof(uint8_t) * n_patch_elements);
    for (int n = 0; n < batch.n_batch; ++n) {
        float* cur_patch = &batch.patch_data[n * n_patch_elements];
        for (int i = 0; i < n_patch_elements; ++i) {
            float de_normalized = (cur_patch[i] * 0.5f) + 0.5f;
            temp_image[i] = static_cast<uint8_t>(de_normalized * 255.0f);
        }
        const char* base_path = "../../../data/image_patch_%d.png";
        size_t base_path_length = strlen(base_path);
        char path_buf[base_path_length];
        snprintf(path_buf, base_path_length, base_path, n);
        const int write_success = stbi_write_png(
            path_buf,
            side_length,
            side_length,
            n_channels,
            temp_image,
            side_length * n_channels);
        if (!write_success) {
            printf("failed to write image patch to png\n");
            free(temp_image);
            return false;
        }
    }
    free(temp_image);
    return true;
}

/*

    TODO: convert to float16 vs GGUF_TYPE_FLOAT32?

*/
bool moondream_mmproj_image_preprocess(const char * img_path, moondream_mmproj_batch & batch) {

    //
    // Step 0: Load image
    //
    moondream_image_alt_u8 image;
    if (!load_img_to_u8(img_path, image))
    {
        printf("failed to load image\n");
        return false;
    }

    //
    // STEP 1: Find target width and height.
    //
    int target_width = -1, target_height = -1;
    find_target_image_size(image, target_width, target_height);

    // Step 2: Resize image to target width and height
    void * resized_image_data = stbir_resize(
        image.data,
        image.width,
        image.height,
        image.width * MOONDREAM_N_IMAGE_CHANNELS,
        NULL,
        target_width,
        target_height,
        target_width * MOONDREAM_N_IMAGE_CHANNELS,
        STBIR_RGB,
        STBIR_TYPE_UINT8,
        STBIR_EDGE_CLAMP,
        STBIR_FILTER_TRIANGLE);
    if (resized_image_data == NULL) {
        printf("stbir failed to resize resized_image_data\n");
        return false;
    }
    moondream_image_alt_u8 resized_image_u8;
    init_moondream_image_alt_u8(
        resized_image_u8, target_width, target_height, (unsigned char *)resized_image_data);

    // Step 3: convert to float32 AND normalize
    // reference python:
    // ToDtype(torch.float32, scale=True)
    // Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    moondream_image_alt_f32 resized_image_f32;
    const float mean[3] = {0.5f, 0.5f, 0.5f};
    const float std[3] = {0.5f, 0.5f, 0.5f};
    if (!normalize_image_u8_to_f32(&resized_image_u8, &resized_image_f32, mean, std)) {
        printf("failed to normalize resized_image_u8\n");
        return false;
    }

    // Step 5: Split image into patches
    moondream_patch_set patch_set;
    init_patch_set(patch_set);
    create_patches(resized_image_f32, patch_set);
    printf("patch_set.count %d\n", patch_set.count);

    // Step 6: Resize image to (378, 378) if it isn't already that size
    moondream_image_alt_f32 patch_size_image_f32;
    if (resized_image_f32.width != MOONDREAM_IMAGE_PATCH_SIDE_LENGTH
        || resized_image_f32.height != MOONDREAM_IMAGE_PATCH_SIDE_LENGTH
    ) {
        void * patch_size_image_data = stbir_resize(
            resized_image_data,
            target_width,
            target_height,
            target_width * MOONDREAM_N_IMAGE_CHANNELS,
            NULL,
            MOONDREAM_IMAGE_PATCH_SIDE_LENGTH,
            MOONDREAM_IMAGE_PATCH_SIDE_LENGTH,
            MOONDREAM_IMAGE_PATCH_SIDE_LENGTH * MOONDREAM_N_IMAGE_CHANNELS,
            STBIR_RGB,
            STBIR_TYPE_UINT8,
            STBIR_EDGE_CLAMP,
            STBIR_FILTER_TRIANGLE);
        if (patch_size_image_data == NULL) {
            printf("stbir failed to resize patch_size_image_data\n");
            return false;
        }
        moondream_image_alt_u8 patch_size_image_u8;
        init_moondream_image_alt_u8(
            patch_size_image_u8,
            MOONDREAM_IMAGE_PATCH_SIDE_LENGTH,
            MOONDREAM_IMAGE_PATCH_SIDE_LENGTH,
            (unsigned char *)patch_size_image_data);
        if (!normalize_image_u8_to_f32(&patch_size_image_u8, &patch_size_image_f32, mean, std)) {
            printf("failed to normalize image\n");
            return false;
        }
    }

    // Step 7: Combined resized image with patches
    if (patch_size_image_f32.data == nullptr) {
        patches_to_batch(resized_image_f32, patch_set, batch);
    } else {
        patches_to_batch(patch_size_image_f32, patch_set, batch);
    }

    free_moondream_image_alt_u8(resized_image_u8);
    free_moondream_image_alt_f32(resized_image_f32);
    free_moondream_image_alt_f32(patch_size_image_f32);
    free_patch_set(patch_set);
    return true;
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

void mmproj_image_downsample_buffer_free(mmproj_image_downsample_buffer &buf) {
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
        "../../../assets/small_text.jpg",
        &base_width, &base_height, &base_channels, MOONDREAM_N_IMAGE_CHANNELS);

    if (!base_stbi_data) {
        printf("stb could not load image\n");
        return;
    }

    int new_width = 378;
    int new_height = 378;

    // unsigned char *resized_image = stbir_resize_uint8_srgb(base_stbi_data, base_width, base_height, base_width * MOONDREAM_N_IMAGE_CHANNELS,
    //                                                        NULL, new_width, new_height, new_width * MOONDREAM_N_IMAGE_CHANNELS,
    //                                                        STBIR_RGB);

    void * resized_image = stbir_resize(
        base_stbi_data,
        base_width,
        base_height,
        base_width * MOONDREAM_N_IMAGE_CHANNELS,
        NULL,
        new_width,
        new_height,
        new_width * MOONDREAM_N_IMAGE_CHANNELS,
        STBIR_RGB,
        STBIR_TYPE_UINT8,
        STBIR_EDGE_CLAMP,
        STBIR_FILTER_TRIANGLE);

    if (resized_image == NULL) {
        printf("stbir failed to resize image\n");
        return;
    }

    // free original image
    stbi_image_free(base_stbi_data);

    const int write_success = stbi_write_png(
        "../../../data/bilinear_downsample_preview.png",
        new_width, new_height, base_channels,
        resized_image,
        new_width * base_channels);
    if (!write_success) {
        printf("stbi failed to write image\n");
        return;
    }
    free(resized_image);
}
