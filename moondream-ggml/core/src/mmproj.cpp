#include <cassert>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <limits>

#include "ggml.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include "stb_image_resize2.h"
#include "helpers.hpp"
#include "mmproj.hpp"
#include "patch.hpp"

static void patch_bilinear_downsample_custom_op(
    ggml_tensor * dst, const ggml_tensor * src, int ith, int nth, void * userdata
) {
    // Only run on the first thread for prototyping
    if (ith != 0) {
        return;
    }
    const int n_channels = src->ne[0];
    const int patch_height = src->ne[1];
    const int patch_width = src->ne[2];
    const int y_gaps = (patch_height / 2) - 1;
    const int x_gaps = (patch_width / 2) - 1;
    const float y_gap_size = (float)patch_height / (float)(y_gaps);
    const float x_gap_size = (float)patch_width / (float)(x_gaps);

    for (int x = 0; x < x_gaps; ++x) {
        for (int y = 0; y < patch_height; ++y) {
            const int src_x_left = (int)floorf(x * x_gap_size);
            const int src_x_right = (int)floorf((x + 1) * x_gap_size);
            for (int channel = 0; channel < n_channels; ++channel) {
                const size_t src_offset_left = channel*src->nb[0] + y*src->nb[1] + src_x_right*src->nb[2];
                const size_t src_offset_right = channel*src->nb[0] + y*src->nb[1] + src_x_right*src->nb[2];
                const float src_left_val = *(float *)(((char *)src->data) + src_offset_left);
                const float src_right_val = *(float *)(((char *)src->data) + src_offset_right);
                const size_t dst_offset = channel*dst->nb[0] + y*dst->nb[1] + x*dst->nb[2];
                *(float *)(((char *)dst->data) + dst_offset) = 0.5f * (src_left_val + src_right_val);
            }
        }
    }

    for (int x = 0; x < x_gaps; ++x) {
        for (int y = 0; y < y_gaps; ++y) {
            const int dst_y_up = (int)floorf(y * y_gap_size);
            const int dst_y_down = (int)floorf((y + 1) * y_gap_size);
            for (int channel = 0; channel < n_channels; ++channel) {
                const size_t dst_offset_up = channel*dst->nb[0] + dst_y_up*src->nb[1] + x*dst->nb[2];
                const size_t dst_offset_down = channel*dst->nb[0] + dst_y_down*dst->nb[1] + x*dst->nb[2];
                const float dst_up_val = *(float *)(((char *)dst->data) + dst_offset_up);
                const float dst_down_val = *(float *)(((char *)dst->data) + dst_offset_down);
                *(float *)(((char *)dst->data) + dst_offset_up) = 0.5f * (dst_up_val + dst_down_val);
            }
        }
    }
}

static ggml_tensor * patch_bilinear_downsample(ggml_context * ctx, ggml_tensor * src) {
    src = ggml_cont(ctx, src);
    ggml_tensor * dst = ggml_map_custom1(ctx, src, patch_bilinear_downsample_custom_op, 1, NULL);
    int patch_out_height = src->ne[1] / 2;
    int patch_out_width = src->ne[2] / 2;
    printf("patch_out_height %d\n", patch_out_height);
    printf("patch_out_width %d\n", patch_out_width);
    printf(
        "dst shape: (%d, %d, %d, %d)\n",
        dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
    dst = ggml_view_3d(
        ctx, dst, dst->ne[0], patch_out_height, patch_out_width, dst->nb[1], dst->nb[2], 0);
    dst = ggml_cont(ctx, dst);
    return dst;
}

// Modification of llama.cpp/examples/llava/clip.cpp clip_image_build_graph.
// Ref: https://github.com/ggerganov/llama.cpp/blob/da799b41891e34aac86ce4e173f9c4   c0afd4fab3/examples/llava/clip.cpp
static ggml_cgraph * mmproj_build_clip(
    moondream_mmproj & model,
    moondream_mmproj_batch & batch,
    moondream_mmproj_context & mctx
) {
    moondream_mmproj_hparams & hparams = model.hparams;
    const int n_batch = batch.n_batch;
    const int n_outer_patches = batch.n_outer_patches;
    const int n_outer_patch_rows = batch.n_outer_patch_rows;
    const int n_outer_patch_cols = batch.n_outer_patch_cols;
    const int image_size = hparams.image_size;
    const int patch_size = hparams.patch_size;
    const int n_patches_per_side = mctx.n_patches_per_side;
    const int n_patches = mctx.n_patches;
    const int n_positions = mctx.n_positions;
    const int n_embd = hparams.n_embd;
    const int n_head = hparams.n_head;
    const int n_head_qkv = n_embd / n_head;
    const int n_layer = hparams.n_layer;
    const float eps = hparams.f_norm_eps;
    printf("n_batch: %d\n", n_batch);
    printf("n_head_qkv: %d\n", n_head_qkv);
    printf("n_head: %d\n", n_head);

    ggml_init_params build_ctx_params = {
        mctx.compute_buffer.size(),
        mctx.compute_buffer.data(),
        true};
    ggml_context * ctx0 = ggml_init(build_ctx_params);
    ggml_cgraph * gf = ggml_new_graph(ctx0);

    ggml_tensor * inp_raw = ggml_new_tensor_4d(
        ctx0, GGML_TYPE_F32, MOONDREAM_N_IMAGE_CHANNELS, image_size, image_size, n_batch);
    ggml_set_name(inp_raw, "inp_raw");
    ggml_set_input(inp_raw);
    mctx.inp_raw = inp_raw;
    inp_raw = ggml_permute(ctx0, inp_raw, 2, 1, 0, 3);
    inp_raw = ggml_cont(ctx0, ggml_permute(ctx0, inp_raw, 1, 0, 2, 3));
    printf(
        "inp_raw shape: (%d, %d, %d, %d)\n",
        inp_raw->ne[0], inp_raw->ne[1], inp_raw->ne[2], inp_raw->ne[3]);

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

    printf(
        "patch_embd shape: (%d, %d, %d, %d)\n",
        model.patch_embd->ne[0], model.patch_embd->ne[1],
        model.patch_embd->ne[2], model.patch_embd->ne[3]);
    // Shape: (n_patches_per_side, n_patches_per_side, n_embd, n_batch)
    ggml_tensor * inp = ggml_conv_2d(ctx0, model.patch_embd, inp_raw, patch_size, patch_size, 0, 0, 1, 1);
    // Shape: (n_patches, n_embd, n_batch, 1)
    inp = ggml_reshape_3d(ctx0, inp, n_patches, n_embd, n_batch);
    // Shape: (n_embd, n_patches, n_batch, 1)
    inp = ggml_cont(ctx0, ggml_permute(ctx0, inp, 1, 0, 2, 3));
    if (model.patch_bias != nullptr) {
        inp = ggml_add(ctx0, inp, model.patch_bias);
    }

    ggml_tensor * embeddings = inp;
    // NOTE: skipped class embeddings.
    ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_positions);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);
    mctx.positions = positions;
    embeddings = ggml_add(ctx0, embeddings, ggml_get_rows(ctx0, model.pos_embd, positions));
    // NOTE: skipped pre-layernorm.
    printf(
        "embeddinggs pre-block shape: (%d, %d, %d, %d)\n",
        embeddings->ne[0], embeddings->ne[1], embeddings->ne[2], embeddings->ne[3]);

    for (int il = 0; il < n_layer - 1; ++il) {
        // embeddings = residual, cur = hidden_states
        ggml_tensor * cur = embeddings;

        // Layernorm 1
        cur = ggml_norm(ctx0, cur, eps);
        cur = ggml_add(ctx0, ggml_mul(ctx0, cur, model.layers[il].ln_1_w), model.layers[il].ln_1_b);

        // Self-attention
        ggml_tensor * q = ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].q_w, cur), model.layers[il].q_b);
        //q = ggml_scale_inplace(ctx0, q, 1.0f / sqrtf((float)n_head_qkv));
        q = ggml_reshape_4d(ctx0, q, n_head_qkv, n_head, n_positions, n_batch);
        q = ggml_cont(ctx0, ggml_permute(ctx0, q, 0, 2, 1, 3));
        q = ggml_reshape_3d(ctx0, q, n_head_qkv, n_positions, n_head * n_batch);

        ggml_tensor * k = ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].k_w, cur), model.layers[il].k_b);
        k = ggml_reshape_4d(ctx0, k, n_head_qkv, n_head, n_positions, n_batch);
        k = ggml_cont(ctx0, ggml_permute(ctx0, k, 0, 2, 1, 3));
        k = ggml_reshape_3d(ctx0, k, n_head_qkv, n_positions, n_head * n_batch);

        ggml_tensor * v = ggml_add(ctx0, ggml_mul_mat(ctx0, model.layers[il].v_w, cur), model.layers[il].v_b);
        v = ggml_reshape_4d(ctx0, v, n_head_qkv, n_head, n_positions, n_batch);
        v = ggml_cont(ctx0, ggml_permute(ctx0, v, 1, 2, 0, 3));
        v = ggml_reshape_3d(ctx0, v, n_positions, n_head_qkv, n_head * n_batch);

        if (il == 0) {
            printf(
                "v il: (%d, %d, %d, %d)\n",
                v->ne[0], v->ne[1], v->ne[2], v->ne[3]);
            printf(
                "k il: (%d, %d, %d, %d)\n",
                k->ne[0], k->ne[1], k->ne[2], k->ne[3]);
            printf(
                "q il: (%d, %d, %d, %d)\n",
                q->ne[0], q->ne[1], q->ne[2], q->ne[3]);
            q = log_tensor(ctx0, q);
        }

        ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
        kq = ggml_scale_inplace(ctx0, kq, 1.0f / sqrtf((float)n_head_qkv));
        kq = ggml_soft_max_inplace(ctx0, kq);

        if (il == 0) {
            printf(
                "kq il: (%d, %d, %d, %d)\n",
                kq->ne[0], kq->ne[1], kq->ne[2], kq->ne[3]);
        }

        ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);

        if (il == 0) {
            printf(
                "kqv il: (%d, %d, %d, %d)\n",
                kqv->ne[0], kqv->ne[1], kqv->ne[2], kqv->ne[3]);
        }

        kqv = ggml_reshape_4d(ctx0, kqv, n_head_qkv, n_positions, n_head, n_batch);
        kqv = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
        cur = ggml_cont_3d(ctx0, kqv, n_embd, n_positions, n_batch);

        if (il == 0) {
            printf(
                "cur il: (%d, %d, %d, %d)\n",
                cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);
        }

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
    embeddings = ggml_cont(ctx0, embeddings);
    //embeddings = log_tensor(ctx0, embeddings);

    // Post-layernorm
    embeddings = ggml_norm(ctx0, embeddings, eps);
    ggml_set_name(embeddings, "post_ln");
    // Shape: (n_embd, n_patches, n_batch, 1)
    embeddings = ggml_add(ctx0, ggml_mul(ctx0, embeddings, model.post_ln_w), model.post_ln_b);
    printf(
        "embeddings post-block shape: (%d, %d, %d, %d)\n",
        embeddings->ne[0], embeddings->ne[1], embeddings->ne[2], embeddings->ne[3]);

    // Merge patch and full image features.
    ggml_tensor * full_img_features = ggml_view_2d(
        ctx0, embeddings, embeddings->ne[0], embeddings->ne[1], embeddings->nb[1], 0);
    printf(
        "full_img_features shape: (%d, %d, %d, %d)\n",
        full_img_features->ne[0], full_img_features->ne[1], full_img_features->ne[2], full_img_features->ne[3]);
    printf("n_outer_patches: %d\n", n_outer_patches);

    assert(embeddings->ne[2] - 1 == n_outer_patches);
    if (n_outer_patches > 0) {
        ggml_tensor * patch_features[n_outer_patches];
        for (int i = 0; i < n_outer_patches; ++i) {
            printf("row size %zu %zu %zu\n",embeddings->nb[0], embeddings->nb[1], embeddings->nb[2]);
            patch_features[i] = ggml_view_2d(
                ctx0, embeddings, embeddings->ne[0], embeddings->ne[1],
                ggml_row_size(embeddings->type, embeddings->ne[0]), embeddings->nb[2]);
            printf(
                "patch_features %d shape: (%d, %d, %d, %d)\n", i,
                patch_features[i]->ne[0], patch_features[i]->ne[1],
                patch_features[i]->ne[2], patch_features[i]->ne[3]);
            patch_features[i] = ggml_reshape_3d(
                ctx0, patch_features[i],
                patch_features[i]->ne[0], n_patches_per_side, n_patches_per_side);
            printf(
                "patch_features %d shape: (%d, %d, %d, %d)\n", i,
                patch_features[i]->ne[0], patch_features[i]->ne[1],
                patch_features[i]->ne[2], patch_features[i]->ne[3]);
        }

        ggml_tensor * row_features[n_outer_patch_rows];
        for (int row = 0; row < n_outer_patch_rows; ++row) {
            row_features[row] = patch_features[row * n_outer_patch_cols];
            for (int col = 1; col < n_outer_patch_cols; ++col) {
                row_features[row] = ggml_concat(
                    ctx0, row_features[row], patch_features[row * n_outer_patch_cols + col], 2);
            }
            printf(
                "row_features %d shape: (%d, %d, %d, %d)\n", row,
                row_features[row]->ne[0], row_features[row]->ne[1],
                row_features[row]->ne[2], row_features[row]->ne[3]);
        }

        ggml_tensor * merged_patch_features = row_features[0];
        for (int row = 1; row < n_outer_patch_rows; ++row) {
            merged_patch_features = ggml_concat(ctx0, merged_patch_features, row_features[row], 1);
        }
        printf(
            "merged_patch_features 1 shape: (%d, %d, %d, %d)\n",
            merged_patch_features->ne[0], merged_patch_features->ne[1],
            merged_patch_features->ne[2], merged_patch_features->ne[3]);
        merged_patch_features = patch_bilinear_downsample(ctx0, merged_patch_features);
        merged_patch_features = ggml_reshape_2d(
            ctx0, merged_patch_features, merged_patch_features->ne[0], n_patches);
        embeddings = ggml_concat(ctx0, full_img_features, merged_patch_features, 0);
    } else {
        embeddings = ggml_concat(ctx0, full_img_features, full_img_features, 0);
    }

    // Shape: (2*n_embd, n_patches)
    embeddings = ggml_cont(ctx0, embeddings);
    printf(
        "final embeddings shape: (%d, %d, %d, %d)\n",
        embeddings->ne[0], embeddings->ne[1],
        embeddings->ne[2], embeddings->ne[3]);

    if (hparams.proj_type == PROJECTOR_TYPE_MLP) {
        printf(
            "mm_0_w shape: (%d, %d, %d, %d)\n",
            model.mm_0_w->ne[0], model.mm_0_w->ne[1],
            model.mm_0_w->ne[2], model.mm_0_w->ne[3]);
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
    mctx.positions_storage = (int32_t *)malloc(sizeof(int32_t) * mctx.n_positions);
    if (!mctx.positions_storage) {
        printf("Failed to allocate memory for moondream_mmproj_batch postions buffer\n");
        return false;
    }
    for (int i = 0; i < mctx.n_positions; ++i) {
        mctx.positions_storage[i] = (int32_t)i;
    }

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
        ggml_tensor_overhead() * MMPROJ_MAX_NODES + ggml_graph_overhead_custom(MMPROJ_MAX_NODES, false);
    if (normal_logs_enabled) {
        const double compute_buf_size_gib = bytes_to_gib(compute_buf_size);
        printf("new mmproj compute_buf_size is %zu B, %lf GiB\n", compute_buf_size, compute_buf_size_gib);
    }
    mctx.compute_buffer.resize(compute_buf_size);

    // Initialize scheduler with worst case graph.
    moondream_mmproj_batch batch;
    batch.n_outer_patches = MOONDREAM_MAX_IMAGE_PATCHES;
    batch.n_outer_patch_rows = MOONDREAM_MAX_OUTER_PATCH_ROWS;
    batch.n_outer_patch_cols = MOONDREAM_MAX_OUTER_PATCH_COLS;
    batch.n_batch = 1 + batch.n_outer_patches;
    ggml_cgraph * gf = mmproj_build_clip(model, batch, mctx);
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

void moondream_mmproj_context_free(moondream_mmproj_context & mctx)
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
    moondream_mmproj_batch & batch
)  {
    ggml_cgraph * gf = mmproj_build_clip(model, batch, mctx);
    if (!gf) {
        printf("failed to build mmproj compute graph\n");
        return false;
    }
    if (!ggml_backend_sched_alloc_graph(mctx.sched, gf)) {
        printf("failed to allocate graph for ggml_backend_sched_t\n");
        return false;
    }

    printf("mmproj batch n_scalars: %d\n", batch.n_scalars);
    printf("mmproj batch data[0]: %f\n", batch.data[0]);

    ggml_backend_tensor_set(
        mctx.inp_raw, batch.data, 0, batch.n_scalars * ggml_element_size(mctx.inp_raw));
    ggml_backend_tensor_set(
        mctx.positions, mctx.positions_storage, 0, mctx.n_positions * ggml_element_size(mctx.positions));

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

bool moondream_mmproj_batch_init(moondream_mmproj_batch & batch) {
    constexpr int n_elements_per_image =
        MOONDREAM_IMAGE_PATCH_SIDE_LENGTH * MOONDREAM_IMAGE_PATCH_SIDE_LENGTH * MOONDREAM_N_IMAGE_CHANNELS;
    constexpr int n_batch_max = 1 + MOONDREAM_N_IMAGE_CHANNELS;
    constexpr int n_elements_max = n_batch_max * n_elements_per_image;
    batch.data = (float *)malloc(sizeof(float) * n_elements_max);
    if (!batch.data) {
        printf("Failed to allocate memory for moondream_mmproj_batch data buffer\n");
        return false;
    }
    batch.image_side_length = MOONDREAM_IMAGE_PATCH_SIDE_LENGTH;
    batch.n_batch = 0;
    return true;
}

void moondream_mmproj_batch_free(moondream_mmproj_batch & batch) {
    if (batch.data) {
        free(batch.data);
        batch.data = nullptr;
    }
}

static void find_target_image_size(moondream_image_alt_u8 & image, int & target_width, int & target_height) {

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

bool patches_to_batch(
    moondream_image_alt_f32 & image, moondream_patch_set patch_set, moondream_mmproj_batch & batch
) {
    constexpr size_t n_patch_elements =
        MOONDREAM_IMAGE_PATCH_SIDE_LENGTH * MOONDREAM_IMAGE_PATCH_SIDE_LENGTH * MOONDREAM_N_IMAGE_CHANNELS;
    size_t patch_byte_size = n_patch_elements * sizeof(float);
    batch.n_outer_patch_rows = patch_set.n_rows;
    batch.n_outer_patch_cols = patch_set.n_cols;
    batch.n_outer_patches = patch_set.count;
    batch.n_batch = batch.n_outer_patches + 1;
    batch.n_scalars = n_patch_elements * batch.n_batch;
    assert(batch.n_outer_patch_rows * batch.n_outer_patch_cols == batch.n_outer_patches);
    if (batch.data == nullptr) {
        printf("failed to allocate memory for moondream_mmproj_batch data\n");
        return false;
    }
    memcpy(batch.data, image.data, patch_byte_size);

    for (int i = 0; i < patch_set.count; ++i) {
        // offset should be in number of floats
        size_t offset = (i + 1) * n_patch_elements;
        memcpy((void *)&batch.data[offset], patch_set.patches[i].data, patch_byte_size);
    }
    return true;
}

bool moondream_mmproj_batch_save_to_pngs(moondream_mmproj_batch & batch) {
    constexpr size_t side_length = MOONDREAM_IMAGE_PATCH_SIDE_LENGTH;
    constexpr size_t n_channels = MOONDREAM_N_IMAGE_CHANNELS;
    constexpr size_t n_patch_elements = side_length * side_length * n_channels;
    uint8_t * temp_image = (uint8_t *)malloc(sizeof(uint8_t) * n_patch_elements);
    for (int n = 0; n < batch.n_batch; ++n) {
        float * cur_patch = &batch.data[n * n_patch_elements];
        for (int i = 0; i < n_patch_elements; ++i) {
            float de_normalized = (cur_patch[i] * 0.5f) + 0.5f;
            temp_image[i] = static_cast<uint8_t>(de_normalized * 255.0f);
        }
        const char * base_path = "../../../data/image_patch_%d.png";
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
bool moondream_mmproj_load_image_to_batch(const char * img_path, moondream_mmproj_batch & batch) {

    //
    // Step 0: Load image
    //
    moondream_image_alt_u8 image;
    if (!load_img_to_u8(img_path, image)) {
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
        STBIR_FILTER_CATMULLROM /* bicubic */);
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
        patch_size_image_f32.data = (float *)stbir_resize(
            resized_image_f32.data,
            target_width,
            target_height,
            4 * target_width * MOONDREAM_N_IMAGE_CHANNELS,
            NULL,
            MOONDREAM_IMAGE_PATCH_SIDE_LENGTH,
            MOONDREAM_IMAGE_PATCH_SIDE_LENGTH,
            4 * MOONDREAM_IMAGE_PATCH_SIDE_LENGTH * MOONDREAM_N_IMAGE_CHANNELS,
            STBIR_RGB,
            STBIR_TYPE_FLOAT,
            STBIR_EDGE_CLAMP,
            STBIR_FILTER_TRIANGLE /* bilinear */);

        if (patch_size_image_f32.data == NULL) {
            printf("stbir failed to resize patch_size_image_data\n");
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
