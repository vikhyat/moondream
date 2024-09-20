#include <cstdint>
#include <cstring>
#include <string>

#include "helpers.hpp"
#include "lm.hpp"
#include "mmproj.hpp"

struct moondream_api_state
{
    bool is_init = false;
    bool normal_logs_enabled = false;
    moondream_lm model;
    moondream_mmproj mmproj_model;
    moondream_lm_context mctx;
    moondream_mmproj_context mmproj_ctx;
    moondream_image image;
};

static moondream_api_state api_state;

bool moondream_api_state_init(
    const char *text_model_path, const char *mmproj_path,
    int n_threads, bool normal_logs_enabled)
{
    if (api_state.is_init)
    {
        printf("API has already been initialized\n");
        return false;
    }

    /* Start of moondream_lm load. */
    bool result = moondream_lm_load_from_file(text_model_path, api_state.model, normal_logs_enabled);
    if (!result)
    {
        printf("could not load text model\n");
        return false;
    }
    /* End of moondream_lm load. */

    /* Start of moondream_mmproj load. */
    result = moondream_mmproj_load_from_file(mmproj_path, api_state.mmproj_model, normal_logs_enabled);
    if (!result)
    {
        printf("could not load mmproj model\n");
        return false;
    }
    /* End of moondream_mmproj load. */

    /* Start of moondream_mmproj_context init. */
    result = moondream_mmproj_context_init(
        api_state.mmproj_ctx, api_state.mmproj_model, n_threads, normal_logs_enabled);
    if (!result)
    {
        printf("failed to initialze moondream_mmproj_context\n");
        return 1;
    }
    /* End of moondream_mmproj_context init. */

    /* Start of moondream_lm_context init. */
    moondream_lm_cparams cparams = {
        .n_ctx = 2048, /*api_state.model.hparams.n_ctx_train,*/
        .n_batch = 2048,
        .n_ubatch = 512,
        .n_seq_max = 1,
        .n_threads = n_threads,
        .n_threads_batch = n_threads,
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
        .flash_attn = false};
    const ggml_type type_k = GGML_TYPE_F16;
    const ggml_type type_v = GGML_TYPE_F16;
    result = moondream_lm_context_init(
        api_state.mctx, api_state.model.hparams, cparams, api_state.model,
        type_k, type_v, normal_logs_enabled);
    if (!result)
    {
        printf("failed to initialze moondream_lm_context\n");
        return false;
    }
    api_state.mctx.n_outputs = 1;
    /* End of moondream_lm_context init. */

    /* Start of moondream_image init. */
    const int image_size = api_state.mmproj_model.hparams.image_size;
    const int n_positions_mm = api_state.mmproj_ctx.n_positions;
    if (!moondream_image_init(api_state.image, image_size, n_positions_mm))
    {
        printf("failed to initialize moondream_image\n");
        return false;
    }
    /* End of moondream_image init. */

    api_state.normal_logs_enabled = normal_logs_enabled;
    api_state.is_init = true;
    return true;
}

void moondream_api_state_cleanup(void)
{
    if (!api_state.is_init)
    {
        return;
    }
    moondream_lm_context_free(api_state.mctx);
    moondream_mmproj_context_free(api_state.mmproj_ctx);
    ggml_free(api_state.mmproj_model.ctx);
    ggml_free(api_state.model.ctx);
    moondream_image_free(api_state.image);
    api_state.is_init = false;
}

bool moondream_api_prompt(
    const char *image_path, const char *prompt, std::string &response,
    int n_max_gen, bool log_response_stream)
{
    moondream_lm &model = api_state.model;
    moondream_lm_hparams &hparams = model.hparams;
    moondream_lm_context &mctx = api_state.mctx;
    moondream_lm_cparams &cparams = mctx.cparams;
    moondream_mmproj_context &mmproj_ctx = api_state.mmproj_ctx;
    moondream_mmproj &mmproj = api_state.mmproj_model;
    moondream_image &image = api_state.image;
    const bool normal_logs_enabled = api_state.normal_logs_enabled;

    moondream_mmproj_batch img_batch;

    if (!moondream_mmproj_image_preprocess(image_path, img_batch))
    {
        printf("failed to initialized moondream_lm_batch\n");
        return false;
    }

    moondream_lm_batch batch;

    if (!moondream_lm_batch_init(batch, cparams.n_ctx, model.hparams.n_embd, false))
    {
        printf("failed to initialized moondream_lm_batch\n");
        return false;
    }

    int32_t prompt_len = 0;
    if (!size_to_int32(strlen(prompt), &prompt_len))
    {
        printf("prompt was too big (length greater than max 32 bit integer\n");
        return false;
    }
    int32_t prompt_token_ids[prompt_len];
    int32_t n_prompt_tokens = moondream_lm_tokenize(model.vocab, prompt, prompt_len, prompt_token_ids);
    if (n_prompt_tokens < 0)
    {
        printf("failed to tokenize prompt\n");
        return 1;
    }

    if (api_state.normal_logs_enabled)
    {
        printf("n_prompt_tokens: %d\n", n_prompt_tokens);
        printf("prompt_token_ids: ");
        for (int i = 0; i < n_prompt_tokens; ++i)
        {
            printf("%d ", prompt_token_ids[i]);
        }
        printf("\n");
    }

    if (log_response_stream)
    {
        printf("------------\n");
    }

#ifdef MOONDREAM_MULTI_MODAL
    // call image_preprocess
    // tghen pass it into moondream_mmproj_embed

    if (!moondream_image_load_and_set(image_path, image))
    {
        printf("failed to load and set moondream_image\n");
        return false;
    }
    if (!moondream_mmproj_embed(api_state.mmproj_ctx, api_state.mmproj_model, image))
    {
        printf("failed to create image embeddings\n");
        return false;
    }
    const bool decode_success = moondream_lm_decode(mctx,
                                                    model, batch, response,
                                                    n_prompt_tokens, prompt_token_ids,
                                                    n_max_gen, log_response_stream,
                                                    mmproj_ctx.output_buffer, mmproj_ctx.n_patches, mmproj.hparams.n_proj);
#else  // MOONDREAM_MULTI_MODAL
    const bool decode_success = moondream_lm_decode(
        mctx, model, batch, response,
        n_prompt_tokens, prompt_token_ids,
        n_max_gen, log_response_stream,
        nullptr, 0, 0 /* Don't pass any mmproj embeddings. */
    );
#endif // !MOONDREAM_MULTI_MODAL

    if (!decode_success)
    {
        printf("moondream decode failed\n");
        return false;
    }

    if (log_response_stream)
    {
        printf("------------\n");
    }

    moondream_lm_batch_free(batch);
    return true;
}

#ifndef MOONDREAM_LIBRARY_BUILD
int main(int argc, char *argv[])
{
    test_bilinear_downsample();

    const char *lm_fname = "moondream2-text-model-f16.gguf";
    const char *mmproj_fname = "moondream2-mmproj-f16.gguf";

    if (argc < 2)
    {
        printf("incorrect number of arguments\n");
        return 1;
    }
    const char *data_path = argv[1];
    const size_t data_path_length = strlen(data_path);
    if (data_path_length > DATA_PATH_MAX_LEN)
    {
        printf("provided data path exceeded max length");
        return 1;
    }

    // Resolve text model file path.
    const size_t lm_fname_length = strlen(lm_fname);
    // Add 1 to give space for null-terminator in concatenated string.
    const size_t lm_path_length = data_path_length + lm_fname_length + 1;
    char lm_path[lm_path_length];
    snprintf(lm_path, lm_path_length, "%s%s", data_path, lm_fname);

    // Resolve mmproj file path.
    const size_t mmproj_fname_length = strlen(mmproj_fname);
    // Add 1 to give space for null-terminator in concatenated string.
    const size_t mmproj_path_length = data_path_length + mmproj_fname_length + 1;
    char mmproj_path[mmproj_path_length];
    snprintf(mmproj_path, mmproj_path_length, "%s%s", data_path, mmproj_fname);

    printf("lm path: %s\n", lm_path);
    printf("mmproj path: %s\n", mmproj_path);

    // Initialize GGML.
    ggml_time_init();
    // Optional NUMA initialization for better performance on supported systems.
    /*enum ggml_numa_strategy numa_strat = GGML_NUMA_STRATEGY_DISTRIBUTE;
    if (ggml_is_numa()) {
        printf("numa node detected, initializing ggml numa\n");
        ggml_numa_init(numa_strat);
    }*/

    const int n_threads = 8;
    const bool normal_logs_enabled = true;
    if (!moondream_api_state_init(lm_path, mmproj_path, n_threads, normal_logs_enabled))
    {
        printf("failed to initialize api state\n");
        return 1;
    }

    // Assuming the binary will be run from ../build/
    const char *image_path = "../../../assets/demo-1.jpg";
    const char *prompt = "<image>\n\nQuestion: Describe the image.\n\nAnswer:";
    std::string response = "";
    if (!moondream_api_prompt(image_path, prompt, response, 128, true))
    {
        printf("prompt failed\n");
        return 1;
    }
    moondream_api_state_cleanup();
    return 0;
}
#endif // !MOONDREAM_LIBRARY_BUILD
