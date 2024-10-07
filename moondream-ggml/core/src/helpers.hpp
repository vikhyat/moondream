#pragma once

#include <cstdint>

#include "ggml.h"

#define MOONDREAM_N_IMAGE_CHANNELS 3
#define MOONDREAM_MAX_IMAGE_SIDE_LENGTH 756
#define MOONDREAM_MAX_IMAGE_PATCHES 4
#define MOONDREAM_IMAGE_PATCH_SIDE_LENGTH 378
// Corresponds to LLAMA_ROPE_TYPE_NEOX from llama.cpp which is what is used for phi2.
#define MOONDREAM_ROPE_TYPE 2
#define DATA_PATH_MAX_LEN 512
#define LLAMA_MAX_NODES 8192

// Define MOONDREAM_EXTRA_LOGS if you want additional logs for debugging.
// #define MOONDREAM_EXTRA_LOGS

// Define MOONDREAM_MULTI_MODAL if you want image embeddings to be generated
// and used for text model.
#define MOONDREAM_MULTI_MODAL

#define ARCH_PREFIX(t) ("phi2." t)
#define TOK_PREFIX(t) ("tokenizer.ggml." t)

size_t utf8_len(char src);
double bytes_to_gib(size_t n_bytes);
bool size_to_int32(size_t s, int32_t * i);
void set_tensor_name(ggml_tensor * cur, const char * name, int il);
void log_tensor(ggml_tensor * dst, const ggml_tensor * src, int ith, int nth, void * userdata);
