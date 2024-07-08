#pragma once

#include <cstdint>

bool moondream_init_api_state(const char * text_model_path, const char * mmproj_path, uint32_t n_threads);
void moondream_cleanup_api_state(void);
