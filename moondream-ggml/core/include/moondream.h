#pragma once

#include <cstdint>

bool moondream_api_state_init(const char * text_model_path, const char * mmproj_path, uint32_t n_threads);
void moondream_api_state_cleanup(void);
bool moondream_api_prompt(const char * prompt);
