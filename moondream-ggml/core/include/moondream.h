#pragma once

#include <cstdint>
#include <string.h>

bool moondream_api_state_init(
    const char * text_model_path, const char * mmproj_path, uint32_t n_threads
);
void moondream_api_state_cleanup(void);
bool moondream_api_prompt(
    const char * image_path, const char * prompt, std::string & response,
    int n_max_gen, bool log_response_stream
);
