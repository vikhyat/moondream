#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <tuple>
#include "moondream.h"

PYBIND11_MODULE(cpp_ffi, m) {
    m.def(
        "init", 
        &moondream_api_state_init,
        pybind11::arg("text_model_path"),
        pybind11::arg("mmproj_path"),
        pybind11::arg("n_threads"),
        "Initialize the Moondream API state"
    );

    m.def(
        "cleanup", 
        &moondream_api_state_cleanup,
        "Clean up the Moondream API state"
    );

    m.def(
        "prompt",
        [](const std::string & prompt_str, int n_max_gen, bool log_response_stream) {
            std::string response;
            bool result = moondream_api_prompt(
                prompt_str.c_str(), response, n_max_gen, log_response_stream
            );
            return std::make_tuple(result, response);
        },
        pybind11::arg("prompt_str"),
        pybind11::arg("n_max_gen"),
        pybind11::arg("log_response_stream"),
        "Prompt Moondream"
    );
}
