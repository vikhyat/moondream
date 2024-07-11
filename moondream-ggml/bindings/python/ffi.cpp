#include <pybind11/pybind11.h>
#include "moondream.h"

PYBIND11_MODULE(moondream_ggml_ffi, m) {
    m.def(
        "moondream_init_api_state", 
        &moondream_init_api_state, 
        pybind11::arg("text_model_path"),
        pybind11::arg("mmproj_path"),
        pybind11::arg("n_threads"),
        "Initialize the Moondream API state"
    );

    m.def(
        "moondream_cleanup_api_state", 
        &moondream_cleanup_api_state,
        "Clean up the Moondream API state"
    );
}
