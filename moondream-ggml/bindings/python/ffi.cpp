#include <pybind11/pybind11.h>
#include "moondream.h"

PYBIND11_MODULE(moondream_ggml_ffi, m) {
    m.def(
        "init_api", 
        &moondream_api_state_init,
        pybind11::arg("text_model_path"),
        pybind11::arg("mmproj_path"),
        pybind11::arg("n_threads"),
        "Initialize the Moondream API state"
    );

    m.def(
        "cleanup_api", 
        &moondream_api_state_cleanup,
        "Clean up the Moondream API state"
    );
}
