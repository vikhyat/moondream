// const moodream = require("bindings")("moondream_ggml");
import bindings from "bindings";

const moondream = bindings("moondream_node");

export function init_local(
  text_model_path,
  mmproj_path
  //   n_threads,
  //   normal_logs_enabled
) {
  console.log("init_local", text_model_path, mmproj_path);
  const result = moondream.init(text_model_path, mmproj_path); // n_threads, normal_logs_enabled);
  console.log("result", result);

  return {
    query: function (image_path, query, callback) {
      return moondream.prompt(image_path, query, callback);
    },
  };
}

// export function cleanup() {
//   moondream.cleanup();
// }

// export function local_prompt(
//   image_path,
//   prompt_str,
//   n_max_gen,
//   log_response_stream
// ) {
//   return moondream.prompt(
//     image_path,
//     prompt_str,
//     n_max_gen,
//     log_response_stream
//   );
// }

// m.def(
//     "prompt",
//     [](
//         const std::string & image_path,
//         const std::string & prompt_str,
//         int n_max_gen, bool log_response_stream
//     ) {
//         std::string response;
//         bool result = moondream_api_prompt(
//             image_path.c_str(), prompt_str.c_str(),
//             response, n_max_gen, log_response_stream
//         );
//         return std::make_tuple(result, response);
//     },
//     pybind11::arg("image_path"),
//     pybind11::arg("prompt_str"),
//     pybind11::arg("n_max_gen"),
//     pybind11::arg("log_response_stream"),
//     "Prompt Moondream"
// );
