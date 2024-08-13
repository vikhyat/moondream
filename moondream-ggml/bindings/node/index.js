import { query_cloud } from "./md_cloud.js";
// const moodream = require("bindings")("moondream_ggml");
import bindings from "bindings";

const moondream = bindings("moondream_node");
function init_cloud(api_key, options) {
  return {
    query: function (image_path, query, callback) {
      return query_cloud(api_key, options, image_path, query, callback);
    },
  };
}

function init_local(model_path) {
  throw new Error("Not implemented yet");
}

function is_api_key(key_or_path) {
  return key_or_path.startsWith("XCP.");
}

function init(key_or_path, options) {
  moondream.add(2, 3);
  if (is_api_key(key_or_path)) {
    return init_cloud(key_or_path, options);
  }
  return init_local(key_or_path);
}

export default init;
// module.exports = init; // Just reexport it
