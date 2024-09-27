import { query_cloud, find_cloud, describe_cloud } from "./md_cloud.js";
import { init_local } from "./md_local.js";

// const moodream = require("bindings")("moondream_ggml");
import bindings from "bindings";

const moondream = bindings("moondream_node");
function init_cloud(api_key, options) {
  return {
    query: function (image_path, query, callback) {
      return query_cloud(api_key, options, image_path, query, callback);
    },
    find: function (image_path, query, callback) {
      return find_cloud(api_key, options, image_path, query, callback);
    },
    describe: function (image_path, query, callback) {
      return describe_cloud(api_key, options, image_path, query, callback);
    },
  };
}

function is_api_key(key_or_path) {
  if (key_or_path.text_model) {
    return false;
  }
  return true;
  // return key_or_path.startsWith("XCP.");
}

// function init_local(text_model_path, mmproj_path) {
//   console.log("init_local", text_model_path, mmproj_path);
//   return query_local.init(text_model_path, mmproj_path);
// }

function init(key_or_path, options) {
  moondream.add(2, 3);
  if (is_api_key(key_or_path)) {
    return init_cloud(key_or_path, options);
  }
  return init_local(key_or_path.text_model, key_or_path.mmproj);
}

export default init;
// module.exports = init; // Just reexport it
