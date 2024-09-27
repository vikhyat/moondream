import fs from "fs";
import fetch, { fileFrom } from "node-fetch";
import { fileTypeFromFile } from "file-type";

const api_url = "https://beta.api.moondream.ai";

async function invoke_cloud(api_key, query, image, options, verb) {
  const fileType = await fileTypeFromFile(image);
  const imgData = await fileFrom(image, fileType.mime);
  if (!imgData) {
    throw new Error(`Could not read file ${image}`);
  }

  const body = JSON.stringify({ prompt: query });
  const uri = `${api_url}/v1/${verb}`;
  const formData = new FormData();
  formData.append("body", body);
  formData.append("content", imgData);
  const headers = {
    "X-MD-Auth": api_key,
  };
  const request = {
    method: "POST",
    body: formData,
    headers: headers,
  };
  const response = await fetch(uri, request);
  return {
    status: response.status,
    answer: await response.json(),
  };
}

export async function query_cloud(
  api_key,
  options,
  image_path,
  query,
  callback
) {
  return invoke_cloud(api_key, query, image_path, options, "query");
}

export async function find_cloud(
  api_key,
  options,
  image_path,
  query,
  callback
) {
  return invoke_cloud(api_key, query, image_path, options, "find");
}

export async function describe_cloud(
  api_key,
  options,
  image_path,
  query,
  callback
) {
  return invoke_cloud(api_key, query, image_path, options, "describe");
}
