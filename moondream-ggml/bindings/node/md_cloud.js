import fs from "fs";
import fetch, { fileFrom } from "node-fetch";
import { fileTypeFromFile } from "file-type";

const api_url = "https://beta.api.moondream.ai";

export async function query_cloud(
  api_key,
  options,
  image_path,
  query,
  callback
) {
  const fileType = await fileTypeFromFile(image_path);
  const imgData = await fileFrom(image_path, fileType.mime);
  if (!imgData) {
    throw new Error(`Could not read file ${image_path}`);
  }

  const body = JSON.stringify({ prompt: query });
  const uri = `${api_url}/v1/query`;
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
