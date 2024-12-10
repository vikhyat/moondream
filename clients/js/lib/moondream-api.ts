
import sharp    from 'sharp';
import FormData from 'form-data';
import path     from 'node:path';
import axios    from 'axios';

const TARGET_SIZE = 760;
const TARGET_MIME = 'image/jpeg';
const JSON_MIME   = 'application/json';
const BODY_KEY    = 'body';
const CONTENT_KEY = 'content';
const AUTH_HEADER = 'X-MD-Auth';

import { APIConfiguration, APIResponse, APIRequest } from "./types"

export class MoondreamAPI {

  secret: string;
  endpoint: string;
  domain: string;

  constructor(config: APIConfiguration) {
    this.secret = config.secret;
    this.endpoint = config.endpoint;
    this.domain = process.env.MD_DOMAIN || 'https://api.moondream.api';
  }

  async request(req : APIRequest) : Promise<APIResponse> {
    try {
      const filename = path.basename(req.file);
      const buffer = await sharp(req.file)
        .resize(TARGET_SIZE, TARGET_SIZE)
        .jpeg()
        .toBuffer();
      return this.#call_api(filename, req.prompt, buffer);
    } catch (e) {
      if (e instanceof Error) {
        return { status: 400, type: "error", text: e.message };
      } else {
        return { status: 400, type: "error", text: JSON.stringify(e) };
      }
    }
  }

  async #call_api(filename : string, prompt : string, buffer: Buffer) : Promise<APIResponse> {
    const uri = `${this.domain}/v1/${this.endpoint}`;
    const form = new FormData();
    form.append(BODY_KEY, JSON.stringify({ prompt }), {
      contentType: JSON_MIME,
    });
    form.append(CONTENT_KEY, buffer, {
      filename,
      contentType: TARGET_MIME,
    });
    const headers = { [AUTH_HEADER]: this.secret };
    try {
      const response = await axios.post(uri, form, { headers });
      return {
        status: response.status,
        type: response.data?.type,
        text: response.data?.result
      }
    } catch(e) {
      if (axios.isAxiosError(e)) {
        return {
          status: e.response?.status || 503,
          type: "error",
          text: e.response?.data?.reason || e.code || "Undefined Error"
        }
      } else {
        // Unexpected result.
        return {
          status: 503,
          type: "error",
          text: JSON.stringify(e) 
        }
      }
    }
  }
}