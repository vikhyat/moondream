import { Buffer } from 'buffer';
import sharp from 'sharp';
import http from 'http';
import https from 'https';
import { version } from '../package.json';
import {
  Base64EncodedImage,
  CaptionOutput,
  QueryOutput,
  DetectOutput,
  PointOutput,
  CaptionRequest,
  QueryRequest,
  DetectRequest,
  PointRequest,
} from './types';

export interface MoondreamVLConfig {
  apiKey?: string;
  apiUrl?: string;
}
const DEFAULT_API_URL = 'https://api.moondream.ai/v1';

export class vl {
  private apiKey: string;
  private apiUrl: string;

  constructor(config: MoondreamVLConfig) {
    this.apiKey = config.apiKey || '';
    this.apiUrl = config.apiUrl || DEFAULT_API_URL;
    if (this.apiKey === '' && this.apiUrl === DEFAULT_API_URL) {
      throw new Error(
        'An apiKey is required for cloud inference. '
      );
    }
  }

  private async encodeImage(
    image: Buffer | Base64EncodedImage
  ): Promise<Base64EncodedImage> {
    if ('imageUrl' in image) {
      return image;
    }

    try {
      const MAX_SIZE = 768;

      // Process image with Sharp
      const metadata = await sharp(image).metadata();

      if (!metadata.width || !metadata.height) {
        throw new Error('Unable to get image dimensions');
      }

      const scale = MAX_SIZE / Math.max(metadata.width, metadata.height);
      let processedImage = sharp(image);

      if (scale < 1) {
        processedImage = processedImage.resize(
          Math.round(metadata.width * scale),
          Math.round(metadata.height * scale),
          {
            fit: 'inside',
            withoutEnlargement: true,
          }
        );
      }

      const buffer = await processedImage
        .toFormat('jpeg', { quality: 95 })
        .toBuffer();

      const base64Image = buffer.toString('base64');
      return {
        imageUrl: `data:image/jpeg;base64,${base64Image}`,
      };
    } catch (error) {
      throw new Error(
        `Failed to convert image to JPEG: ${(error as Error).message}`
      );
    }
  }

  private makeRequest(path: string, body: any, stream: boolean = false): Promise<any> {
    return new Promise((resolve, reject) => {
      const url = new URL(this.apiUrl + path);
      const requestBody = JSON.stringify(body);

      const options = {
        method: 'POST',
        headers: {
          'X-Moondream-Auth': this.apiKey,
          'Content-Type': 'application/json',
          'User-Agent': `moondream-node/${version}`,
          'Content-Length': Buffer.byteLength(requestBody)
        }
      };

      const client = url.protocol === 'https:' ? https : http;
      const req = client.request(url, options, (res) => {
        if (stream) {
          resolve(res);
          return;
        }

        let data = '';
        res.on('data', chunk => data += chunk);
        res.on('end', () => {
          if (res.statusCode !== 200) {
            reject(new Error(`HTTP error! status: ${res.statusCode}`));
            return;
          }
          try {
            resolve(JSON.parse(data));
          } catch (error) {
            reject(new Error(`Failed to parse JSON response: ${(error as Error).message}`));
          }
        });
      });

      req.on('error', (error) => {
        reject(error);
      });

      req.write(requestBody);
      req.end();
    });
  }

  private async* streamResponse(response: any): AsyncGenerator<string, void, unknown> {
    let buffer = '';

    // Create a promise that resolves when we get data or end
    const getNextChunk = () => new Promise<{ value: string | undefined; done: boolean }>((resolve, reject) => {
      const onData = (chunk: Buffer) => {
        try {
          buffer += chunk.toString();
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = JSON.parse(line.slice(6));
              if ('chunk' in data) {
                response.removeListener('data', onData);
                response.removeListener('end', onEnd);
                response.removeListener('error', onError);
                resolve({ value: data.chunk, done: false });
                return;
              }
              if (data.completed) {
                response.removeListener('data', onData);
                response.removeListener('end', onEnd);
                response.removeListener('error', onError);
                resolve({ value: undefined, done: true });
                return;
              }
            }
          }
        } catch (error) {
          response.removeListener('data', onData);
          response.removeListener('end', onEnd);
          response.removeListener('error', onError);
          reject(new Error(`Failed to parse JSON response from server: ${(error as Error).message}`));
        }
      };

      const onEnd = () => {
        response.removeListener('data', onData);
        response.removeListener('end', onEnd);
        response.removeListener('error', onError);
        resolve({ value: undefined, done: true });
      };

      const onError = (error: Error) => {
        response.removeListener('data', onData);
        response.removeListener('end', onEnd);
        response.removeListener('error', onError);
        reject(error);
      };

      response.on('data', onData);
      response.on('end', onEnd);
      response.on('error', onError);
    });

    try {
      while (true) {
        const result = await getNextChunk();
        if (result.done) {
          return;
        }
        if (result.value !== undefined) {
          yield result.value;
        }
      }
    } catch (error) {
      throw new Error(`Failed to stream response: ${(error as Error).message}`);
    }
  }

  public async caption(
    request: CaptionRequest
  ): Promise<CaptionOutput> {
    const encodedImage = await this.encodeImage(request.image);

    const response = await this.makeRequest('/caption', {
      image_url: encodedImage.imageUrl,
      length: request.length,
      stream: request.stream,
    }, request.stream);

    if (request.stream) {
      return { caption: this.streamResponse(response) };
    }

    return { caption: response.caption };
  }

  public async query(
    request: QueryRequest
  ): Promise<QueryOutput> {
    const encodedImage = await this.encodeImage(request.image);

    const response = await this.makeRequest('/query', {
      image_url: encodedImage.imageUrl,
      question: request.question,
      stream: request.stream,
    }, request.stream);

    if (request.stream) {
      return { answer: this.streamResponse(response) };
    }

    return { answer: response.answer };
  }

  public async detect(
    request: DetectRequest
  ): Promise<DetectOutput> {
    const encodedImage = await this.encodeImage(request.image);

    const response = await this.makeRequest('/detect', {
      image_url: encodedImage.imageUrl,
      object: request.object,
    });

    return { objects: response.objects };
  }

  public async point(
    request: PointRequest
  ): Promise<PointOutput> {
    const encodedImage = await this.encodeImage(request.image);

    const response = await this.makeRequest('/point', {
      image_url: encodedImage.imageUrl,
      object: request.object,
    });

    return { points: response.points };
  }
}