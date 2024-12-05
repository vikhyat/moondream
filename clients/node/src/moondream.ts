import { Buffer } from 'buffer';
import sharp from 'sharp';
import fetch from 'node-fetch';
import { Response as NodeFetchResponse } from 'node-fetch';
import { Readable } from 'node:stream';
import { ReadableStream } from 'node:stream/web';
import {
  Base64EncodedImage,
  Length,
  SamplingSettings,
  CaptionOutput,
  QueryOutput,
  DetectOutput,
} from './types';

export interface MoondreamVLConfig {
  apiKey: string;
}

export class vl {
  private apiKey: string;
  private apiUrl: string;

  constructor(config: MoondreamVLConfig) {
    this.apiKey = config.apiKey;
    this.apiUrl = 'https://api.moondream.ai/v1';
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

  private async* streamResponse(response: NodeFetchResponse): AsyncGenerator<string, void, unknown> {
    if (!response.body) {
      throw new Error('Response body is null');
    }

    // Convert Node.js readable stream to Web ReadableStream
    const webStream = new ReadableStream({
      start(controller) {
        if (response.body instanceof Readable) {
          response.body.on('data', chunk => controller.enqueue(chunk));
          response.body.on('end', () => controller.close());
          response.body.on('error', err => controller.error(err));
        } else {
          controller.error(new Error('Response body is not a readable stream'));
        }
      }
    });
    const reader = webStream.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6));
            if ('chunk' in data) {
              yield data.chunk;
            }
            if (data.completed) {
              return;
            }
          } catch (error) {
            throw new Error(
              `Failed to parse JSON response from server: ${(error as Error).message}`
            );
          }
        }
      }
    }
  }

  public async caption(
    image: Buffer | Base64EncodedImage,
    length: Length = 'normal',
    stream = false,
    settings?: SamplingSettings
  ): Promise<CaptionOutput> {
    const encodedImage = await this.encodeImage(image);

    const response = await fetch(`${this.apiUrl}/caption`, {
      method: 'POST',
      headers: {
        'X-Moondream-Auth': this.apiKey,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        image_url: encodedImage.imageUrl,
        length,
        stream,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    if (stream) {
      return { caption: this.streamResponse(response) };
    }

    const result = await response.json();
    return { caption: result.caption };
  }

  public async query(
    image: Buffer | Base64EncodedImage,
    question: string,
    stream = false,
    settings?: SamplingSettings
  ): Promise<QueryOutput> {
    const encodedImage = await this.encodeImage(image);

    const response = await fetch(`${this.apiUrl}/query`, {
      method: 'POST',
      headers: {
        'X-Moondream-Auth': this.apiKey,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        image_url: encodedImage.imageUrl,
        question,
        stream,
        // TODO: Pass sampling settings
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    if (stream) {
      return { answer: this.streamResponse(response) };
    }

    const result = await response.json();
    return { answer: result.answer };
  }

  public async detect(
    image: Buffer | Base64EncodedImage,
    object: string
  ): Promise<DetectOutput> {
    const encodedImage = await this.encodeImage(image);

    const response = await fetch(`${this.apiUrl}/detect`, {
      method: 'POST',
      headers: {
        'X-Moondream-Auth': this.apiKey,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        image_url: encodedImage.imageUrl,
        object,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const result = await response.json();
    return { objects: result.objects };
  }
}