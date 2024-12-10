import { Buffer } from 'buffer';
import sharp from 'sharp';
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

  private async* streamResponse(response: Response): AsyncGenerator<string, void, unknown> {
    if (!response.body) {
      throw new Error('Response body is null');
    }

    // Use the Web ReadableStream directly since fetch returns that
    const reader = response.body.getReader();
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
    request: CaptionRequest
  ): Promise<CaptionOutput> {
    const encodedImage = await this.encodeImage(request.image);

    const response = await fetch(`${this.apiUrl}/caption`, {
      method: 'POST',
      headers: {
        'X-Moondream-Auth': this.apiKey,
        'Content-Type': 'application/json',
        'User-Agent': `moondream-node/${version}`,
      },
      body: JSON.stringify({
        image_url: encodedImage.imageUrl,
        length: request.length,
        stream: request.stream,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    if (request.stream) {
      return { caption: this.streamResponse(response) };
    }

    const result = await response.json();
    return { caption: result.caption };
  }

  public async query(
    request: QueryRequest
  ): Promise<QueryOutput> {
    const encodedImage = await this.encodeImage(request.image);

    const response = await fetch(`${this.apiUrl}/query`, {
      method: 'POST',
      headers: {
        'X-Moondream-Auth': this.apiKey,
        'Content-Type': 'application/json',
        'User-Agent': `moondream-node/${version}`,
      },
      body: JSON.stringify({
        image_url: encodedImage.imageUrl,
        question: request.question,
        stream: request.stream,
        // TODO: Pass sampling settings
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    if (request.stream) {
      return { answer: this.streamResponse(response) };
    }

    const result = await response.json();
    return { answer: result.answer };
  }

  public async detect(
    request: DetectRequest
  ): Promise<DetectOutput> {
    const encodedImage = await this.encodeImage(request.image);

    const response = await fetch(`${this.apiUrl}/detect`, {
      method: 'POST',
      headers: {
        'X-Moondream-Auth': this.apiKey,
        'Content-Type': 'application/json',
        'User-Agent': `moondream-node/${version}`,
      },
      body: JSON.stringify({
        image_url: encodedImage.imageUrl,
        object: request.object,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const result = await response.json();
    return { objects: result.objects };
  }

  public async point(
    request: PointRequest
  ): Promise<PointOutput> {
    const encodedImage = await this.encodeImage(request.image);

    const response = await fetch(`${this.apiUrl}/point`, {
      method: 'POST',
      headers: {
        'X-Moondream-Auth': this.apiKey,
        'Content-Type': 'application/json',
        'User-Agent': `moondream-node/${version}`,
      },
      body: JSON.stringify({
        image_url: encodedImage.imageUrl,
        object: request.object,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const result = await response.json();
    return { points: result.points };
  }
}