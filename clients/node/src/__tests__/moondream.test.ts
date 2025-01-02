import { vl, MoondreamVLConfig } from '../moondream';
import { CaptionRequest, DetectRequest, PointRequest, QueryRequest } from '../types';
import fetchMock from 'jest-fetch-mock';

// Mock sharp
jest.mock('sharp', () => {
  return jest.fn().mockImplementation(() => ({
    metadata: () => Promise.resolve({ width: 1024, height: 768 }),
    resize: function () { return this; },
    toFormat: function () { return this; },
    toBuffer: () => Promise.resolve(Buffer.from('mock-image-data')),
  }));
});

describe('MoondreamClient', () => {
  let client: vl;
  const mockApiKey = 'test-api-key';
  const mockImageBuffer = Buffer.from('mock-image-data');
  const mockBase64Image = {
    imageUrl: 'data:image/jpeg;base64,mock-image-data'
  };

  beforeEach(() => {
    const moondreamConfig: MoondreamVLConfig = {
      apiKey: mockApiKey
    };
    client = new vl(moondreamConfig);
    fetchMock.resetMocks();
  });

  describe('caption', () => {
    it('should successfully get a caption for an image buffer', async () => {
      fetchMock.mockResponseOnce(JSON.stringify({ caption: 'A beautiful landscape' }));

      const request: CaptionRequest = {
        image: mockImageBuffer,
        length: 'normal',
        stream: false
      };
      const result = await client.caption(request);

      expect(result).toEqual({ caption: 'A beautiful landscape' });
      expect(fetchMock).toHaveBeenCalledWith(
        'https://api.moondream.ai/v1/caption',
        expect.objectContaining({
          method: 'POST',
          headers: {
            'X-Moondream-Auth': mockApiKey,
            'Content-Type': 'application/json',
            'User-Agent': expect.stringMatching(/^moondream-node\/\d+\.\d+\.\d+$/)
          }
        })
      );
    });

    it('should handle streaming responses', async () => {
      const response = new Response(new ReadableStream({
        async start(controller) {
          controller.enqueue(new TextEncoder().encode('data: {"chunk":"test chunk"}\n'));
          controller.enqueue(new TextEncoder().encode('data: {"completed":true}\n'));
          controller.close();
        }
      }));
      response.ok = true;

      fetchMock.mockResolvedValueOnce(response);

      const request: CaptionRequest = {
        image: mockImageBuffer,
        length: 'normal',
        stream: true
      };
      const result = await client.caption(request);
      expect(result.caption).toBeDefined();

      const chunks = [];
      for await (const chunk of result.caption as AsyncGenerator<string>) {
        chunks.push(chunk);
      }
      expect(chunks).toEqual(['test chunk']);
    });

    it('should throw an error on API failure', async () => {
      fetchMock.mockResponseOnce('', { status: 400 });

      const request: CaptionRequest = {
        image: mockImageBuffer,
        length: 'normal',
        stream: false
      };
      await expect(client.caption(request))
        .rejects
        .toThrow('HTTP error! status: 400');
    });
  });

  describe('query', () => {
    it('should successfully query about an image', async () => {
      fetchMock.mockResponseOnce(JSON.stringify({ answer: 'This is a dog' }));

      const request: QueryRequest = {
        image: mockImageBuffer,
        question: 'What is in this image?'
      };
      const result = await client.query(request);

      expect(result).toEqual({ answer: 'This is a dog' });
      expect(fetchMock).toHaveBeenCalledWith(
        'https://api.moondream.ai/v1/query',
        expect.objectContaining({
          method: 'POST',
          headers: {
            'X-Moondream-Auth': mockApiKey,
            'Content-Type': 'application/json',
            'User-Agent': expect.stringMatching(/^moondream-node\/\d+\.\d+\.\d+$/)
          }
        })
      );
    });

    it('should handle streaming query responses', async () => {
      const response = new Response(new ReadableStream({
        async start(controller) {
          controller.enqueue(new TextEncoder().encode('data: {"chunk":"test answer"}\n'));
          controller.enqueue(new TextEncoder().encode('data: {"completed":true}\n'));
          controller.close();
        }
      }));
      response.ok = true;

      fetchMock.mockResolvedValueOnce(response);

      const request: QueryRequest = {
        image: mockImageBuffer,
        question: 'What is this?',
        stream: true
      };
      const result = await client.query(request);
      expect(result.answer).toBeDefined();

      const chunks = [];
      for await (const chunk of result.answer as AsyncGenerator<string>) {
        chunks.push(chunk);
      }
      expect(chunks).toEqual(['test answer']);
    });
  });

  describe('detect', () => {
    it('should successfully detect objects in an image', async () => {
      const mockObjects = [
        { x_min: 0, y_min: 0, x_max: 100, y_max: 100 }
      ];
      fetchMock.mockResponseOnce(JSON.stringify({ objects: mockObjects }));

      const request: DetectRequest = {
        image: mockImageBuffer,
        object: 'dog'
      };
      const result = await client.detect(request);

      expect(result).toEqual({ objects: mockObjects });
      expect(fetchMock).toHaveBeenCalledWith(
        'https://api.moondream.ai/v1/detect',
        expect.objectContaining({
          method: 'POST',
          headers: {
            'X-Moondream-Auth': mockApiKey,
            'Content-Type': 'application/json',
            'User-Agent': expect.stringMatching(/^moondream-node\/\d+\.\d+\.\d+$/)
          }
        })
      );
    });
  });

  describe('encodeImage', () => {
    it('should pass through already encoded images', async () => {
      const result = await (client as any).encodeImage(mockBase64Image);
      expect(result).toBe(mockBase64Image);
    });

    it('should encode image buffers', async () => {
      const result = await (client as any).encodeImage(mockImageBuffer);
      expect(result).toHaveProperty('imageUrl');
      expect(result.imageUrl).toContain('data:image/jpeg;base64,');
    });

    it('should handle encoding errors', async () => {
      const invalidBuffer = Buffer.from('invalid-image-data');
      jest.requireMock('sharp').mockImplementationOnce(() => {
        throw new Error('Invalid image data');
      });

      await expect((client as any).encodeImage(invalidBuffer))
        .rejects
        .toThrow('Failed to convert image to JPEG: Invalid image data');
    });
  });

  describe('streamResponse', () => {
    it('should handle streaming data chunks', async () => {
      const response = new Response(new ReadableStream({
        async start(controller) {
          controller.enqueue(new TextEncoder().encode('data: {"chunk":"test chunk"}\n'));
          controller.enqueue(new TextEncoder().encode('data: {"completed":true}\n'));
          controller.close();
        }
      }));

      const generator = (client as any).streamResponse(response);

      const chunks = [];
      for await (const chunk of generator) {
        chunks.push(chunk);
      }

      expect(chunks).toEqual(['test chunk']);
    });

    it('should handle JSON parsing errors', async () => {
      const response = new Response(new ReadableStream({
        async start(controller) {
          controller.enqueue(new TextEncoder().encode('data: invalid-json\n'));
          controller.close();
        }
      }));

      const generator = (client as any).streamResponse(response);

      await expect(async () => {
        for await (const _ of generator) {
          // consume generator
        }
      }).rejects.toThrow('Failed to parse JSON response from server');
    });

    it('should handle null response body', async () => {
      const response = new Response(null);

      const generator = (client as any).streamResponse(response);

      await expect(async () => {
        for await (const _ of generator) {
          // consume generator
        }
      }).rejects.toThrow('Response body is null');
    });
  });

  describe('point', () => {
    it('should successfully point to objects in an image', async () => {
      const mockPoints = [
        { x: 100, y: 200 },
        { x: 300, y: 400 }
      ];
      fetchMock.mockResponseOnce(JSON.stringify({ points: mockPoints }));

      const request: PointRequest = {
        image: mockImageBuffer,
        object: 'dog'
      };
      const result = await client.point(request);

      expect(result).toEqual({ points: mockPoints });
      expect(fetchMock).toHaveBeenCalledWith(
        'https://api.moondream.ai/v1/point',
        expect.objectContaining({
          method: 'POST',
          headers: {
            'X-Moondream-Auth': mockApiKey,
            'Content-Type': 'application/json',
            'User-Agent': expect.stringMatching(/^moondream-node\/\d+\.\d+\.\d+$/)
          },
          body: expect.stringContaining('dog')
        })
      );
    });
  });
});