// src/__tests__/moondream.test.ts
import { vl, MoondreamVLConfig } from '../moondream';
import fetch from 'node-fetch';
import { Readable } from 'stream';

// Mock node-fetch
jest.mock('node-fetch');
const mockedFetch = fetch as jest.MockedFunction<typeof fetch>;

// Mock sharp
jest.mock('sharp', () => {
  return jest.fn().mockImplementation(() => ({
    metadata: () => Promise.resolve({ width: 1024, height: 768 }),
    resize: function() { return this; },
    toFormat: function() { return this; },
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
    jest.clearAllMocks();
  });

  describe('caption', () => {
    it('should successfully get a caption for an image buffer', async () => {
      const mockResponse = {
        ok: true,
        json: () => Promise.resolve({ caption: 'A beautiful landscape' })
      };
      mockedFetch.mockResolvedValueOnce(mockResponse as any);

      const result = await client.caption(mockImageBuffer);
      
      expect(result).toEqual({ caption: 'A beautiful landscape' });
      expect(mockedFetch).toHaveBeenCalledWith(
        'https://api.moondream.ai/v1/caption',
        expect.objectContaining({
          method: 'POST',
          headers: {
            'X-Moondream-Auth': mockApiKey,
            'Content-Type': 'application/json'
          }
        })
      );
    });

    it('should handle streaming responses', async () => {
      const mockStream = new Readable();
      mockStream._read = () => {};
      const mockResponse = {
        ok: true,
        body: mockStream,
      };
      mockedFetch.mockResolvedValueOnce(mockResponse as any);

      // Simulate streaming data
      setTimeout(() => {
        mockStream.push('data: {"chunk":"test chunk"}\n');
        mockStream.push('data: {"completed":true}\n');
        mockStream.push(null);
      }, 0);

      const result = await client.caption(mockImageBuffer, 'normal', true);
      expect(result.caption).toBeDefined();
      
      const chunks = [];
      for await (const chunk of result.caption as AsyncGenerator<string>) {
        chunks.push(chunk);
      }
      expect(chunks).toEqual(['test chunk']);
    });

    it('should throw an error on API failure', async () => {
      const mockResponse = {
        ok: false,
        status: 400
      };
      mockedFetch.mockResolvedValueOnce(mockResponse as any);

      await expect(client.caption(mockImageBuffer))
        .rejects
        .toThrow('HTTP error! status: 400');
    });
  });

  describe('query', () => {
    it('should successfully query about an image', async () => {
      const mockResponse = {
        ok: true,
        json: () => Promise.resolve({ answer: 'This is a dog' })
      };
      mockedFetch.mockResolvedValueOnce(mockResponse as any);

      const result = await client.query(mockImageBuffer, 'What is in this image?');
      
      expect(result).toEqual({ answer: 'This is a dog' });
      expect(mockedFetch).toHaveBeenCalledWith(
        'https://api.moondream.ai/v1/query',
        expect.objectContaining({
          method: 'POST',
          headers: {
            'X-Moondream-Auth': mockApiKey,
            'Content-Type': 'application/json'
          }
        })
      );
    });

    it('should handle streaming query responses', async () => {
      const mockStream = new Readable();
      mockStream._read = () => {};
      const mockResponse = {
        ok: true,
        body: mockStream,
      };
      mockedFetch.mockResolvedValueOnce(mockResponse as any);

      // Simulate streaming data
      setTimeout(() => {
        mockStream.push('data: {"chunk":"test answer"}\n');
        mockStream.push('data: {"completed":true}\n');
        mockStream.push(null);
      }, 0);

      const result = await client.query(mockImageBuffer, 'What is this?', true);
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
        { bbox: [0, 0, 100, 100], score: 0.95 }
      ];
      const mockResponse = {
        ok: true,
        json: () => Promise.resolve({ objects: mockObjects })
      };
      mockedFetch.mockResolvedValueOnce(mockResponse as any);

      const result = await client.detect(mockImageBuffer, 'dog');
      
      expect(result).toEqual({ objects: mockObjects });
      expect(mockedFetch).toHaveBeenCalledWith(
        'https://api.moondream.ai/v1/detect',
        expect.objectContaining({
          method: 'POST',
          headers: {
            'X-Moondream-Auth': mockApiKey,
            'Content-Type': 'application/json'
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
      const mockStream = new Readable();
      mockStream._read = () => {};
      const mockResponse = {
        body: mockStream
      };

      const generator = (client as any).streamResponse(mockResponse as any);
      
      // Simulate streaming data
      mockStream.push('data: {"chunk":"test chunk"}\n');
      mockStream.push('data: {"completed":true}\n');
      mockStream.push(null);
      
      const chunks = [];
      for await (const chunk of generator) {
        chunks.push(chunk);
      }
      
      expect(chunks).toEqual(['test chunk']);
    });

    it('should handle JSON parsing errors', async () => {
      const mockStream = new Readable();
      mockStream._read = () => {};
      const mockResponse = {
        body: mockStream
      };

      const generator = (client as any).streamResponse(mockResponse as any);
      
      // Push invalid JSON data
      mockStream.push('data: invalid-json\n');
      mockStream.push(null);

      await expect(async () => {
        for await (const _ of generator) {
          // consume generator
        }
      }).rejects.toThrow('Failed to parse JSON response from server');
    });
  });
});