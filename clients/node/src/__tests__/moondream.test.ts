import { vl, MoondreamVLConfig } from '../moondream';
import { CaptionRequest, DetectRequest, PointRequest, QueryRequest } from '../types';
import http from 'http';
import https from 'https';
import { EventEmitter } from 'events';

// Extend EventEmitter to include http/https specific properties
interface MockResponse extends EventEmitter {
  statusCode: number;
  [Symbol.asyncIterator]?: () => AsyncGenerator<Buffer, void, unknown>;
}

interface MockRequest extends EventEmitter {
  write: jest.Mock;
  end: jest.Mock;
}

// Mock sharp
jest.mock('sharp', () => {
  return jest.fn().mockImplementation(() => ({
    metadata: () => Promise.resolve({ width: 1024, height: 768 }),
    resize: function () { return this; },
    toFormat: function () { return this; },
    toBuffer: () => Promise.resolve(Buffer.from('mock-image-data')),
  }));
});

// Mock http and https modules
jest.mock('http');
jest.mock('https');

describe('MoondreamClient', () => {
  let client: vl;
  const mockApiKey = 'test-api-key';
  const mockImageBuffer = Buffer.from('mock-image-data');
  const mockBase64Image = {
    imageUrl: 'data:image/jpeg;base64,mock-image-data'
  };

  // Helper to create mock response
  function createMockResponse(data: any, status = 200): MockResponse {
    const response = new EventEmitter() as MockResponse;
    response.statusCode = status;
    
    process.nextTick(() => {
      response.emit('data', Buffer.from(JSON.stringify(data)));
      response.emit('end');
    });

    return response;
  }

  // Helper to create mock streaming response
  function createMockStreamingResponse(chunks: string[], status = 200): MockResponse {
    const response = new EventEmitter() as MockResponse;
    response.statusCode = status;
    
    // Emit chunks immediately
    for (const chunk of chunks) {
      response.emit('data', Buffer.from(chunk));
    }
    response.emit('end');

    return response;
  }

  // Helper to create mock request
  function createMockRequest(): MockRequest {
    const request = new EventEmitter() as MockRequest;
    request.write = jest.fn();
    request.end = jest.fn();
    return request;
  }

  beforeEach(() => {
    const moondreamConfig: MoondreamVLConfig = {
      apiKey: mockApiKey
    };
    client = new vl(moondreamConfig);

    // Reset mocks
    (https.request as jest.Mock).mockReset();
    (http.request as jest.Mock).mockReset();
  });

  describe('caption', () => {
    it('should successfully get a caption for an image buffer', async () => {
      const mockReq = createMockRequest();
      const mockRes = createMockResponse({ caption: 'A beautiful landscape' });
      
      (https.request as jest.Mock).mockImplementation((url, options, callback) => {
        callback(mockRes);
        return mockReq;
      });

      const request: CaptionRequest = {
        image: mockImageBuffer,
        length: 'normal',
        stream: false
      };
      const result = await client.caption(request);

      expect(result).toEqual({ caption: 'A beautiful landscape' });
      expect(https.request).toHaveBeenCalledWith(
        expect.any(URL),
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            'X-Moondream-Auth': mockApiKey,
            'Content-Type': 'application/json',
            'User-Agent': expect.stringMatching(/^moondream-node\/\d+\.\d+\.\d+$/)
          })
        }),
        expect.any(Function)
      );
    });

    it('should handle streaming responses', async () => {
      const mockReq = createMockRequest();
      const mockRes = new EventEmitter() as MockResponse;
      mockRes.statusCode = 200;
      mockRes[Symbol.asyncIterator] = async function* () {
        yield Buffer.from('data: {"chunk":"test chunk"}\n');
        yield Buffer.from('data: {"completed":true}\n');
      };
      
      (https.request as jest.Mock).mockImplementation((url, options, callback) => {
        callback(mockRes);
        return mockReq;
      });

      const request: CaptionRequest = {
        image: mockImageBuffer,
        length: 'normal',
        stream: true
      };
      const result = await client.caption(request);
      expect(result.caption).toBeDefined();

      const chunks: string[] = [];
      for await (const chunk of result.caption as AsyncGenerator<string>) {
        chunks.push(chunk);
      }

      expect(chunks).toEqual(['test chunk']);
    });

    it('should throw an error on API failure', async () => {
      const mockReq = createMockRequest();
      const mockRes = createMockResponse({}, 400);
      
      (https.request as jest.Mock).mockImplementation((url, options, callback) => {
        callback(mockRes);
        return mockReq;
      });

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
      const mockReq = createMockRequest();
      const mockRes = createMockResponse({ answer: 'This is a dog' });
      
      (https.request as jest.Mock).mockImplementation((url, options, callback) => {
        callback(mockRes);
        return mockReq;
      });

      const request: QueryRequest = {
        image: mockImageBuffer,
        question: 'What is in this image?'
      };
      const result = await client.query(request);

      expect(result).toEqual({ answer: 'This is a dog' });
      expect(https.request).toHaveBeenCalledWith(
        expect.any(URL),
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            'X-Moondream-Auth': mockApiKey,
            'Content-Type': 'application/json',
            'User-Agent': expect.stringMatching(/^moondream-node\/\d+\.\d+\.\d+$/)
          })
        }),
        expect.any(Function)
      );
    });

    it('should handle streaming query responses', async () => {
      const mockReq = createMockRequest();
      const mockRes = new EventEmitter() as MockResponse;
      mockRes.statusCode = 200;
      mockRes[Symbol.asyncIterator] = async function* () {
        yield Buffer.from('data: {"chunk":"test answer"}\n');
        yield Buffer.from('data: {"completed":true}\n');
      };
      
      (https.request as jest.Mock).mockImplementation((url, options, callback) => {
        callback(mockRes);
        return mockReq;
      });

      const request: QueryRequest = {
        image: mockImageBuffer,
        question: 'What is this?',
        stream: true
      };
      const result = await client.query(request);
      expect(result.answer).toBeDefined();

      const chunks: string[] = [];
      for await (const chunk of result.answer as AsyncGenerator<string>) {
        chunks.push(chunk);
      }

      expect(chunks).toEqual(['test answer']);
    });
  });

  describe('detect', () => {
    it('should successfully detect objects in an image', async () => {
      const mockReq = createMockRequest();
      const mockObjects = [
        { x_min: 0, y_min: 0, x_max: 100, y_max: 100 }
      ];
      const mockRes = createMockResponse({ objects: mockObjects });
      
      (https.request as jest.Mock).mockImplementation((url, options, callback) => {
        callback(mockRes);
        return mockReq;
      });

      const request: DetectRequest = {
        image: mockImageBuffer,
        object: 'dog'
      };
      const result = await client.detect(request);

      expect(result).toEqual({ objects: mockObjects });
      expect(https.request).toHaveBeenCalledWith(
        expect.any(URL),
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            'X-Moondream-Auth': mockApiKey,
            'Content-Type': 'application/json',
            'User-Agent': expect.stringMatching(/^moondream-node\/\d+\.\d+\.\d+$/)
          })
        }),
        expect.any(Function)
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

  describe('point', () => {
    it('should successfully point to objects in an image', async () => {
      const mockReq = createMockRequest();
      const mockPoints = [
        { x: 100, y: 200 },
        { x: 300, y: 400 }
      ];
      const mockRes = createMockResponse({ points: mockPoints });
      
      (https.request as jest.Mock).mockImplementation((url, options, callback) => {
        callback(mockRes);
        return mockReq;
      });

      const request: PointRequest = {
        image: mockImageBuffer,
        object: 'dog'
      };
      const result = await client.point(request);

      expect(result).toEqual({ points: mockPoints });
      expect(https.request).toHaveBeenCalledWith(
        expect.any(URL),
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            'X-Moondream-Auth': mockApiKey,
            'Content-Type': 'application/json',
            'User-Agent': expect.stringMatching(/^moondream-node\/\d+\.\d+\.\d+$/)
          })
        }),
        expect.any(Function)
      );
    });
  });

  describe('HTTP support', () => {
    it('should use http module for http URLs', async () => {
      const mockReq = createMockRequest();
      const mockRes = createMockResponse({ caption: 'A beautiful landscape' });
      
      (http.request as jest.Mock).mockImplementation((url, options, callback) => {
        callback(mockRes);
        return mockReq;
      });

      const httpClient = new vl({
        apiKey: mockApiKey,
        apiUrl: 'http://api.example.com'
      });

      const request: CaptionRequest = {
        image: mockImageBuffer,
        length: 'normal',
        stream: false
      };
      const result = await httpClient.caption(request);

      expect(result).toEqual({ caption: 'A beautiful landscape' });
      expect(http.request).toHaveBeenCalled();
      expect(https.request).not.toHaveBeenCalled();
    });
  });
});