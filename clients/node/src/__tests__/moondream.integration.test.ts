import { vl, MoondreamVLConfig } from '../moondream';

import fs from 'fs/promises';
import path from 'path';
import dotenv from 'dotenv';
import { Base64EncodedImage, CaptionRequest, DetectRequest, PointRequest, QueryRequest } from '../types';

dotenv.config();

const apiKey = process.env.MOONDREAM_API_KEY;
if (!apiKey) {
    throw new Error('MOONDREAM_API_KEY environment variable is required');
}

describe('MoondreamClient Integration Tests', () => {
    let client: vl;
    let imageBuffer: Base64EncodedImage;

    const moondreamConfig: MoondreamVLConfig = {
        apiKey: apiKey
    };

    beforeAll(async () => {
        client = new vl(moondreamConfig);
        // Load test image and convert to base64
        const rawBuffer = await fs.readFile(path.join(__dirname, '../../../../assets/demo-1.jpg'));
        imageBuffer = {
            imageUrl: rawBuffer.toString('base64')
        };
    });

    describe('caption', () => {
        it('should get a caption for a real image', async () => {
            
            const request: CaptionRequest = {
                image: imageBuffer,
                length: 'normal',
                stream: false
            };
            const result = await client.caption(request);
            expect(result.caption).toBeDefined();
            expect(typeof result.caption).toBe('string');
            console.log('Caption:', result.caption);
        }, 100000); // Increased timeout for API call

        it('should stream captions for a real image', async () => {
            
            const request: CaptionRequest = {
                image: imageBuffer,
                length: 'normal',
                stream: true
            };
            const result = await client.caption({
                image: imageBuffer,
                length: 'short',
                stream: true
            });

            // Handle both streaming and non-streaming responses
            if (typeof result.caption === 'string') {
                expect(result.caption).toBeTruthy();
                console.log('Caption (non-streamed):', result.caption);
            } else {
                const chunks: string[] = [];
                for await (const chunk of result.caption) {
                    chunks.push(chunk);
                }
                const finalCaption = chunks.join('');
                expect(finalCaption).toBeTruthy();
                expect(chunks.length).toBeGreaterThan(0);
                console.log('Streamed caption:', finalCaption);
            }
        }, 100000);
    });

    describe('caption-no-stream', () => {
        it('should get a caption for a real image', async () => {
            
            const request: CaptionRequest = {
                image: imageBuffer,
                length: 'short',
                stream: false
            };
            const result = await client.caption(request);
            
            expect(result.caption).toBeDefined();
            expect(typeof result.caption).toBe('string');
            console.log('Caption:', result.caption);
            expect((result.caption as string).length).toBeGreaterThan(0);
        }, 100000);
    });

    describe('query', () => {
        it('should answer questions about a real image', async () => {
            
            const question = "What colors are present in this image?";
            const request: QueryRequest = {
                image: imageBuffer,
                question: question,
                stream: false
            };
            const result = await client.query(request);

            expect(result.answer).toBeDefined();
            expect(typeof result.answer).toBe('string');
            console.log('Question:', question);
            console.log('Answer:', result.answer);
        }, 100000);

        it('should stream answers about a real image', async () => {
            
            const question = "What is the character doing?";
            const request: QueryRequest = {
                image: imageBuffer,
                question: question,
                stream: true
            };
            const result = await client.query(request);

            // Handle both streaming and non-streaming responses
            if (typeof result.answer === 'string') {
                expect(result.answer).toBeTruthy();
                console.log('Question:', question);
                console.log('Answer (non-streamed):', result.answer);
            } else {
                const chunks: string[] = [];
                for await (const chunk of result.answer) {
                    chunks.push(chunk);
                }
                const finalAnswer = chunks.join('');
                expect(finalAnswer).toBeTruthy();
                expect(chunks.length).toBeGreaterThan(0);
                console.log('Question:', question);
                console.log('Streamed answer:', finalAnswer);
            }
        }, 100000);
    });

    describe('query-no-stream', () => {
        it('should answer questions about a real image', async () => {
            
            const question = "What colors are present in this image?";
            const request: QueryRequest = {
                image: imageBuffer,
                question: question,
                stream: false
            };
            const result = await client.query(request);
            expect(result.answer).toBeDefined();
            expect(typeof result.answer).toBe('string');
            console.log('Answer:', result.answer);
        }, 100000);
    });

    describe('detect', () => {
        it('should detect objects in a real image', async () => {
            
            const objectToDetect = "burger";
            const request: DetectRequest = {
                image: imageBuffer,
                object: objectToDetect,
            };
            const result = await client.detect(request);

            expect(result.objects).toBeDefined();
            expect(Array.isArray(result.objects)).toBe(true);
            console.log('Detected objects:', result.objects);
        }, 100000);
    });

    describe('point', () => {
        it('should point to objects in a real image', async () => {
            
            const objectToPoint = "burger";
            const request: PointRequest = {
                image: imageBuffer,
                object: objectToPoint,
            };
            const result = await client.point(request);

            expect(result.points).toBeDefined();
            expect(Array.isArray(result.points)).toBe(true);
            result.points.forEach(point => {
                expect(point).toHaveProperty('x');
                expect(point).toHaveProperty('y');
                expect(typeof point.x).toBe('number');
                expect(typeof point.y).toBe('number');
            });
            console.log('Pointed locations:', result.points);
        }, 100000);
    });
});
