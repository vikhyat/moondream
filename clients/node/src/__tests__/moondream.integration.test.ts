import { vl, MoondreamVLConfig } from '../moondream';

import fs from 'fs/promises';
import path from 'path';
import dotenv from 'dotenv';

dotenv.config();

const apiKey = process.env.MOONDREAM_API_KEY;
if (!apiKey) {
    throw new Error('MOONDREAM_API_KEY environment variable is required');
}

describe('MoondreamClient Integration Tests', () => {
    let client: vl;
    let imageBuffer: Buffer;

    const moondreamConfig: MoondreamVLConfig = {
        apiKey: apiKey
    };

    beforeAll(async () => {
        client = new vl(moondreamConfig);
        // Load test image
        imageBuffer = await fs.readFile(path.join(__dirname, '../__fixtures__/demo-1.jpg'));
    });

    describe('caption', () => {
        it('should get a caption for a real image', async () => {
            const result = await client.caption(imageBuffer);
            expect(result.caption).toBeDefined();
            expect(typeof result.caption).toBe('string');
            console.log('Caption:', result.caption);
        }, 10000); // Increased timeout for API call

        it('should stream captions for a real image', async () => {
            const result = await client.caption(imageBuffer, 'normal', true);
            const chunks: string[] = [];

            for await (const chunk of result.caption) {
                chunks.push(chunk);
            }

            const finalCaption = chunks.join('');
            expect(finalCaption).toBeTruthy();
            expect(chunks.length).toBeGreaterThan(0);
            console.log('Streamed caption:', finalCaption);
        }, 10000);
    });

    describe('caption-no-stream', () => {
        it('should get a caption for a real image', async () => {
            const result = await client.caption(imageBuffer, 'normal', false);
            expect(result.caption).toBeDefined();
            expect(typeof result.caption).toBe('string');
            console.log('Caption:', result.caption);
            expect((result.caption as string).length).toBeGreaterThan(0);
        }, 10000);
    });

    describe('query', () => {
        it('should answer questions about a real image', async () => {
            const question = "What colors are present in this image?";
            const result = await client.query(imageBuffer, question);

            expect(result.answer).toBeDefined();
            expect(typeof result.answer).toBe('string');
            console.log('Question:', question);
            console.log('Answer:', result.answer);
        }, 10000);

        it('should stream answers about a real image', async () => {
            const question = "What is the character doing?";
            const result = await client.query(imageBuffer, question, true);
            const chunks: string[] = [];

            for await (const chunk of result.answer) {
                chunks.push(chunk);
            }

            const finalAnswer = chunks.join('');
            expect(finalAnswer).toBeTruthy();
            expect(chunks.length).toBeGreaterThan(0);
            console.log('Question:', question);
            console.log('Streamed answer:', finalAnswer);
        }, 10000);
    });

    describe('query-no-stream', () => {
        it('should answer questions about a real image', async () => {
            const question = "What colors are present in this image?";
            const result = await client.query(imageBuffer, question, false);
            expect(result.answer).toBeDefined();
            expect(typeof result.answer).toBe('string');
            console.log('Answer:', result.answer);
        });
    });

    describe('detect', () => {
        it('should detect objects in a real image', async () => {
            const objectToDetect = "burger"; // Adjust based on what's in your test image
            const result = await client.detect(imageBuffer, objectToDetect);

            expect(result.objects).toBeDefined();
            expect(Array.isArray(result.objects)).toBe(true);
            console.log('Detected objects:', result.objects);
        }, 10000);
    });
});