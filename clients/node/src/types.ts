/**
 * Base interface for encoded images
 */
export interface Base64EncodedImage {
  imageUrl: string;
}

/**
 * Length options for caption generation
 */
export type Length = 'normal' | 'short';

/**
 * Settings for controlling the model's generation behavior
 */
export interface SamplingSettings {
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
  repetitionPenalty?: number;
  seed?: number;
}

/**
 * Response structure for image caption requests
 */
export interface CaptionOutput {
  caption: string | AsyncGenerator<string, void, unknown>;
}

/**
 * Response structure for image query requests
 */
export interface QueryOutput {
  answer: string | AsyncGenerator<string, void, unknown>;
}

/**
 * Bounding box coordinates [x1, y1, x2, y2]
 */
export type BoundingBox = [number, number, number, number];

/**
 * Object detection result
 */
export interface DetectedObject {
  bbox: BoundingBox;
  score: number;
  label?: string;
}

/**
 * Response structure for object detection requests
 */
export interface DetectOutput {
  objects: DetectedObject[];
}

/**
 * Error response from the API
 */
export interface ApiError {
  error: {
    message: string;
    code?: string;
    details?: unknown;
  };
}

/**
 * Configuration options for the client
 */
export interface ClientConfig {
  apiKey: string;
  apiUrl?: string;
  timeout?: number;
  retries?: number;
}

/**
 * API response for streaming requests
 */
export interface StreamResponse {
  chunk?: string;
  completed?: boolean;
  error?: string;
}

/**
 * Options for image processing
 */
export interface ImageProcessingOptions {
  maxSize?: number;
  quality?: number;
  format?: 'jpeg' | 'png';
}

/**
 * Common response type for all API endpoints
 */
export type ApiResponse<T> = {
  success: boolean;
  data?: T;
  error?: ApiError;
  timestamp?: string;
  requestId?: string;
}