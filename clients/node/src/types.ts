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
}

/**
 * Request structure for image caption requests
 */
export interface CaptionRequest {
  image: Buffer | Base64EncodedImage;
  length?: Length;
  stream?: boolean;
  settings?: SamplingSettings;
}
/**
 * Response structure for image caption requests
 */
export interface CaptionOutput {
  caption: string | AsyncGenerator<string, void, unknown>;
}

/**
 * Request structure for image query requests
 */
export interface QueryRequest {
  image: Buffer | Base64EncodedImage;
  question: string;
  stream?: boolean;
  settings?: SamplingSettings;
}
/**
 * Response structure for image query requests
 */
export interface QueryOutput {
  answer: string | AsyncGenerator<string, void, unknown>;
}

/**
 * Request structure for object detection requests
 */
export interface DetectRequest {
  image: Buffer | Base64EncodedImage;
  object: string;
}
/**
 * Response structure for object detection requests
 */
export interface DetectOutput {
  objects: DetectedObject[];
}

/**
 * Object detection result
 */
export interface DetectedObject {
  x_min: number;
  y_min: number;
  x_max: number;
  y_max: number;
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

/**
 * Pointing request structure
 */
export interface PointRequest {
  image: Buffer | Base64EncodedImage;
  object: string;
}
/**
 * Point coordinates for object location
 */
export interface Point {
  x: number;
  y: number;
}

/**
 * Response structure for point requests
 */
export interface PointOutput {
  points: Point[];
}