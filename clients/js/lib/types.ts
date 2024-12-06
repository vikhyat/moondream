export interface APIConfiguration {
  secret: string,
  endpoint: string
}

// X, Y
export type APIPoint = [number, number]

// X1, Y1, X2, Y2
export type BoundingBox = [number, number, number, number]

export interface APIResponse {
  status: number,
  type: string,
  text: string,
  obj?: [APIPoint] | [BoundingBox]
}

export interface APIRequest {
  prompt: string,
  file: string
}

