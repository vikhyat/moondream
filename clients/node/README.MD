# Moondream NodeJS Client Library

Official NodeJS client library for Moondream, a tiny vision language model that can
analyze images and answer questions about them. This client library provides easy
access to Moondream's API endpoints for image analysis.

## Installation

Install the package using npm:

```bash
npm install moondream
```

Or using yarn:

```bash
yarn add moondream
```

## Quick Start

Before using this client library, you'll need an API key to access Moondream's hosted service.
You can get a free API key from [console.moondream.ai](https://console.moondream.ai). Currently
local inference is only available in Python, but Node.js support is coming very soon.

### Cloud

```javascript
import { vl } from "moondream";
import fs from "fs";

// Initialize the client
const model = new vl({
  apiKey: "your-api-key",
});

// Read an image file
const image = fs.readFileSync("path/to/image.jpg");

// Basic usage examples
async function main() {
  // Generate a caption for the image
  const caption = await model.caption(image);
  console.log("Caption:", caption);

  // Ask a question about the image
  const answer = await model.query(image, "What's in this image?");
  console.log("Answer:", answer);

  // Stream the response
  const stream = await model.caption(image, "normal", true);
  for await (const chunk of stream.caption) {
    process.stdout.write(chunk);
  }
}

main();
```

### Local Inference

- Install the `moondream` CLI: `pip install moondream`
- Run the local server: `moondream serve --model <path-to-model>`
- Set the `apiUrl` parameter to the URL of the local server (the default is `http://localhost:3475`)

```javascript
const model = new vl({
  apiUrl: "http://localhost:3475",
});

const image = fs.readFileSync("path/to/image.jpg");

// Basic usage examples
async function main() {
  // Generate a caption for the image
  const caption = await model.caption(image);
  console.log("Caption:", caption);

  // Ask a question about the image
  const answer = await model.query(image, "What's in this image?");
  console.log("Answer:", answer);

  // Stream the response
  const stream = await model.caption(image, "normal", true);
  for await (const chunk of stream.caption) {
    process.stdout.write(chunk);
  }
}

main();
```

## Features

- **caption**: Generate descriptive captions for images
- **query**: Ask questions about image content
- **detect**: Find bounding boxes around objects in images
- **point**: Identify the center location of specified objects in images

## API Reference

### Constructor

```javascript
const model = new vl({
  apiKey: "your-api-key",
});
```

### Methods

#### caption(image, length?, stream?, settings?)

Generate a caption for an image.

```javascript
const result = await model.caption(image, "normal", false);
// or with streaming
const stream = await model.caption(image, "normal", true);
```

#### query(image, question, stream?, settings?)

Ask a question about an image.

```javascript
const result = await model.query(image, "What's in this image?", false);
// or with streaming
const stream = await model.query(image, "What's in this image?", true);
```

#### detect(image, object)

Detect specific objects in an image.

```javascript
const result = await model.detect(image, "car");
```

#### point(image, object)

Get coordinates of specific objects in an image.

```javascript
const result = await model.point(image, "person");
```

### Input Types

- Images can be provided as:
  - Buffer: Raw image data
  - Base64EncodedImage: `{ imageUrl: string }`

### Response Types

All methods return promises that resolve to typed responses:

- CaptionOutput: `{ caption: string | AsyncGenerator }`
- QueryOutput: `{ answer: string | AsyncGenerator }`
- DetectOutput: `{ objects: Array<Object> }`
- PointOutput: `{ points: Array<Point> }`

## Links

- [Website](https://moondream.ai/)
- [Demo](https://moondream.ai/playground)
