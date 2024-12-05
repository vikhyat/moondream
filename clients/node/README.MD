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
You can get a free API key from [console.moondream.ai](https://console.moondream.ai).

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
  const caption = await model.caption({
    image: image,
    length: "normal",
    stream: false
  });
  console.log("Caption:", caption);

  // Ask a question about the image
  const answer = await model.query({
    image: image,
    question: "What's in this image?",
    stream: false
  });
  console.log("Answer:", answer);

  // Stream the response
  const stream = await model.caption({
    image: image,
    length: "normal",
    stream: true
  });
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
  const caption = await model.caption({
    image: image,
    length: "normal",
    stream: false
  });
  console.log("Caption:", caption);

  // Ask a question about the image
  const answer = await model.query({
    image: image,
    question: "What's in this image?",
    stream: false
  });
  console.log("Answer:", answer);

  // Stream the response
  const stream = await model.caption({
    image: image,
    length: "normal",
    stream: true
  });
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
// for cloud inference
const model = new vl({
  apiKey: "your-api-key",
});

// or for local inference
const model = new vl({
  apiUrl: "http://localhost:3475",
});
```

### Methods

#### caption({ image: string, length: string, stream?: boolean })

Generate a caption for an image.

```javascript
const result = await model.caption({
  image: image,
  length: "normal",
  stream: false
});

// or with streaming
const stream = await model.caption({
  image: image,
  length: "normal",
  stream: true
});
```

#### query({ image: string, question: string, stream?: boolean })

Ask a question about an image.

```javascript
const result = await model.query({
  image: image,
  question: "What's in this image?",
  stream: false
});

// or with streaming
const stream = await model.query({
  image: image,
  question: "What's in this image?",
  stream: true
});
```

#### detect({ image: string, object: string })

Detect specific objects in an image.

```javascript
const result = await model.detect({
  image: image,
  object: "car"
});
```

#### point({ image: string, object: string })

Get coordinates of specific objects in an image.

```javascript
const result = await model.point({
  image: image,
  object: "person"
});
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
