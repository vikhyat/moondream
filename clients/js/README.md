# `moondream`

The Javascript/Typescript/Node library to access the Moondream Cloud API.

## Quick Start

To use the moondream API client, you will need to set up an account with Moondream
at https://moondream.ai

You will need to create a client key to access the moondream API. You will additionally
need to create a capability.

This process takes a few minutes to use the core capabilities out of the box. You can
additionally improve the API for your desired capability through the available tools
on the Moondream console.

#### Javascript:

```
import { MoondreamAPI } from "moondream";

const md = new MoondreamAPI({
  secret: "<Your Moondream Console Secret>",
  endpoint: "<Your Moondream Console endpoint>",
});

md.request({prompt: "Describe this image", file: "img_file.jpg"}).then((rsp) => {
  // The response body will vary by inference type. See documentation.
  console.log(rsp.text);
});
```

#### Typescript

```
import { MoondreamAPI, APIConfiguration, APIRequest } from "moondream";

const cfg : APIConfiguration = {
  secret: "<Your Moondream Console Secret>",
  endpoint: "<Your Moondream Console endpoint>",
}

const req : APIRequest = {
  prompt: "Describe this image",
  file: "img_file.jpg"
}

const api = new MoondreamAPI(cfg);

// Different core capabilities may have variant return types, see documentation.
api.request(req).then((response) =>
  console.log(response.text)
);
```

## Classes & Types

Currently, all Moondream access is via the API only; local inference is TBD.

The MoondreamAPI class should be imported from moondream.

#### MoondreamAPI

##### `constructor(APIConfiguration)`

returns ready API client.

##### `async request(APIRequest) : APIResponse`

takes a valid api request object;
returns (a promise for) a valid API response object
errors are signalled via the status code in the response object

#### APIConfiguration

```
{
  secret: "<Your Moondream Console Secret>",
  endpoint: "<Your Moondream Console endpoint>",
}
```

#### APIRequest

```
{
  prompt: "String prompt; optional or unnecessary for some capabilities",
  file: "A valid local file of type jpg, webp, or png."
}
```

### APIResponse

```

{
  type: Enum { caption, count, classify, detect, ocr, vqa },
  status: Enum {
    200 -- success,
    400 -- bad parameters,
    401 -- invalid authentication credentials,
    413 -- image too large: the client should be resizing this for you, so if
           get this there may be a problem with the moondream service
    500 -- server error
  },
  text: "moondream's response to the image, by the capability defined",
  obj: Some capabilities return a structured object.
}

```
