import { vl } from "moondream";
import fs from "fs";

// this will run it against a local Moondream Server
const model = new vl({ endpoint: "http://localhost:2020/v1" });

// ...uncomment this line if you prefer to run it against Moondream Cloud
// const model = new vl({ apiKey: "<your-api-key>" });

// read the image we're going to use
const image = fs.readFileSync("../images/frieren.jpg");

async function main() {
  // let's generate a caption for the image
  const captionResponse = await model.caption({
    image,
    length: "normal",
    stream: false,
  });
  console.log(captionResponse);
}

main();
