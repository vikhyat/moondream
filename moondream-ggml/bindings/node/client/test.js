import moondream from "moondream";

async function test_cloud(api_key) {
  const md = moondream(api_key);

  const query = await md.query("./dog.png", "What is this?");
  console.log("answer ======================");
  console.log(query.answer);

  const find = await md.find("./dog.png", "nose");
  console.log("find ======================");
  console.log(find.answer);

  const describe = await md.describe("./dog.png", "");
  console.log("describe ======================");
  console.log(describe.answer);
}

async function test_local() {
  const arg = {
    text_model: "/Users/jasonallen/Downloads/moondream2-text-model-f16.gguf",
    mmproj: "/Users/jasonallen/Downloads/moondream2-mmproj-f16.gguf",
  };
  const md = moondream(arg);
  const query = await md.query("./dog.png", "What is this?");
  console.log("answer ======================");
  console.log(query);
}

const k = process.argv.at(-1);
// test_cloud(k);
test_local();
