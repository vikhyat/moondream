import moondream from "moondream";

// const a = moondream.add(2, 3);
// console.log(a); // 5

const api_key =
  "XCP.lLS6a3wiJ8-ea3Jkil3xuuU4ZRy35DK8idBA65Jqild8ryqSs2FjNlGy3Bn4k5FHRkqxObEbtSuVBMgHbFubOTf8CyXr5ApjZ5ayDwqZ70sNyMoj9_XqV-isnpl4z2r8pBl8ZIFWfSboQ1D0jGCRNlwGaGkYPPd0dA";

async function test() {
  const md = moondream(api_key);
  const result = await md.query("./dog.png", "What is this?");
  console.log(result.answer);
}

test();
