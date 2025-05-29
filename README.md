# 🌔 moondream

a tiny vision language model that kicks ass and runs anywhere

[Website](https://moondream.ai/) | [Demo](https://moondream.ai/playground)

## Examples

| Image                  | Example                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ![](assets/demo-1.jpg) | **What is the girl doing?**<br>The girl is sitting at a table and eating a large hamburger.<br><br>**What color is the girl's hair?**<br>The girl's hair is white.                                                                                                                                                                                                                                                                                                                                                                                                    |
| ![](assets/demo-2.jpg) | **What is this?**<br>This is a computer server rack, which is a device used to store and manage multiple computer servers. The rack is filled with various computer servers, each with their own dedicated space and power supply. The servers are connected to the rack via multiple cables, indicating that they are part of a larger system. The rack is placed on a carpeted floor, and there is a couch nearby, suggesting that the setup is in a living or entertainment area.<br><br>**What is behind the stand?**<br>Behind the stand, there is a brick wall. |

## About

Moondream is a highly efficient open-source vision language model that combines powerful image understanding capabilities with a remarkably small footprint. It's designed to be versatile and accessible, capable of running on a wide range of devices and platforms.

The project offers two model variants:

- **Moondream 2B**: The primary model with 2 billion parameters, offering robust performance for general-purpose image understanding tasks including captioning, visual question answering, and object detection.
- **Moondream 0.5B**: A compact 500 million parameter model specifically optimized as a distillation target for edge devices, enabling efficient deployment on resource-constrained hardware while maintaining impressive capabilities.

## How to use

Moondream can be run locally, or in the cloud. Please refer to the [Getting Started](https://moondream.ai/c/docs/quickstart) page for details.

## Special thanks

* [Modal](https://modal.com/?utm_source=github&utm_medium=github&utm_campaign=moondream) - Modal lets you run jobs in the cloud, by just writing a few lines of Python. Here's an [example of how to run Moondream on Modal](https://github.com/m87-labs/moondream-examples/tree/main/quickstart/modal).
