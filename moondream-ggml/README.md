# moondream-ggml

Moondream inference with GGML (work in progress).

## Dependencies
- cmake version >= 3.0
- c++11 capable compiler
- ggml (comes packaged in this repo)

## Instructions
0. Make sure there is a directory called `data` one level above this one (i.e. `../data/`),
and make sure it contains the moondream2 gguf files
(`moondream2-text-model-f16.gguf` and `moondream2-mmproj-f16.gguf`, obtained 
[here](https://huggingface.co/vikhyatk/moondream2/tree/fa8398d264205ac3890b62e97d3c588268ed9ec4)).
Currently these file names are hardcoded for simplicity during development.
1. Create build directory and cd into it
```
mkdir build && cd build
```
2. Generate makefile with cmake
```
cmake ..
```
3. Execute build with make
```
make
```
4. Run executable with data path argument
```
./moondream_ggml ../../data/
```
