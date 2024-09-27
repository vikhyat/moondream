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
1. cd into core directory.

```
cd core
```

2. Create build directory and cd into it.

```
mkdir build && cd build
```

3. Generate makefile with cmake.

```
cmake -DDEBUG_BUILD=OFF -DMOONDREAM_EXE=ON ..
```

4. Build with make.

```
make
```

5. Run executable with data path argument.

```
./moondream_ggml ../../../data/
```

## Static Analysis

1. cd into core directory.

```
cd core
```

2. Run cppcheck.

```
bash scripts/cppcheck.sh
```

## Debugging

To build in debug mode, add the -DDEBUG_MODE=ON flag when calling cmake.

```
cmake -DDEBUG_BUILD=ON ..
```

## CMake Build Options
- `-DDEBUG_BUILD=`, default: `OFF`, description: adds debug symbols when `ON`
- `-DMOONDREAM_SHARED_LIBS=`, default `OFF`, description: builds shared libraries when `ON`
- `-DMOONDREAM_EXE=`, default `off`, description: builds standalone executable instead of library when `ON`

## Work remaining

- [ ] complete pre-processing in `moondream_mmproj_image_preprocess`
- [ ] modify build_clip() in mmproj.cpp in order to implement image/patch feature merging

`moondream_image` is being replaced with `moondream_image_alt` because it uses the new pre-processing pipeline `moondream_mmproj_image_preprocess()`.

since the pytorch implementation uses bilinear downsampling to merge patch features, and ggml doesn't have a bilinear resampling op, your best bet is probably to implement it as a convolution with a handcrafted kernel. failing that you could split the mmproj into two different graphs and do the merge outside of ggml with mmproj_bilinear_downsample()

something important to note is that the patches tensor in build_clip() is marked as an input but it doesn't have its values set anywhere so it causes a bunch of unitialized value accesses (valgrind will yell about this). i didn't fix this because the way the values have to be initialized will most likely change once patch merging is implemented (or patches won't be needed at all). once you finish the image pre-processing and the changes to build_clip() then the image embeddings should be correct and interpretable by the language model.

add disclaimer about using stb_image since it may not be safe to load images...