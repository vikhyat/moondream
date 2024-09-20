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
cmake -DDEBUG_BUILD=OFF ..
```

4. Build with make.

```
make
```

5. Run executable with data path argument.

```
./moondream_ggml_exe ../../../data/
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

### Work remaining

- [ ] complete pre-processing in `moondream_mmproj_image_preprocess`
- [ ] modify build_clip() in mmproj.cpp in order to implement image/patch feature merging

`moondream_image` is being replaced with `moondream_image_alt` because it uses the new pre-processing pipeline `moondream_mmproj_image_preprocess()`.

since the pytorch implementation uses bilinear downsampling to merge patch features, and ggml doesn't have a bilinear resampling op, your best bet is probably to implement it as a convolution with a handcrafted kernel. failing that you could split the mmproj into two different graphs and do the merge outside of ggml with mmproj_bilinear_downsample()

something important to note is that the patches tensor in build_clip() is marked as an input but it doesn't have its values set anywhere so it causes a bunch of unitialized value accesses (valgrind will yell about this). i didn't fix this because the way the values have to be initialized will most likely change once patch merging is implemented (or patches won't be needed at all). once you finish the image pre-processing and the changes to build_clip() then the image embeddings should be correct and interpretable by the language model.

add disclaimer about using stb_image since it may not be safe to load images...

##### original blurb

08/04/2024 9:07 PM hey, unfortunately i've been sick the last few days and wasn't able to get everything done, but i did my best to cleanup the code and added some comments to show where i was going with it. the most confusing part of the codebase right now is probably the moondream_image / moondream_image_alt structs and their associated functions. moondream_image is currently being used as input for the mmproj but i was in the process of replacing it with moondream_image_alt since that one uses the new pre-processing pipeline. moondream_mmproj_image_preprocess() is the function for the new pre-processing pipeline, i left a TODO list there for everything it still needs. once the pre-processing is done you'll have to modify build_clip() in mmproj.cpp in order to implement image/patch feature merging. since the pytorch implementation uses bilinear downsampling to merge patch features, and ggml doesn't have a bilinear resampling op, your best bet is probably to implement it as a convolution with a handcrafted kernel. failing that you could split the mmproj into two different graphs and do the merge outside of ggml with mmproj_bilinear_downsample() . something important to note is that the patches tensor in build_clip() is marked as an input but it doesn't have its values set anywhere so it causes a bunch of unitialized value accesses (valgrind will yell about this). i didn't fix this because the way the values have to be initialized will most likely change once patch merging is implemented (or patches won't be needed at all). once you finish the image pre-processing and the changes to build_clip() then the image embeddings should be correct and interpretable by the language model. also some notes about the stb headers. first, stb_image_write.h is only there because i needed an easy way to view the pre-processed images and make sure they looked correct. it's useful for dev but doesn't have to be shipped. second, stb_image.h was originally only included in the executable build, but i added it to the library build since i needed some way for the API to access images and didn't have time to implement passing pre-loaded image data. this setup doesn't require any other dependencies on the part of the user (e.g. PIL), so it may be useful for quick demos, but I'm not sure how secure stb_image. h is against malicious images. because of this, if you decide to keep it, a warning to the user before they toad an image might be a good idea. i think that's about it. let me know if there's anything i can clarify and best of luck with the upcoming launch
