#include <stdlib.h>
#include <cassert>

#include "helpers.hpp"
#include "patch.hpp"

bool init_patch(moondream_patch &patch)
{
    int patch_size_in_bytes = sizeof(float) * MOONDREAM_IMAGE_PATCH_SIDE_LENGTH * MOONDREAM_IMAGE_PATCH_SIDE_LENGTH * MOONDREAM_N_IMAGE_CHANNELS;
    patch.data = (float *)malloc(patch_size_in_bytes);
    if (!patch.data)
    {
        printf("failed to allocate memory for moondream_patch data\n");
        return false;
    }
    return true;
}

bool free_patch(moondream_patch &patch)
{
    if (patch.data)
    {
        free(patch.data);
        patch.data = nullptr;
    }
    return true;
}

void init_patch_set(moondream_patch_set &patch_set)
{
    patch_set.count = 0;
    for (int i = 0; i < MAX_PATCHES; ++i)
    {
        init_patch(patch_set.patches[i]);
    }
}

void free_patch_set(moondream_patch_set &patch_set)
{
    for (int i = 0; i < MAX_PATCHES; ++i)
    {
        free_patch(patch_set.patches[i]);
    }
}

// Copy a patch from an image to a patch struct.
void add_patch(moondream_patch_set &patch_set, moondream_image_alt_f32 &src, int x, int y)
{
    assert(patch_set.count < MAX_PATCHES);
    moondream_patch dst = patch_set.patches[patch_set.count];
    patch_set.count++;

    for (int i = 0; i < MOONDREAM_IMAGE_PATCH_SIDE_LENGTH; ++i)
    {
        for (int j = 0; j < MOONDREAM_IMAGE_PATCH_SIDE_LENGTH; ++j)
        {
            for (int c = 0; c < MOONDREAM_N_IMAGE_CHANNELS; ++c)
            {
                int src_idx = ((y + i) * src.width + (x + j)) * MOONDREAM_N_IMAGE_CHANNELS + c;
                int dst_idx = (i * MOONDREAM_IMAGE_PATCH_SIDE_LENGTH + j) * MOONDREAM_N_IMAGE_CHANNELS + c;
                dst.data[dst_idx] = src.data[src_idx];
            }
        }
    }
}

bool create_patches(moondream_image_alt_f32 &img_f32, moondream_patch_set &patch_set)
{
    // Reference python code:
    //
    // def create_patches(image, patch_size=(378, 378)):
    //     assert image.dim() == 3, "Image must be in CHW format"
    //     _, height, width = image.shape  # Channels, Height, Width
    //     patch_height, patch_width = patch_size
    //     if height == patch_height and width == patch_width:
    //         return []
    //     # Iterate over the image and create patches
    //     patches = []
    //     for i in range(0, height, patch_height):
    //         row_patches = []
    //         for j in range(0, width, patch_width):
    //             patch = image[:, i : i + patch_height, j : j + patch_width]
    //             row_patches.append(patch)
    //         patches.append(torch.stack(row_patches))
    //     return patches
    //

    if (MOONDREAM_IMAGE_PATCH_SIDE_LENGTH == img_f32.width && MOONDREAM_IMAGE_PATCH_SIDE_LENGTH == img_f32.height)
    {
        patch_set.count = 0;
        return true;
    }

    // const int elements_per_patch = MOONDREAM_IMAGE_PATCH_SIDE_LENGTH * MOONDREAM_IMAGE_PATCH_SIDE_LENGTH;
    // batch.patch_count = (img_f32.width * img_f32.height) / elements_per_patch;
    // batch.patches = (float *)malloc(sizeof(float) * batch.patch_count * elements_per_patch * MOONDREAM_N_IMAGE_CHANNELS);
    int patch_index = 0;
    for (int i = 0; i < img_f32.height; i += MOONDREAM_IMAGE_PATCH_SIDE_LENGTH)
    {
        for (int j = 0; j < img_f32.width; i += MOONDREAM_IMAGE_PATCH_SIDE_LENGTH)
        {
            // init_patch(patch_set.patches[patch_index]);
            // make_patch(img_f32, i, j, patch_set.patches[patch_index]);
            add_patch(patch_set, img_f32, i, j);
            patch_set.count++;
        }
    }
    return true;
}
