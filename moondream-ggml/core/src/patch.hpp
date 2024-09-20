#pragma once

#include "image.hpp"

#define MAX_PATCHES 5

struct moondream_patch
{
    float *data;
};

struct moondream_patch_set
{
    int count;
    moondream_patch patches[MAX_PATCHES];
};

void init_patch_set(moondream_patch_set &patch_set);
void free_patch_set(moondream_patch_set &patch_set);

void add_patch(moondream_patch_set &patch_set, moondream_image_alt_f32 &src, int x, int y);

bool create_patches(moondream_image_alt_f32 &img_f32, moondream_patch_set &patch_set);