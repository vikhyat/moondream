#pragma once

#include <cstdint>

// NOTE: this is a temporary name. It will replace moondream_image and be renamed when it does.
struct moondream_image_alt_u8
{
    int width = 0;
    int height = 0;
    int n_channels = 0;
    unsigned char *data = nullptr;
};

struct moondream_image_alt_f32
{
    int width = 0;
    int height = 0;
    int n_channels = 0;
    float *data = nullptr;
};

bool load_img_to_u8(const char *path, moondream_image_alt_u8 &image);

void init_moondream_image_alt_u8(moondream_image_alt_u8 &image, int width, int height, unsigned char *data);
void free_moondream_image_alt_u8(moondream_image_alt_u8 &image);
void free_moondream_image_alt_f32(moondream_image_alt_f32 &image);

bool normalize_image_u8_to_f32(moondream_image_alt_u8 *src, moondream_image_alt_f32 *dst, const float mean[3], const float std[3]);
bool image_u8_to_f32(moondream_image_alt_u8 *src, moondream_image_alt_f32 *dst);
