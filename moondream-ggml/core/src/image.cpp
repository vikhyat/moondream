#include <stdlib.h>
#include <cassert>

#include "stb_image.h"
#include "image.hpp"
#include "helpers.hpp"

void init_moondream_image_alt_u8(
    moondream_image_alt_u8 & image, int width, int height, unsigned char * data
) {
    image.width = width;
    image.height = height;
    image.n_channels = MOONDREAM_N_IMAGE_CHANNELS;
    image.data = data;
}

void free_moondream_image_alt_u8(moondream_image_alt_u8 & image) {
    if (image.data) {
        free(image.data);
        image.data = nullptr;
    }
}

void free_moondream_image_alt_f32(moondream_image_alt_f32 & image) {
    if (image.data) {
        free(image.data);
        image.data = nullptr;
    }
}

bool normalize_image_u8_to_f32(
    moondream_image_alt_u8 * src, moondream_image_alt_f32 * dst, const float mean[3], const float std[3]
) {
    dst->width = src->width;
    dst->height = src->height;
    dst->n_channels = src->n_channels;
    dst->data = (float *)malloc(sizeof(float) * src->width * src->height * MOONDREAM_N_IMAGE_CHANNELS);
    if (!dst->data) {
        printf("failed to allocate memory for moondream_image_alt_f32 data\n");
        return false;
    }

    for (int i = 0; i < src->width * src->height * MOONDREAM_N_IMAGE_CHANNELS; ++i) {
        int c = i % 3; // rgb
        dst->data[i] = (static_cast<float>(src->data[i]) / 255.0f - mean[c]) / std[c];
    }
    return true;
}

bool image_u8_to_f32(moondream_image_alt_u8 * src, moondream_image_alt_f32 * dst) {
    dst->width = src->width;
    dst->height = src->height;
    dst->n_channels = src->n_channels;
    dst->data = (float *)malloc(sizeof(float) * src->width * src->height * MOONDREAM_N_IMAGE_CHANNELS);
    if (!dst->data) {
        printf("failed to allocate memory for moondream_image_alt_f32 data\n");
        return false;
    }

    for (int i = 0; i < src->width * src->height; ++i) {
        dst->data[i] = static_cast<float>(src->data[i]) / 255.0f;
    }
    return true;
}

bool load_img_to_u8(const char *path, moondream_image_alt_u8 & image) {
    int base_width = -1, base_height = -1, base_channels = -1;
    unsigned char * base_stbi_data = stbi_load(
        path, &base_width, &base_height, &base_channels, MOONDREAM_N_IMAGE_CHANNELS);
    if (!base_stbi_data)
    {
        printf(
            "could not load \"%s\", stbi_failure_reason \"%s\"\n",
            path, stbi_failure_reason());
        return false;
    }

    assert(base_width > 0);
    assert(base_height > 0);
    int n_base_scalars = base_width * base_height * MOONDREAM_N_IMAGE_CHANNELS;
    unsigned char * img_data = (unsigned char *)malloc(sizeof(unsigned char) * n_base_scalars);
    if (!img_data)
    {
        printf("could not allocate memory for image data\n");
        stbi_image_free(base_stbi_data);
        return false;
    }
    for (int i = 0; i < n_base_scalars; ++i)
    {
        img_data[i] = static_cast<unsigned char>(base_stbi_data[i]);
    }
    stbi_image_free(base_stbi_data);

    init_moondream_image_alt_u8(image, base_width, base_height, img_data);
    return true;
}
