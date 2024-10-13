#include <cstdio>
#include <limits>

#include "helpers.hpp"

size_t utf8_len(char src) {
    const size_t lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}

double bytes_to_gib(size_t n_bytes) {
    return static_cast<double>(n_bytes) / (1024.0 * 1024.0 * 1024.0);
}

bool size_to_int32(size_t s, int32_t * i) {
    if (s <= static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        *i = static_cast<int32_t>(s);
        return true;
    }
    return false;
}

void set_tensor_name(ggml_tensor * cur, const char * name, int il) {
    if (il >= 0) {
        ggml_format_name(cur, "%s-%d", name, il);
    } else {
        ggml_set_name(cur, name);
    }
}

void log_tensor(ggml_tensor * dst, const ggml_tensor * src, int ith, int nth, void * userdata) {
    if (ith != 0) {
        // Only log from the first thread.
        return;
    }

    printf("Shape: %lld %lld %lld %lld\n", dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
    printf("Strides: %zu %zu %zu %zu\n",dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
    switch (dst->type) {
        case GGML_TYPE_F16:
            printf("Type: f16\n");
            break;
        case GGML_TYPE_F32:
            printf("Type: f32\n");
            break;
        default:
            printf("Type: unknown/not setup for logging\n");
            break;
    }

    for (int i = 0; i < src->ne[2]; i++) {
        printf("[");
        for (int j = 0; j < 2/*src->ne[1]*/; j++) {
            printf("[");
            /*if (i > 0) {
                printf("\t");
            }*/
            for (int k = 0; k < src->ne[0]; ++k) {
                float f = *(float *)(((char *)src->data) + i*src->nb[2] + j*src->nb[1] + k*src->nb[0]);
                printf("%.7f ", (double)f);
            }
            printf("]\n");
        }
        printf("]\n");
    }
}
