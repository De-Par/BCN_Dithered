#pragma once
#include <stdint.h>
#include <stdbool.h>


typedef enum {
    BC1_MODE,
    BC3_MODE
} bc_mode;

typedef enum {
    DITHER_NONE,
    DITHER_BAYER,
    DITHER_FLOYD_STEINBERG,
    DITHER_BLUE_NOISE,
    DITHER_RIEMERSMA
} dither_mode;

typedef struct {
    bc_mode mode;
    dither_mode dither;
    const char* input_file;
    const char* output_file;
    bool verbose;
} config_t;

typedef struct {
    uint8_t r, g, b, a;
} rgba_t;

typedef struct {
    uint16_t c0, c1;
    uint32_t indices;
} bc1_block;

typedef struct {
    uint8_t a0, a1;
    uint8_t alpha_indices[6];
    bc1_block color_block;
} bc3_block;

static inline uint16_t rgb888_to_rgb565(uint8_t r, uint8_t g, uint8_t b);

rgba_t* load_image(const char* filename, int* width, int* height);
int save_image(const char* filename, const rgba_t* image, int width, int height);
int process_image(const config_t* config);

void encode_bc1(const rgba_t *image, int width, int height, bc1_block *blocks);
void decode_bc1(const bc1_block *block, rgba_t *pixels, int block_x, int block_y, int width, int height);

void encode_bc3(const rgba_t *image, int width, int height, bc3_block *blocks);
void decode_bc3(const bc3_block *block, rgba_t *pixels, int block_x, int block_y, int width, int height);