#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <getopt.h>

#define STB_IMAGE_IMPLEMENTATION
#include "include/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "include/stb_image_write.h"

// Compression mode enumeration
typedef enum {
    BC1_MODE,
    BC3_MODE
} bc_mode;

// Dithering mode enumeration
typedef enum {
    DITHER_NONE,
    DITHER_BAYER,
    DITHER_FLOYD_STEINBERG
} dither_mode;

// Configuration structure
typedef struct {
    bc_mode mode;
    dither_mode dither;
    const char* input_file;
    const char* output_file;
    int measure_performance;
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

// Bayer matrix for ordered dithering
static const uint8_t bayer_matrix[4][4] = {
    { 0,  8,  2, 10 },
    { 12, 4, 14, 6 },
    { 3,  11, 1, 9 },
    { 15, 7, 13, 5 }
};

static uint16_t rgb888_to_rgb565(uint8_t r, uint8_t g, uint8_t b) {
   return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3); 
}

static int clamp(int value, int min, int max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

// Apply Bayer ordered dithering to an image
void apply_bayer_dithering(rgba_t* image, int width, int height) {
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            rgba_t* pixel = &image[y * width + x];
            int threshold = bayer_matrix[y % 4][x % 4];
            
            // Add dithering to each channel
            pixel->r = clamp(pixel->r + ((threshold - 8) * 2), 0, 255);
            pixel->g = clamp(pixel->g + ((threshold - 8) * 2), 0, 255);
            pixel->b = clamp(pixel->b + ((threshold - 8) * 2), 0, 255);
        }
    }
}

// Apply Floyd-Steinberg dithering to an image
void apply_floyd_steinberg_dithering(rgba_t* image, int width, int height) {
    // Create a temporary copy of the image for error diffusion
    rgba_t* temp_image = malloc(width * height * sizeof(rgba_t));
    memcpy(temp_image, image, width * height * sizeof(rgba_t));
    
    // Floyd-Steinberg dithering (sequential as it's a diffusion algorithm)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            rgba_t* old_pixel = &temp_image[y * width + x];
            rgba_t* new_pixel = &image[y * width + x];
            
            // Quantize the pixel (this is a simplified version)
            new_pixel->r = (old_pixel->r >> 3) << 3;
            new_pixel->g = (old_pixel->g >> 2) << 2;
            new_pixel->b = (old_pixel->b >> 3) << 3;
            
            // Calculate quantization error
            int err_r = old_pixel->r - new_pixel->r;
            int err_g = old_pixel->g - new_pixel->g;
            int err_b = old_pixel->b - new_pixel->b;
            
            // Diffuse error to neighboring pixels
            if (x + 1 < width) {
                rgba_t* right = &temp_image[y * width + x + 1];
                right->r = clamp(right->r + err_r * 7 / 16, 0, 255);
                right->g = clamp(right->g + err_g * 7 / 16, 0, 255);
                right->b = clamp(right->b + err_b * 7 / 16, 0, 255);
            }
            
            if (y + 1 < height) {
                if (x > 0) {
                    rgba_t* down_left = &temp_image[(y + 1) * width + x - 1];
                    down_left->r = clamp(down_left->r + err_r * 3 / 16, 0, 255);
                    down_left->g = clamp(down_left->g + err_g * 3 / 16, 0, 255);
                    down_left->b = clamp(down_left->b + err_b * 3 / 16, 0, 255);
                }
                
                rgba_t* down = &temp_image[(y + 1) * width + x];
                down->r = clamp(down->r + err_r * 5 / 16, 0, 255);
                down->g = clamp(down->g + err_g * 5 / 16, 0, 255);
                down->b = clamp(down->b + err_b * 5 / 16, 0, 255);
                
                if (x + 1 < width) {
                    rgba_t* down_right = &temp_image[(y + 1) * width + x + 1];
                    down_right->r = clamp(down_right->r + err_r * 1 / 16, 0, 255);
                    down_right->g = clamp(down_right->g + err_g * 1 / 16, 0, 255);
                    down_right->b = clamp(down_right->b + err_b * 1 / 16, 0, 255);
                }
            }
        }
    }
    free(temp_image);
}

// BC1 Decoder - Optimized with SIMD-friendly code
void decode_bc1(const bc1_block *block, rgba_t *pixels, int block_x, int block_y, int width, int height) {
    rgba_t colors[4];
    
    // Decode endpoints
    colors[0].r = (block->c0 >> 11) & 0x1F; 
    colors[0].g = (block->c0 >> 5) & 0x3F; 
    colors[0].b = block->c0 & 0x1F;
    
    colors[1].r = (block->c1 >> 11) & 0x1F; 
    colors[1].g = (block->c1 >> 5) & 0x3F; 
    colors[1].b = block->c1 & 0x1F;
    
    // Convert to 8-bit (optimized with bitwise operations)
    colors[0].r = (colors[0].r << 3) | (colors[0].r >> 2);
    colors[0].g = (colors[0].g << 2) | (colors[0].g >> 4);
    colors[0].b = (colors[0].b << 3) | (colors[0].b >> 2);
    
    colors[1].r = (colors[1].r << 3) | (colors[1].r >> 2);
    colors[1].g = (colors[1].g << 2) | (colors[1].g >> 4);
    colors[1].b = (colors[1].b << 3) | (colors[1].b >> 2);
    
    // Determine color palette
    if (block->c0 > block->c1) {
        colors[2].r = (2 * colors[0].r + colors[1].r) / 3;
        colors[2].g = (2 * colors[0].g + colors[1].g) / 3;
        colors[2].b = (2 * colors[0].b + colors[1].b) / 3;
        
        colors[3].r = (colors[0].r + 2 * colors[1].r) / 3;
        colors[3].g = (colors[0].g + 2 * colors[1].g) / 3;
        colors[3].b = (colors[0].b + 2 * colors[1].b) / 3;
        
    } else {
        colors[2].r = (colors[0].r + colors[1].r) / 2;
        colors[2].g = (colors[0].g + colors[1].g) / 2;
        colors[2].b = (colors[0].b + colors[1].b) / 2;
        colors[3].r = 0; colors[3].g = 0; colors[3].b = 0; colors[3].a = 0;
    }
    
    colors[0].a = colors[1].a = colors[2].a = 255;
    colors[3].a = (block->c0 > block->c1) ? 255 : 0;

    // Decode indices and place pixels in correct positions
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
            int px = block_x * 4 + x;
            int py = block_y * 4 + y;
            
            // Only decode if within image bounds
            if (px < width && py < height) {
                uint8_t idx = (block->indices >> (2 * (y * 4 + x))) & 3;
                pixels[py * width + px] = colors[idx];
            }
        }
    }
}

// BC1 Encoder with OpenMP and SIMD optimizations
void encode_bc1(const rgba_t *image, int width, int height, bc1_block *blocks, dither_mode dither) {
    int block_count_x = (width + 3) / 4;
    int block_count_y = (height + 3) / 4;

    // Apply dithering if requested
    rgba_t* dithered_image = NULL;
    const rgba_t* source_image = image;
    
    if (dither != DITHER_NONE) {
        dithered_image = malloc(width * height * sizeof(rgba_t));
        memcpy(dithered_image, image, width * height * sizeof(rgba_t));
        
        if (dither == DITHER_BAYER) {
            apply_bayer_dithering(dithered_image, width, height);
        } else if (dither == DITHER_FLOYD_STEINBERG) {
            apply_floyd_steinberg_dithering(dithered_image, width, height);
        }
        source_image = dithered_image;
    }
    
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int by = 0; by < block_count_y; by++) {
        for (int bx = 0; bx < block_count_x; bx++) {
            rgba_t block_pixels[16];
            
            // Extract block pixels
            #pragma omp simd
            for (int i = 0; i < 16; i++) {
                int x = i % 4;
                int y = i / 4;
                int px = bx * 4 + x;
                int py = by * 4 + y;
                
                if (px < width && py < height) {
                    block_pixels[i] = source_image[py * width + px];
                } else {
                    block_pixels[i] = (rgba_t){0, 0, 0, 0};
                }
            }
            
            // Find min and max colors using reduction
            uint8_t min_r = 255, min_g = 255, min_b = 255;
            uint8_t max_r = 0, max_g = 0, max_b = 0;
            
            #pragma omp simd reduction(min:min_r,min_g,min_b) reduction(max:max_r,max_g,max_b)
            for (int i = 0; i < 16; i++) {
                min_r = block_pixels[i].r < min_r ? block_pixels[i].r : min_r;
                min_g = block_pixels[i].g < min_g ? block_pixels[i].g : min_g;
                min_b = block_pixels[i].b < min_b ? block_pixels[i].b : min_b;
                max_r = block_pixels[i].r > max_r ? block_pixels[i].r : max_r;
                max_g = block_pixels[i].g > max_g ? block_pixels[i].g : max_g;
                max_b = block_pixels[i].b > max_b ? block_pixels[i].b : max_b;
            }
            
            // Convert to 5-6-5 format using precomputed table
            uint16_t c0 = rgb888_to_rgb565(max_r, max_g, max_b);
            uint16_t c1 = rgb888_to_rgb565(min_r, min_g, min_b);
            
            // Create indices using SIMD
            uint32_t indices = 0;
            
            #pragma omp simd
            for (int i = 0; i < 16; i++) {
                rgba_t pixel = block_pixels[i];
                uint16_t color_val = rgb888_to_rgb565(pixel.r, pixel.g, pixel.b);
                
                // Simple distance calculation
                int dist0 = abs((int)color_val - (int)c0);
                int dist1 = abs((int)color_val - (int)c1);
                
                uint8_t idx = (dist0 < dist1) ? 0 : 1;
                indices |= (idx << (2 * i));
            }
            
            bc1_block encoded_block;
            encoded_block.c0 = c0;
            encoded_block.c1 = c1;
            encoded_block.indices = indices;
            
            blocks[by * block_count_x + bx] = encoded_block;
        }
    }

    if (dithered_image) {
        free(dithered_image);
    }
}

// BC3 Decoder - Optimized with SIMD
void decode_bc3(const bc3_block *block, rgba_t *pixels, int block_x, int block_y, int width, int height) {
    // Decode alpha
    uint8_t alpha[8];
    alpha[0] = block->a0;
    alpha[1] = block->a1;
    
    if (alpha[0] > alpha[1]) {
        #pragma omp simd
        for (int i = 0; i < 6; i++) {
            alpha[2 + i] = ((6 - i) * alpha[0] + (1 + i) * alpha[1] + 3) / 7;
        }
    } else {
        #pragma omp simd
        for (int i = 0; i < 4; i++) {
            alpha[2 + i] = ((4 - i) * alpha[0] + (1 + i) * alpha[1] + 2) / 5;
        }
        alpha[6] = 0;
        alpha[7] = 255;
    }
    
    // Decode alpha indices
    uint64_t alpha_indices = 0;
    memcpy(&alpha_indices, block->alpha_indices, 6);
    
    // Decode color first
    rgba_t color_pixels[16];
    decode_bc1(&block->color_block, color_pixels, 0, 0, 4, 4);
    
    // Apply alpha and place pixels in correct positions
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
            int px = block_x * 4 + x;
            int py = block_y * 4 + y;
            
            // Only decode if within image bounds
            if (px < width && py < height) {
                uint8_t idx = (alpha_indices >> (3 * (y * 4 + x))) & 7;
                rgba_t pixel = color_pixels[y * 4 + x];
                pixel.a = alpha[idx];
                pixels[py * width + px] = pixel;
            }
        }
    }
}

// BC3 Encoder with OpenMP and SIMD optimizations
void encode_bc3(const rgba_t *image, int width, int height, bc3_block *blocks, dither_mode dither) {
    int block_count_x = (width + 3) / 4;
    int block_count_y = (height + 3) / 4;

    // Apply dithering if requested
    rgba_t* dithered_image = NULL;
    const rgba_t* source_image = image;
    
    if (dither != DITHER_NONE) {
        dithered_image = malloc(width * height * sizeof(rgba_t));
        memcpy(dithered_image, image, width * height * sizeof(rgba_t));
        
        if (dither == DITHER_BAYER) {
            apply_bayer_dithering(dithered_image, width, height);
        } else if (dither == DITHER_FLOYD_STEINBERG) {
            apply_floyd_steinberg_dithering(dithered_image, width, height);
        }
        source_image = dithered_image;
    }
    
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int by = 0; by < block_count_y; by++) {
        for (int bx = 0; bx < block_count_x; bx++) {
            rgba_t block_pixels[16];
            
            // Extract block pixels
            #pragma omp simd
            for (int i = 0; i < 16; i++) {
                int x = i % 4;
                int y = i / 4;
                int px = bx * 4 + x;
                int py = by * 4 + y;
                
                if (px < width && py < height) {
                    block_pixels[i] = source_image[py * width + px];
                } else {
                    block_pixels[i] = (rgba_t){0, 0, 0, 0};
                }
            }
            
            // Encode alpha
            uint8_t min_alpha = 255, max_alpha = 0;
            
            #pragma omp simd reduction(min:min_alpha) reduction(max:max_alpha)
            for (int i = 0; i < 16; i++) {
                min_alpha = block_pixels[i].a < min_alpha ? block_pixels[i].a : min_alpha;
                max_alpha = block_pixels[i].a > max_alpha ? block_pixels[i].a : max_alpha;
            }
            
            bc3_block encoded_block;
            encoded_block.a0 = max_alpha;
            encoded_block.a1 = min_alpha;
            
            // Create alpha indices using SIMD
            uint64_t alpha_indices = 0;
            
            #pragma omp simd
            for (int i = 0; i < 16; i++) {
                uint8_t alpha_val = block_pixels[i].a;
                uint8_t idx;
                
                if (max_alpha > min_alpha) {
                    // 8-alpha mode
                    if (alpha_val == max_alpha) idx = 0;
                    else if (alpha_val == min_alpha) idx = 1;
                    else if (alpha_val > (6*max_alpha + 1*min_alpha)/7) idx = 2;
                    else if (alpha_val > (5*max_alpha + 2*min_alpha)/7) idx = 3;
                    else if (alpha_val > (4*max_alpha + 3*min_alpha)/7) idx = 4;
                    else if (alpha_val > (3*max_alpha + 4*min_alpha)/7) idx = 5;
                    else if (alpha_val > (2*max_alpha + 5*min_alpha)/7) idx = 6;
                    else idx = 7;
                } else {
                    // 6-alpha mode
                    if (alpha_val == max_alpha) idx = 0;
                    else if (alpha_val == min_alpha) idx = 1;
                    else if (alpha_val > (4*max_alpha + 1*min_alpha)/5) idx = 2;
                    else if (alpha_val > (3*max_alpha + 2*min_alpha)/5) idx = 3;
                    else if (alpha_val > (2*max_alpha + 3*min_alpha)/5) idx = 4;
                    else if (alpha_val > (1*max_alpha + 4*min_alpha)/5) idx = 5;
                    else idx = 6; // Reserved
                }
                
                alpha_indices |= ((uint64_t)idx << (3 * i));
            }
            
            memcpy(encoded_block.alpha_indices, &alpha_indices, 6);
            
            // Encode color
            encode_bc1(block_pixels, 4, 4, &encoded_block.color_block, DITHER_NONE);
            
            blocks[by * block_count_x + bx] = encoded_block;
        }
    }

    if (dithered_image) {
        free(dithered_image);
    }
}

// Load image using stb_image with OpenMP optimization
rgba_t* load_image(const char* filename, int* width, int* height) {
    int channels;
    unsigned char* data = stbi_load(filename, width, height, &channels, 4);
    if (!data) {
        fprintf(stderr, "Error loading image: %s\n", filename);
        return NULL;
    }
    
    rgba_t* image = malloc(*width * *height * sizeof(rgba_t));
    
    #pragma omp parallel for
    for (int i = 0; i < *width * *height; i++) {
        image[i].r = data[i * 4];
        image[i].g = data[i * 4 + 1];
        image[i].b = data[i * 4 + 2];
        image[i].a = data[i * 4 + 3];
    }
    
    stbi_image_free(data);
    return image;
}

// Save image using stb_image_write with OpenMP optimization
int save_image(const char* filename, const rgba_t* image, int width, int height) {
    unsigned char* data = malloc(width * height * 4);
    
    #pragma omp parallel for
    for (int i = 0; i < width * height; i++) {
        data[i * 4] = image[i].r;
        data[i * 4 + 1] = image[i].g;
        data[i * 4 + 2] = image[i].b;
        data[i * 4 + 3] = image[i].a;
    }
    
    int result = stbi_write_png(filename, width, height, 4, data, width * 4);
    free(data);
    return result;
}

void print_usage(const char* program_name) {
    printf("Usage: %s [OPTIONS] <mode> <ref> <dist>\n", program_name);
    printf("Process images with specified compression mode\n\n");
    printf("Arguments:\n");
    printf("  mode            Processing mode: bc1 or bc3\n");
    printf("  ref             Path to reference image\n");
    printf("  dist            Path to distorted image\n\n");
    printf("Options:\n");
    printf("  -d, --dither TYPE   Dithering type: none, bayer, or floyd (default: none)\n");
    printf("  -p, --perf          Measure performance\n");
    printf("  -h, --help          Display this help message\n");
}

// Parse command line arguments
int parse_arguments(int argc, char* argv[], config_t* config) {
    // Define long options
    static struct option long_options[] = {
        {"dither", required_argument, 0, 'd'},
        {"perf", no_argument, 0, 'p'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };
    
    // Set default values
    config->dither = DITHER_NONE;
    config->measure_performance = 0;
    
    int opt;
    int option_index = 0;
    
    // Parse options
    while ((opt = getopt_long(argc, argv, "d:ph", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'd':
                if (strcmp(optarg, "bayer") == 0) {
                    config->dither = DITHER_BAYER;
                } else if (strcmp(optarg, "floyd") == 0) {
                    config->dither = DITHER_FLOYD_STEINBERG;
                } else if (strcmp(optarg, "none") == 0) {
                    config->dither = DITHER_NONE;
                } else {
                    fprintf(stderr, "Error: Invalid dither type '%s'. Use 'none', 'bayer', or 'floyd'.\n", optarg);
                    return 0;
                }
                break;
                
            case 'p':
                config->measure_performance = 1;
                break;
                
            case 'h':
                print_usage(argv[0]);
                return 0;
                
            case '?':
                // getopt_long already printed an error message
                return 0;
                
            default:
                fprintf(stderr, "Error: Unknown option\n");
                return 0;
        }
    }
    
    // Check for required positional arguments
    if (optind + 3 > argc) {
        fprintf(stderr, "Error: Missing required arguments\n");
        print_usage(argv[0]);
        return 0;
    }
    
    // Parse mode
    if (strcmp(argv[optind], "bc1") == 0) {
        config->mode = BC1_MODE;
    } else if (strcmp(argv[optind], "bc3") == 0) {
        config->mode = BC3_MODE;
    } else {
        fprintf(stderr, "Error: Invalid mode '%s'. Use 'bc1' or 'bc3'.\n", argv[optind]);
        return 0;
    }
    
    // Parse file names
    config->input_file = argv[optind + 1];
    config->output_file = argv[optind + 2];
    
    return 1;
}

// Process image based on configuration
int process_image(const config_t* config) {
    // Load image
    int width, height;
    rgba_t* image = load_image(config->input_file, &width, &height);
    if (!image) {
        return 0;
    }
    
    int block_count_x = (width + 3) / 4;
    int block_count_y = (height + 3) / 4;
    int block_count = block_count_x * block_count_y;
    
    // Process based on mode
    if (config->mode == BC1_MODE) {
        bc1_block* bc1_blocks = malloc(block_count * sizeof(bc1_block));
        
        // Encode with BC1
        double start = omp_get_wtime();
        encode_bc1(image, width, height, bc1_blocks, config->dither);
        double encode_time = omp_get_wtime() - start;
        
        // Decode BC1
        rgba_t* decoded_image = malloc(width * height * sizeof(rgba_t));
        start = omp_get_wtime();
        
        // Initialize decoded image to black
        memset(decoded_image, 0, width * height * sizeof(rgba_t));
        
        // Decode each block to the correct position
        #pragma omp parallel for collapse(2)
        for (int by = 0; by < block_count_y; by++) {
            for (int bx = 0; bx < block_count_x; bx++) {
                int block_idx = by * block_count_x + bx;
                decode_bc1(&bc1_blocks[block_idx], decoded_image, bx, by, width, height);
            }
        }
        
        double decode_time = omp_get_wtime() - start;
        
        // Save result
        if (!save_image(config->output_file, decoded_image, width, height)) {
            fprintf(stderr, "Error saving image: %s\n", config->output_file);
            free(bc1_blocks);
            free(decoded_image);
            free(image);
            return 0;
        }
        
        if (config->measure_performance) {
            printf("Encode Time: %.6f s\n", encode_time);
            printf("Decode Time: %.6f s\n", decode_time);
            printf("Total Time: %.6f s\n", encode_time + decode_time);
        }
        
        free(bc1_blocks);
        free(decoded_image);
        
    } else { // BC3_MODE
        bc3_block* bc3_blocks = malloc(block_count * sizeof(bc3_block));
        
        // Encode with BC3
        double start = omp_get_wtime();
        encode_bc3(image, width, height, bc3_blocks, config->dither);
        double encode_time = omp_get_wtime() - start;
        
        // Decode BC3
        rgba_t* decoded_image = malloc(width * height * sizeof(rgba_t));
        start = omp_get_wtime();
        
        // Initialize decoded image to black
        memset(decoded_image, 0, width * height * sizeof(rgba_t));
        
        // Decode each block to the correct position
        #pragma omp parallel for collapse(2)
        for (int by = 0; by < block_count_y; by++) {
            for (int bx = 0; bx < block_count_x; bx++) {
                int block_idx = by * block_count_x + bx;
                decode_bc3(&bc3_blocks[block_idx], decoded_image, bx, by, width, height);
            }
        }
        
        double decode_time = omp_get_wtime() - start;
        
        // Save result
        if (!save_image(config->output_file, decoded_image, width, height)) {
            fprintf(stderr, "Error saving image: %s\n", config->output_file);
            free(bc3_blocks);
            free(decoded_image);
            free(image);
            return 0;
        }
        
        if (config->measure_performance) {
            printf("Encode Time: %.6f s\n", encode_time);
            printf("Decode Time: %.6f s\n", decode_time);
            printf("Total Time: %.6f s\n", encode_time + decode_time);
        }
        
        free(bc3_blocks);
        free(decoded_image);
    }
    
    free(image);
    return 1;
}

int main(int argc, char* argv[]) {
    config_t config;
    
    if (!parse_arguments(argc, argv, &config)) {
        return 1;
    }

    printf("Mode: %s\n", config.mode == BC1_MODE ? "BC1" : "BC3");
    printf("Reference: %s\n", config.input_file);
    printf("Distorted: %s\n", config.output_file);
    printf("Dither: ");
    switch (config.dither) {
        case DITHER_NONE: printf("none\n"); break;
        case DITHER_BAYER: printf("bayer\n"); break;
        case DITHER_FLOYD_STEINBERG: printf("floyd\n"); break;
    }
    printf("Performance measurement: %s\n", config.measure_performance ? "enabled" : "disabled");
    
    return process_image(&config) ? 0 : 1;
}