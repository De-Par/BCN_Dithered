#include <omp.h>

#define STB_IMAGE_IMPLEMENTATION
#include "include/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "include/stb_image_write.h"

#include "dxt_enc.h"
#include "dither.h"


static inline uint16_t rgb888_to_rgb565(uint8_t r, uint8_t g, uint8_t b) {
   return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3); 
}

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

void encode_bc1(const rgba_t *image, int width, int height, bc1_block *blocks) {
    int block_count_x = (width + 3) / 4;
    int block_count_y = (height + 3) / 4;
    
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
                    block_pixels[i] = image[py * width + px];
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
}

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

void encode_bc3(const rgba_t *image, int width, int height, bc3_block *blocks) {
    int block_count_x = (width + 3) / 4;
    int block_count_y = (height + 3) / 4;
    
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
                    block_pixels[i] = image[py * width + px];
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
            encode_bc1(block_pixels, 4, 4, &encoded_block.color_block);
            
            blocks[by * block_count_x + bx] = encoded_block;
        }
    }
}

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
        encode_bc1(image, width, height, bc1_blocks);
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

        if (config->dither == DITHER_BAYER) {
            bayer_dithering(image, decoded_image, width, height);
        } else if (config->dither == DITHER_BLUE_NOISE) {
            blue_noise_dithering(image, decoded_image, width, height);
        } else if (config->dither == DITHER_FLOYD_STEINBERG) {
            floyd_steinberg_dithering(image, decoded_image, width, height);
        } else if (config->dither == DITHER_RIEMERSMA) {
            riemersma_dithering(image, decoded_image, width, height);
        }
        
        // Save result
        if (!save_image(config->output_file, decoded_image, width, height)) {
            fprintf(stderr, "Error saving image: %s\n", config->output_file);
            free(bc1_blocks);
            free(decoded_image);
            free(image);
            return 0;
        }
        
        if (config->verbose) {
            printf("Encode Time: %.6f s\n", encode_time);
            printf("Decode Time: %.6f s\n", decode_time);
            printf(" Total Time: %.6f s\n", encode_time + decode_time);
        }
        
        free(bc1_blocks);
        free(decoded_image);
        
    } else { // BC3_MODE
        bc3_block* bc3_blocks = malloc(block_count * sizeof(bc3_block));
        
        // Encode with BC3
        double start = omp_get_wtime();
        encode_bc3(image, width, height, bc3_blocks);
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

        if (config->dither == DITHER_BAYER) {
            bayer_dithering(image, decoded_image, width, height);
        } else if (config->dither == DITHER_BLUE_NOISE) {
            blue_noise_dithering(image, decoded_image, width, height);
        } else if (config->dither == DITHER_FLOYD_STEINBERG) {
            floyd_steinberg_dithering(image, decoded_image, width, height);
        } else if (config->dither == DITHER_RIEMERSMA) {
            riemersma_dithering(image, decoded_image, width, height);
        }
        
        // Save result
        if (!save_image(config->output_file, decoded_image, width, height)) {
            fprintf(stderr, "Error saving image: %s\n", config->output_file);
            free(bc3_blocks);
            free(decoded_image);
            free(image);
            return 0;
        }
        
        if (config->verbose) {
            printf("Encode Time: %.6f s\n", encode_time);
            printf("Decode Time: %.6f s\n", decode_time);
            printf(" Total Time: %.6f s\n", encode_time + decode_time);
        }
        
        free(bc3_blocks);
        free(decoded_image);
    }
    
    free(image);
    return 1;
}