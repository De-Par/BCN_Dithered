#pragma once
#include <stdlib.h>
#include <math.h>
#include "dxt_enc.h"


// Bayer matrix for 8x8 dithering
static const float bayer_matrix[8][8] = {
    { 0.0/64, 32.0/64, 8.0/64, 40.0/64, 2.0/64, 34.0/64, 10.0/64, 42.0/64 },
    { 48.0/64, 16.0/64, 56.0/64, 24.0/64, 50.0/64, 18.0/64, 58.0/64, 26.0/64 },
    { 12.0/64, 44.0/64, 4.0/64, 36.0/64, 14.0/64, 46.0/64, 6.0/64, 38.0/64 },
    { 60.0/64, 28.0/64, 52.0/64, 20.0/64, 62.0/64, 30.0/64, 54.0/64, 22.0/64 },
    { 3.0/64, 35.0/64, 11.0/64, 43.0/64, 1.0/64, 33.0/64, 9.0/64, 41.0/64 },
    { 51.0/64, 19.0/64, 59.0/64, 27.0/64, 49.0/64, 17.0/64, 57.0/64, 25.0/64 },
    { 15.0/64, 47.0/64, 7.0/64, 39.0/64, 13.0/64, 45.0/64, 5.0/64, 37.0/64 },
    { 63.0/64, 31.0/64, 55.0/64, 23.0/64, 61.0/64, 29.0/64, 53.0/64, 21.0/64 }
};

// Precomputed 8x8 blue noise threshold matrix
static const float blue_noise_matrix[8][8] = {
    { 0.0/255, 128.0/255, 32.0/255, 160.0/255, 8.0/255, 136.0/255, 40.0/255, 168.0/255 },
    { 192.0/255, 64.0/255, 224.0/255, 96.0/255, 200.0/255, 72.0/255, 232.0/255, 104.0/255 },
    { 48.0/255, 176.0/255, 16.0/255, 144.0/255, 56.0/255, 184.0/255, 24.0/255, 152.0/255 },
    { 240.0/255, 112.0/255, 208.0/255, 80.0/255, 248.0/255, 120.0/255, 216.0/255, 88.0/255 },
    { 12.0/255, 140.0/255, 44.0/255, 172.0/255, 4.0/255, 132.0/255, 36.0/255, 164.0/255 },
    { 204.0/255, 76.0/255, 236.0/255, 108.0/255, 196.0/255, 68.0/255, 228.0/255, 100.0/255 },
    { 60.0/255, 188.0/255, 28.0/255, 156.0/255, 52.0/255, 180.0/255, 20.0/255, 148.0/255 },
    { 252.0/255, 124.0/255, 220.0/255, 92.0/255, 244.0/255, 116.0/255, 212.0/255, 84.0/255 }
};

void bayer_dithering(const rgba_t* ref, rgba_t* dist, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            rgba_t pixel = ref[y * width + x];
            float threshold = bayer_matrix[y % 8][x % 8];
            
            dist[y * width + x].r = (pixel.r / 255.0f > threshold) ? 255 : 0;
            dist[y * width + x].g = (pixel.g / 255.0f > threshold) ? 255 : 0;
            dist[y * width + x].b = (pixel.b / 255.0f > threshold) ? 255 : 0;
            dist[y * width + x].a = pixel.a;
        }
    }
}

void blue_noise_dithering(const rgba_t* ref, rgba_t* dist, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            rgba_t pixel = ref[y * width + x];
            float threshold = blue_noise_matrix[y % 8][x % 8];
            
            dist[y * width + x].r = (pixel.r / 255.0f > threshold) ? 255 : 0;
            dist[y * width + x].g = (pixel.g / 255.0f > threshold) ? 255 : 0;
            dist[y * width + x].b = (pixel.b / 255.0f > threshold) ? 255 : 0;
            dist[y * width + x].a = pixel.a;
        }
    }
}

// Hilbert curve utility functions
static void rot(int n, int *x, int *y, int rx, int ry) {
    if (ry == 0) {
        if (rx == 1) {
            *x = n - 1 - *x;
            *y = n - 1 - *y;
        }
        int t = *x;
        *x = *y;
        *y = t;
    }
}

static void d2xy(int n, int d, int *x, int *y) {
    int rx, ry, s, t = d;
    *x = 0;
    *y = 0;
    for (s = 1; s < n; s *= 2) {
        rx = 1 & (t / 2);
        ry = 1 & (t ^ rx);
        rot(s, x, y, rx, ry);
        *x += s * rx;
        *y += s * ry;
        t /= 4;
    }
}

void riemersma_dithering(const rgba_t* ref, rgba_t* dist, int width, int height) {
    int order = 1;
    int max_dim = width > height ? width : height;
    while (1 << order < max_dim) order++;
    int n = 1 << order;
    int total = n * n;
    
    float *seq = (float*)malloc(16 * sizeof(float));
    for (int i = 0; i < 16; i++) {
        seq[i] = 1.0f - powf(2.0f, -i - 1);
    }
    
    float err_r = 0, err_g = 0, err_b = 0;
    
    for (int idx = 0; idx < total; idx++) {
        int x, y;
        d2xy(n, idx, &x, &y);
        if (x < width && y < height) {
            int pos = y * width + x;
            rgba_t pixel = ref[pos];
            
            float r_val = pixel.r + err_r;
            float g_val = pixel.g + err_g;
            float b_val = pixel.b + err_b;
            
            float threshold = seq[idx % 16];
            
            dist[pos].r = (r_val > threshold * 255) ? 255 : 0;
            dist[pos].g = (g_val > threshold * 255) ? 255 : 0;
            dist[pos].b = (b_val > threshold * 255) ? 255 : 0;
            dist[pos].a = pixel.a;
            
            err_r = r_val - dist[pos].r;
            err_g = g_val - dist[pos].g;
            err_b = b_val - dist[pos].b;
        }
    }
    free(seq);
}

void floyd_steinberg_dithering(const rgba_t* ref, rgba_t* dist, int width, int height) {
    // Create temporary buffers for error diffusion
    float* errors_r = (float*)calloc(width * height, sizeof(float));
    float* errors_g = (float*)calloc(width * height, sizeof(float));
    float* errors_b = (float*)calloc(width * height, sizeof(float));
    
    // Copy reference values to temporary buffers
    for (int i = 0; i < width * height; i++) {
        errors_r[i] = ref[i].r;
        errors_g[i] = ref[i].g;
        errors_b[i] = ref[i].b;
    }
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            
            // Get current pixel values with accumulated error
            float r_val = errors_r[idx];
            float g_val = errors_g[idx];
            float b_val = errors_b[idx];
            
            // Clamp values to [0, 255]
            r_val = (r_val < 0) ? 0 : (r_val > 255) ? 255 : r_val;
            g_val = (g_val < 0) ? 0 : (g_val > 255) ? 255 : g_val;
            b_val = (b_val < 0) ? 0 : (b_val > 255) ? 255 : b_val;
            
            // Quantize to 0 or 255
            unsigned char r_new = (r_val < 128) ? 0 : 255;
            unsigned char g_new = (g_val < 128) ? 0 : 255;
            unsigned char b_new = (b_val < 128) ? 0 : 255;
            
            // Set output pixel
            dist[idx].r = r_new;
            dist[idx].g = g_new;
            dist[idx].b = b_new;
            dist[idx].a = ref[idx].a;
            
            // Calculate quantization errors
            float err_r = r_val - r_new;
            float err_g = g_val - g_new;
            float err_b = b_val - b_new;
            
            // Distribute errors to neighboring pixels using Floyd-Steinberg weights
            if (x + 1 < width) {
                errors_r[idx + 1] += err_r * 7.0f / 16.0f;
                errors_g[idx + 1] += err_g * 7.0f / 16.0f;
                errors_b[idx + 1] += err_b * 7.0f / 16.0f;
            }
            
            if (y + 1 < height) {
                if (x > 0) {
                    errors_r[idx + width - 1] += err_r * 3.0f / 16.0f;
                    errors_g[idx + width - 1] += err_g * 3.0f / 16.0f;
                    errors_b[idx + width - 1] += err_b * 3.0f / 16.0f;
                }
                
                errors_r[idx + width] += err_r * 5.0f / 16.0f;
                errors_g[idx + width] += err_g * 5.0f / 16.0f;
                errors_b[idx + width] += err_b * 5.0f / 16.0f;
                
                if (x + 1 < width) {
                    errors_r[idx + width + 1] += err_r * 1.0f / 16.0f;
                    errors_g[idx + width + 1] += err_g * 1.0f / 16.0f;
                    errors_b[idx + width + 1] += err_b * 1.0f / 16.0f;
                }
            }
        }
    }
    
    free(errors_r);
    free(errors_g);
    free(errors_b);
}