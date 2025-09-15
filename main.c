#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <string.h>

#include "dxt_enc.h"


void print_usage(const char* program_name) {
    printf("Usage: %s [OPTIONS] <mode> <ref> <dist>\n", program_name);
    printf("Process images with specified compression mode\n\n");
    printf("Arguments:\n");
    printf("  mode            Processing mode: bc1 or bc3\n");
    printf("  ref             Path to reference image\n");
    printf("  dist            Path to distorted image\n\n");
    printf("Options:\n");
    printf("  -d, --dither TYPE   Dithering type: none, bayer, blue, riemersma or floyd (default: none)\n");
    printf("  -v, --verbose       Print output information\n");
    printf("  -h, --help          Display this help message\n");
}

int parse_arguments(int argc, char* argv[], config_t* config) {
    // Define long options
    static struct option long_options[] = {
        {"dither", required_argument, 0, 'd'},
        {"verbose", no_argument, 0, 'v'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };
    
    // Set default values
    config->dither = DITHER_NONE;
    config->verbose = false;
    
    int opt;
    int option_index = 0;
    
    // Parse options
    while ((opt = getopt_long(argc, argv, "d:vh", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'd':
                if (strcmp(optarg, "bayer") == 0) {
                    config->dither = DITHER_BAYER;
                } else if (strcmp(optarg, "blue") == 0) {
                    config->dither = DITHER_BLUE_NOISE;
                } else if (strcmp(optarg, "floyd") == 0) {
                    config->dither = DITHER_FLOYD_STEINBERG;
                } else if (strcmp(optarg, "riemersma") == 0) {
                    config->dither = DITHER_RIEMERSMA;
                } else if (strcmp(optarg, "none") == 0) {
                    config->dither = DITHER_NONE;
                } else {
                    fprintf(stderr, "Error: Invalid dither type '%s'. Use 'none', 'bayer', 'blue' or 'floyd'.\n", optarg);
                    return 0;
                }
                break;
                
            case 'v':
                config->verbose = true;
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

int main(int argc, char* argv[]) {
    config_t config;
    
    if (!parse_arguments(argc, argv, &config)) {
        return EXIT_FAILURE;
    }

    if (config.verbose) {
        printf("Mode: %s\n", config.mode == BC1_MODE ? "BC1" : "BC3");
        printf("Reference: %s\n", config.input_file);
        printf("Distorted: %s\n", config.output_file);
        printf("Dither: ");

        switch (config.dither) {
            case DITHER_NONE: printf("none\n"); break;
            case DITHER_BAYER: printf("bayer\n"); break;
            case DITHER_BLUE_NOISE: printf("blue\n"); break;
            case DITHER_RIEMERSMA: printf("riemersma\n"); break;
            case DITHER_FLOYD_STEINBERG: printf("floyd\n"); break;
        }
        printf("Verbose: %s\n", config.verbose ? "true" : "false");
    }

    if (!process_image(&config)) {
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}