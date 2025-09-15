#!/bin/bash

export CC=/usr/local/opt/llvm/bin/clang
clang -O3 -march=native -ffast-math -fopenmp main.c dxt_enc.c -o app  