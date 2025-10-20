// randmat.c (C99)
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

/* -------- tiny RNG (xorshift32) -------- */
static uint32_t xorshift32(uint32_t *s) {
    uint32_t x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x ? x : 0x9E3779B9u;   // avoid zero state
    return *s;
}
static double urand01(uint32_t *s) {              // in [0,1)
    return (xorshift32(s)) / 4294967296.0;        // 2^32
}
static double urand_double(double lo, double hi, uint32_t *s) {
    return lo + (hi - lo) * urand01(s);
}
static int urand_int(int lo, int hi, uint32_t *s) {  // unbiased in [lo,hi]
    if (lo > hi) { int t = lo; lo = hi; hi = t; }
    uint32_t range = (uint32_t)((uint64_t)hi - (uint64_t)lo + 1u);
    uint32_t limit = (UINT32_MAX / range) * range;   // rejection to avoid bias
    uint32_t r;
    do { r = xorshift32(s); } while (r >= limit);
    return lo + (int)(r % range);
}

/* -------- CLI & printing -------- */
static void usage(const char *prog) {
    fprintf(stderr,
        "Usage:\n"
        "  %s N MIN MAX [--int] [--seed SEED]\n\n"
        "Examples:\n"
        "  %s 5 -1.0 1.0           # 5x5 doubles in [-1,1]\n"
        "  %s 4 0 100 --int        # 4x4 ints in [0,100]\n"
        "  %s 3 -5 5 --seed 42     # deterministic doubles\n",
        prog, prog, prog, prog);
}

int main(int argc, char **argv) {
    if (argc < 4) { usage(argv[0]); return 1; }

    char *end = NULL;
    long N = strtol(argv[1], &end, 10);
    if (*end || N <= 0) { fprintf(stderr, "N must be a positive integer\n"); return 1; }

    double dmin = strtod(argv[2], &end);
    if (*end) { fprintf(stderr, "MIN must be a number\n"); return 1; }
    double dmax = strtod(argv[3], &end);
    if (*end) { fprintf(stderr, "MAX must be a number\n"); return 1; }
    if (dmin > dmax) { double tmp = dmin; dmin = dmax; dmax = tmp; }

    int want_int = 0;
    uint32_t seed = (uint32_t)time(NULL) ^ (uint32_t)(uintptr_t)&seed;

    for (int i = 4; i < argc; ++i) {
        if (strcmp(argv[i], "--int") == 0) {
            want_int = 1;
        } else if (strcmp(argv[i], "--seed") == 0) {
            if (i + 1 >= argc) { fprintf(stderr, "Missing value after --seed\n"); return 1; }
            seed = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else {
            fprintf(stderr, "Warning: ignoring unknown option '%s'\n", argv[i]);
        }
    }

    if (want_int) {
        int imin = (int)dmin, imax = (int)dmax;
        if (imin > imax) { int t = imin; imin = imax; imax = t; }
        for (long r = 0; r < N; ++r) {
            for (long c = 0; c < N; ++c) {
                int v = urand_int(imin, imax, &seed);
                printf("%d%s", v, (c + 1 == N) ? "" : " ");
            }
            printf("\n");
        }
    } else {
        for (long r = 0; r < N; ++r) {
            for (long c = 0; c < N; ++c) {
                double v = urand_double(dmin, dmax, &seed);
                printf("%.6f%s", v, (c + 1 == N) ? "" : ", ");
            }
            printf(",\n");
        }
    }
    return 0;
}
