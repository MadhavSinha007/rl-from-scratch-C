#include "prng.h"

// Default initial state for global PCG generator
static prng_state s_prng_state = { 
    0x853c49e6748fea9bULL, 0xda3e39cb94b95bdbULL,
};

// Seed a specific PRNG instance
void prng_seed_r(prng_state* rng, u64 initstate, u64 initseq) {
    rng->state = 0U;
    rng->inc = (initseq << 1u) | 1u;
    prng_rand_r(rng);
    rng->state += initstate;
    prng_rand_r(rng);
}

// Seed the global PRNG instance
void prng_seed(u64 initstate, u64 initseq) {
    prng_seed_r(&s_prng_state, initstate, initseq);
}

// Generate a 32-bit unsigned pseudo-random integer using PCG-XSH-RR
u32 prng_rand_r(prng_state* rng) {
    u64 oldstate = rng->state;
    rng->state = oldstate * 6364136223846793005ULL + rng->inc;
    u32 xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    u32 rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

// Global 32-bit random integer generator
u32 prng_rand(void) {
    return prng_rand_r(&s_prng_state);
}

// Uniform random integer in range [0, range - 1] (cross-platform)
u32 randn(u32 range) {
    if (range == 0) return 0;
    return prng_rand() % range;
}

// Generate a random float in [0.0, 1.0)
f32 prng_randf_r(prng_state* rng) {
    return (f32)prng_rand_r(rng) / (f32)UINT32_MAX;
}

// Global random float in [0.0, 1.0)
f32 prng_randf(void) {
    return prng_randf_r(&s_prng_state);
}