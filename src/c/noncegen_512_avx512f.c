#include "noncegen_512_avx512f.h"
#include <immintrin.h>
#include <string.h>
#include "common.h"
#include "mshabal_512_avx512f.h"

mshabal512_context global_512;
mshabal512_context_fast global_512_fast;

void init_shabal_avx512f() {
    mshabal_init_avx512f(&global_512, 256);
    global_512_fast.out_size = global_512.out_size;
    for (int i = 0; i < 704; i++) global_512_fast.state[i] = global_512.state[i];
    global_512_fast.Whigh = global_512.Whigh;
    global_512_fast.Wlow = global_512.Wlow;
}

// cache:		    cache to save to
// cache_size:      size of cache in nonces
// cache_offset:	cache offset in nonces
// numeric_id:		numeric account id
// loc_startnonce:	nonce to start generation at
// local_nonces: 	number of nonces to generate
void noncegen_avx512f(char *cache,
                   const uint64_t numeric_id, const uint64_t local_startnonce,
                   const uint64_t local_nonces) {

    mshabal512_context_fast local_512_fast;
    uint64_t nonce1, nonce2, nonce3, nonce4, nonce5, nonce6, nonce7, nonce8, nonce9, nonce10, nonce11, nonce12, nonce13, nonce14, nonce15, nonce16;

    char seed[32];  // 64bit numeric account ID, 64bit nonce (blank), 1bit termination, 127 bits zero
    char term[32];  // 1bit 1, 255bit of zeros
    char zero[32];  // 256bit of zeros

    write_seed(seed, numeric_id);
    write_term(term);
    memset(&zero[0], 0, 32);

    //vars shared
    uint8_t* final = (uint8_t*)malloc(sizeof(uint8_t) * MSHABAL512_VECTOR_SIZE * HASH_SIZE);
    
    // prepare smart SIMD aligned termination strings
    // creation could further be optimized, but not much in it as it only runs once per work package
    // creation could also be moved to plotter start
    union {
        mshabal_u32 words[16 * MSHABAL512_VECTOR_SIZE];
        __m512i data[16];
    } t1, t2, t3;

    for (int j = 0; j < 16 * MSHABAL512_VECTOR_SIZE / 2; j += MSHABAL512_VECTOR_SIZE) {
        size_t o = j / 4;
        // t1
        t1.words[j + 0] = *(mshabal_u32 *)(seed + o);
        t1.words[j + 1] = *(mshabal_u32 *)(seed + o);
        t1.words[j + 2] = *(mshabal_u32 *)(seed + o);
        t1.words[j + 3] = *(mshabal_u32 *)(seed + o);
        t1.words[j + 4] = *(mshabal_u32 *)(seed + o);
        t1.words[j + 5] = *(mshabal_u32 *)(seed + o);
        t1.words[j + 6] = *(mshabal_u32 *)(seed + o);
        t1.words[j + 7] = *(mshabal_u32 *)(seed + o);
        t1.words[j + 8] = *(mshabal_u32 *)(seed + o);
        t1.words[j + 9] = *(mshabal_u32 *)(seed + o);
        t1.words[j + 10] = *(mshabal_u32 *)(seed + o);
        t1.words[j + 11] = *(mshabal_u32 *)(seed + o);
        t1.words[j + 12] = *(mshabal_u32 *)(seed + o);
        t1.words[j + 13] = *(mshabal_u32 *)(seed + o);
        t1.words[j + 14] = *(mshabal_u32 *)(seed + o);
        t1.words[j + 15] = *(mshabal_u32 *)(seed + o);        
        t1.words[j + 0 + 128] = *(mshabal_u32 *)(zero + o);
        t1.words[j + 1 + 128] = *(mshabal_u32 *)(zero + o);
        t1.words[j + 2 + 128] = *(mshabal_u32 *)(zero + o);
        t1.words[j + 3 + 128] = *(mshabal_u32 *)(zero + o);
        t1.words[j + 4 + 128] = *(mshabal_u32 *)(zero + o);
        t1.words[j + 5 + 128] = *(mshabal_u32 *)(zero + o);
        t1.words[j + 6 + 128] = *(mshabal_u32 *)(zero + o);
        t1.words[j + 7 + 128] = *(mshabal_u32 *)(zero + o);
        t1.words[j + 8 + 128] = *(mshabal_u32 *)(zero + o);
        t1.words[j + 9 + 128] = *(mshabal_u32 *)(zero + o);
        t1.words[j + 10 + 128] = *(mshabal_u32 *)(zero + o);
        t1.words[j + 11 + 128] = *(mshabal_u32 *)(zero + o);
        t1.words[j + 12 + 128] = *(mshabal_u32 *)(zero + o);
        t1.words[j + 13 + 128] = *(mshabal_u32 *)(zero + o);
        t1.words[j + 14 + 128] = *(mshabal_u32 *)(zero + o);
        t1.words[j + 15 + 128] = *(mshabal_u32 *)(zero + o);
        // t2
        // (first 256bit skipped, will later be filled with data)
        t2.words[j + 0 + 128] = *(mshabal_u32 *)(seed + o);
        t2.words[j + 1 + 128] = *(mshabal_u32 *)(seed + o);
        t2.words[j + 2 + 128] = *(mshabal_u32 *)(seed + o);
        t2.words[j + 3 + 128] = *(mshabal_u32 *)(seed + o);
        t2.words[j + 4 + 128] = *(mshabal_u32 *)(seed + o);
        t2.words[j + 5 + 128] = *(mshabal_u32 *)(seed + o);
        t2.words[j + 6 + 128] = *(mshabal_u32 *)(seed + o);
        t2.words[j + 7 + 128] = *(mshabal_u32 *)(seed + o);
        t2.words[j + 8 + 128] = *(mshabal_u32 *)(seed + o);
        t2.words[j + 9 + 128] = *(mshabal_u32 *)(seed + o);
        t2.words[j + 10 + 128] = *(mshabal_u32 *)(seed + o);
        t2.words[j + 11 + 128] = *(mshabal_u32 *)(seed + o);
        t2.words[j + 12 + 128] = *(mshabal_u32 *)(seed + o);
        t2.words[j + 13 + 128] = *(mshabal_u32 *)(seed + o);
        t2.words[j + 14 + 128] = *(mshabal_u32 *)(seed + o);
        t2.words[j + 15 + 128] = *(mshabal_u32 *)(seed + o);
        // t3
        t3.words[j + 0] = *(mshabal_u32 *)(term + o);
        t3.words[j + 1] = *(mshabal_u32 *)(term + o);
        t3.words[j + 2] = *(mshabal_u32 *)(term + o);
        t3.words[j + 3] = *(mshabal_u32 *)(term + o);
        t3.words[j + 4] = *(mshabal_u32 *)(term + o);
        t3.words[j + 5] = *(mshabal_u32 *)(term + o);
        t3.words[j + 6] = *(mshabal_u32 *)(term + o);
        t3.words[j + 7] = *(mshabal_u32 *)(term + o);        
        t3.words[j + 8] = *(mshabal_u32 *)(term + o);
        t3.words[j + 9] = *(mshabal_u32 *)(term + o);
        t3.words[j + 10] = *(mshabal_u32 *)(term + o);
        t3.words[j + 11] = *(mshabal_u32 *)(term + o);
        t3.words[j + 12] = *(mshabal_u32 *)(term + o);
        t3.words[j + 13] = *(mshabal_u32 *)(term + o);
        t3.words[j + 14] = *(mshabal_u32 *)(term + o);
        t3.words[j + 15] = *(mshabal_u32 *)(term + o);
        
        t3.words[j + 0 + 128] = *(mshabal_u32 *)(zero + o);
        t3.words[j + 1 + 128] = *(mshabal_u32 *)(zero + o);
        t3.words[j + 2 + 128] = *(mshabal_u32 *)(zero + o);
        t3.words[j + 3 + 128] = *(mshabal_u32 *)(zero + o);
        t3.words[j + 4 + 128] = *(mshabal_u32 *)(zero + o);
        t3.words[j + 5 + 128] = *(mshabal_u32 *)(zero + o);
        t3.words[j + 6 + 128] = *(mshabal_u32 *)(zero + o);
        t3.words[j + 7 + 128] = *(mshabal_u32 *)(zero + o);
        t3.words[j + 8 + 128] = *(mshabal_u32 *)(zero + o);
        t3.words[j + 9 + 128] = *(mshabal_u32 *)(zero + o);
        t3.words[j + 10 + 128] = *(mshabal_u32 *)(zero + o);
        t3.words[j + 11 + 128] = *(mshabal_u32 *)(zero + o);
        t3.words[j + 12 + 128] = *(mshabal_u32 *)(zero + o);
        t3.words[j + 13 + 128] = *(mshabal_u32 *)(zero + o);
        t3.words[j + 14 + 128] = *(mshabal_u32 *)(zero + o);
        t3.words[j + 15 + 128] = *(mshabal_u32 *)(zero + o);
    }

    for (uint64_t n = 0; n < local_nonces; n += 16) {
        // iterate nonces (16 per cycle - avx512)
        // min 16 nonces left for avx512 processing, otherwise SISD

        // generate nonce numbers & change endianness
        nonce1 = bswap_64((uint64_t)(local_startnonce + n + 0));
        nonce2 = bswap_64((uint64_t)(local_startnonce + n + 1));
        nonce3 = bswap_64((uint64_t)(local_startnonce + n + 2));
        nonce4 = bswap_64((uint64_t)(local_startnonce + n + 3));
        nonce5 = bswap_64((uint64_t)(local_startnonce + n + 4));
        nonce6 = bswap_64((uint64_t)(local_startnonce + n + 5));
        nonce7 = bswap_64((uint64_t)(local_startnonce + n + 6));
        nonce8 = bswap_64((uint64_t)(local_startnonce + n + 7));
        nonce9 = bswap_64((uint64_t)(local_startnonce + n + 8));
        nonce10 = bswap_64((uint64_t)(local_startnonce + n + 9));
        nonce11 = bswap_64((uint64_t)(local_startnonce + n + 10));
        nonce12 = bswap_64((uint64_t)(local_startnonce + n + 11));
        nonce13 = bswap_64((uint64_t)(local_startnonce + n + 12));
        nonce14 = bswap_64((uint64_t)(local_startnonce + n + 13));
        nonce15 = bswap_64((uint64_t)(local_startnonce + n + 14));
        nonce16 = bswap_64((uint64_t)(local_startnonce + n + 15));

        // store nonce numbers in relevant termination strings
        for (int j = 32; j < 16 * MSHABAL512_VECTOR_SIZE / 4; j += MSHABAL512_VECTOR_SIZE) {
            size_t o = j / 4 - 8;
            // t1
            t1.words[j + 0] = *(mshabal_u32 *)((char *)&nonce1 + o);
            t1.words[j + 1] = *(mshabal_u32 *)((char *)&nonce2 + o);
            t1.words[j + 2] = *(mshabal_u32 *)((char *)&nonce3 + o);
            t1.words[j + 3] = *(mshabal_u32 *)((char *)&nonce4 + o);
            t1.words[j + 4] = *(mshabal_u32 *)((char *)&nonce5 + o);
            t1.words[j + 5] = *(mshabal_u32 *)((char *)&nonce6 + o);
            t1.words[j + 6] = *(mshabal_u32 *)((char *)&nonce7 + o);
            t1.words[j + 7] = *(mshabal_u32 *)((char *)&nonce8 + o);
            t1.words[j + 8] = *(mshabal_u32 *)((char *)&nonce9 + o);
            t1.words[j + 9] = *(mshabal_u32 *)((char *)&nonce10 + o);
            t1.words[j + 10] = *(mshabal_u32 *)((char *)&nonce11 + o);
            t1.words[j + 11] = *(mshabal_u32 *)((char *)&nonce12 + o);
            t1.words[j + 12] = *(mshabal_u32 *)((char *)&nonce13 + o);
            t1.words[j + 13] = *(mshabal_u32 *)((char *)&nonce14 + o);
            t1.words[j + 14] = *(mshabal_u32 *)((char *)&nonce15 + o);
            t1.words[j + 15] = *(mshabal_u32 *)((char *)&nonce16 + o);

            t2.words[j + 0 + 128] = *(mshabal_u32 *)((char *)&nonce1 + o);
            t2.words[j + 1 + 128] = *(mshabal_u32 *)((char *)&nonce2 + o);
            t2.words[j + 2 + 128] = *(mshabal_u32 *)((char *)&nonce3 + o);
            t2.words[j + 3 + 128] = *(mshabal_u32 *)((char *)&nonce4 + o);
            t2.words[j + 4 + 128] = *(mshabal_u32 *)((char *)&nonce5 + o);
            t2.words[j + 5 + 128] = *(mshabal_u32 *)((char *)&nonce6 + o);
            t2.words[j + 6 + 128] = *(mshabal_u32 *)((char *)&nonce7 + o);
            t2.words[j + 7 + 128] = *(mshabal_u32 *)((char *)&nonce8 + o); 
            t2.words[j + 8 + 128] = *(mshabal_u32 *)((char *)&nonce9 + o);
            t2.words[j + 9 + 128] = *(mshabal_u32 *)((char *)&nonce10 + o);
            t2.words[j + 10 + 128] = *(mshabal_u32 *)((char *)&nonce11 + o);
            t2.words[j + 11 + 128] = *(mshabal_u32 *)((char *)&nonce12 + o);
            t2.words[j + 12 + 128] = *(mshabal_u32 *)((char *)&nonce13 + o);
            t2.words[j + 13 + 128] = *(mshabal_u32 *)((char *)&nonce14 + o);
            t2.words[j + 14 + 128] = *(mshabal_u32 *)((char *)&nonce15 + o);
            t2.words[j + 15 + 128] = *(mshabal_u32 *)((char *)&nonce16 + o);
        }

        // start shabal round    

        // 3 cases: first 128 rounds uses case 1 or 2, after that case 3
        // case 1: first 128 rounds, hashes are even: use termination string 1
        // case 2: first 128 rounds, hashes are odd: use termination string 2
        // case 3: round > 128: use termination string 3
        
        // round 1
        memcpy(&local_512_fast, &global_512_fast,
                sizeof(global_512_fast));  // fast initialize shabal                 
        
            mshabal_hash_fast_avx512f(
            &local_512_fast, NULL, &t1,
            &cache[MSHABAL512_VECTOR_SIZE * (NONCE_SIZE - HASH_SIZE)], 16 >> 6);

        // store first hash into smart termination string 2 (data is vectored and SIMD aligned)
        memcpy(&t2, &cache[MSHABAL512_VECTOR_SIZE * (NONCE_SIZE - HASH_SIZE)],
                MSHABAL512_VECTOR_SIZE * (HASH_SIZE));

        // round 2 - 128
        for (size_t i = NONCE_SIZE - HASH_SIZE; i > (NONCE_SIZE - HASH_CAP); i -= HASH_SIZE) {
            // check if msg can be divided into 512bit packages without a
            // remainder
            if (i % 64 == 0) {
                // last msg = seed + termination
                    mshabal_hash_fast_avx512f(&local_512_fast, &cache[i * MSHABAL512_VECTOR_SIZE],
                                            &t1,
                                            &cache[(i - HASH_SIZE) * MSHABAL512_VECTOR_SIZE],
                                            (NONCE_SIZE + 16 - i) >> 6);
            } else {
                // last msg = 256 bit data + seed + termination
                    mshabal_hash_fast_avx512f(&local_512_fast, &cache[i * MSHABAL512_VECTOR_SIZE],
                                            &t2,
                                            &cache[(i - HASH_SIZE) * MSHABAL512_VECTOR_SIZE],
                                            (NONCE_SIZE + 16 - i) >> 6);
            }
        }

        // round 128-8192
        for (size_t i = NONCE_SIZE - HASH_CAP; i > 0; i -= HASH_SIZE) {
                mshabal_hash_fast_avx512f(&local_512_fast, &cache[i * MSHABAL512_VECTOR_SIZE], &t3,
                                        &cache[(i - HASH_SIZE) * MSHABAL512_VECTOR_SIZE],
                                        (HASH_CAP) >> 6);
        }
        
        // generate final hash
            mshabal_hash_fast_avx512f(&local_512_fast, &cache[0], &t1, &final[0],
                                    (NONCE_SIZE + 16) >> 6);
        
        // XOR using SIMD
        // load final hash
        __m512i F[8];
        for (int j = 0; j < 8; j++) F[j] = _mm512_loadu_si512((__m512i *)final + j);
        // xor all hashes with final hash
        for (int j = 0; j < 8 * 2 * HASH_CAP; j++)
            _mm512_storeu_si512(
                (__m512i *)cache + j,
                _mm512_xor_si512(_mm512_loadu_si512((__m512i *)cache + j), F[j % 8]));
        cache += MSHABAL512_VECTOR_SIZE * NONCE_SIZE;

    }
    free(final);
}

void find_best_deadline_avx512f(char *data, uint64_t scoop, uint64_t nonce_count, char *gensig,
                                uint64_t *best_deadline, uint64_t *best_offset) {
    uint64_t d0 = 0, d1 = 0, d2 = 0, d3 = 0, d4 = 0, d5 = 0, d6 = 0, d7 = 0, d8 = 0, d9 = 0,
             d10 = 0, d11 = 0, d12 = 0, d13 = 0, d14 = 0, d15 = 0;
    char term[32];
    write_term(term);

    // local copy of global fast context
    mshabal512_context_fast x;
    memcpy(&x, &global_512_fast, sizeof(global_512_fast));

    // prepare shabal inputs
    union {
        mshabal_u32 words[8 * MSHABAL512_VECTOR_SIZE];
        __m512i data[8];
    } gensig_simd, term_simd;

    for (uint64_t i = 0; i < 16 * MSHABAL512_VECTOR_SIZE / 2; i += MSHABAL512_VECTOR_SIZE) {
        size_t o = i / 4;
        gensig_simd.words[i + 0] = *(mshabal_u32 *)(gensig + o);
        gensig_simd.words[i + 1] = *(mshabal_u32 *)(gensig + o);
        gensig_simd.words[i + 2] = *(mshabal_u32 *)(gensig + o);
        gensig_simd.words[i + 3] = *(mshabal_u32 *)(gensig + o);
        gensig_simd.words[i + 4] = *(mshabal_u32 *)(gensig + o);
        gensig_simd.words[i + 5] = *(mshabal_u32 *)(gensig + o);
        gensig_simd.words[i + 6] = *(mshabal_u32 *)(gensig + o);
        gensig_simd.words[i + 7] = *(mshabal_u32 *)(gensig + o);
        gensig_simd.words[i + 8] = *(mshabal_u32 *)(gensig + o);
        gensig_simd.words[i + 9] = *(mshabal_u32 *)(gensig + o);
        gensig_simd.words[i + 10] = *(mshabal_u32 *)(gensig + o);
        gensig_simd.words[i + 11] = *(mshabal_u32 *)(gensig + o);
        gensig_simd.words[i + 12] = *(mshabal_u32 *)(gensig + o);
        gensig_simd.words[i + 13] = *(mshabal_u32 *)(gensig + o);
        gensig_simd.words[i + 14] = *(mshabal_u32 *)(gensig + o);
        gensig_simd.words[i + 15] = *(mshabal_u32 *)(gensig + o);
        term_simd.words[i + 0] = *(mshabal_u32 *)(term + o);
        term_simd.words[i + 1] = *(mshabal_u32 *)(term + o);
        term_simd.words[i + 2] = *(mshabal_u32 *)(term + o);
        term_simd.words[i + 3] = *(mshabal_u32 *)(term + o);
        term_simd.words[i + 4] = *(mshabal_u32 *)(term + o);
        term_simd.words[i + 5] = *(mshabal_u32 *)(term + o);
        term_simd.words[i + 6] = *(mshabal_u32 *)(term + o);
        term_simd.words[i + 7] = *(mshabal_u32 *)(term + o);
        term_simd.words[i + 8] = *(mshabal_u32 *)(term + o);
        term_simd.words[i + 9] = *(mshabal_u32 *)(term + o);
        term_simd.words[i + 10] = *(mshabal_u32 *)(term + o);
        term_simd.words[i + 11] = *(mshabal_u32 *)(term + o);
        term_simd.words[i + 12] = *(mshabal_u32 *)(term + o);
        term_simd.words[i + 13] = *(mshabal_u32 *)(term + o);
        term_simd.words[i + 14] = *(mshabal_u32 *)(term + o);
        term_simd.words[i + 15] = *(mshabal_u32 *)(term + o);
    }

    uint64_t mirrorscoop = 4095-scoop;

    for (uint64_t i = 0; i < nonce_count; i+=16) {
            // poc2: u1 first hash, u2 second hash = mirror hash
            char *u1 = data + i * NONCE_SIZE + scoop * SCOOP_SIZE * MSHABAL512_VECTOR_SIZE;
            char *u2 = data + i * NONCE_SIZE + mirrorscoop * SCOOP_SIZE * MSHABAL512_VECTOR_SIZE + HASH_SIZE * MSHABAL512_VECTOR_SIZE; 

            mshabal_deadline_fast_avx512f(&x, &gensig_simd, u1, u2, &term_simd, &d0, &d1, &d2, &d3, &d4, &d5, &d6, &d7,
                                           &d8, &d9, &d10, &d11, &d12, &d13, &d14, &d15);

            SET_BEST_DEADLINE(d0, i + 0);
            SET_BEST_DEADLINE(d1, i + 1);
            SET_BEST_DEADLINE(d2, i + 2);
            SET_BEST_DEADLINE(d3, i + 3);
            SET_BEST_DEADLINE(d4, i + 4);
            SET_BEST_DEADLINE(d5, i + 5);
            SET_BEST_DEADLINE(d6, i + 6);
            SET_BEST_DEADLINE(d7, i + 7);
            SET_BEST_DEADLINE(d8, i + 8);
            SET_BEST_DEADLINE(d9, i + 9);
            SET_BEST_DEADLINE(d10, i + 10);
            SET_BEST_DEADLINE(d11, i + 11);
            SET_BEST_DEADLINE(d12, i + 12);
            SET_BEST_DEADLINE(d13, i + 13);
            SET_BEST_DEADLINE(d14, i + 14);
            SET_BEST_DEADLINE(d15, i + 15);        
    }
}
