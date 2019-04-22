#pragma once

#include <stdint.h>
#include <stdlib.h>

void init_shabal_avx2();

void noncegen_avx2(char *cache,
                   const uint64_t numeric_id, const uint64_t local_startnonce,
                   const uint64_t local_nonces);             

void find_best_deadline_avx2(char *data, uint64_t scoop, uint64_t nonce_count, char *gensig, 
                             uint64_t *best_deadline, uint64_t *best_offset);
