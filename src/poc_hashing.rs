use crate::shabal256::{shabal256_deadline_fast, shabal256_hash_fast};
use hex;
use std::mem::transmute;
use std::u64;

const HASH_SIZE: usize = 32;
const HASH_CAP: usize = 4096;
const NUM_SCOOPS: usize = 4096;
const SCOOP_SIZE: usize = 64;
pub const NONCE_SIZE: usize = (NUM_SCOOPS * SCOOP_SIZE);
const MESSAGE_SIZE: usize = 16;

pub fn decode_gensig(gensig: &str) -> [u8; 32] {
    let mut gensig_bytes = [0; 32];
    gensig_bytes[..].clone_from_slice(&hex::decode(gensig).unwrap());
    gensig_bytes
}

pub fn calculate_scoop(height: u64, gensig: &[u8; 32]) -> u32 {
    let mut data: [u8; 64] = [0; 64];
    let height_bytes: [u8; 8] = unsafe { transmute(height.to_be()) };

    data[..32].clone_from_slice(gensig);
    data[32..40].clone_from_slice(&height_bytes);
    data[40] = 0x80;
    let data = unsafe { std::mem::transmute::<&[u8; 64], &[u32; 16]>(&data) };

    let new_gensig = &shabal256_hash_fast(&[], &data);
    (u32::from(new_gensig[30] & 0x0F) << 8) | u32::from(new_gensig[31])
}

pub fn find_best_deadline_rust(
    data: &[u8],
    scoop: u64,
    number_of_nonces: u64,
    gensig: &[u8; 32],
) -> (u64, u64) {
    let mut best_deadline = u64::MAX;
    let mut best_offset = 0;
    let mirror_scoop = 4095 - scoop;
    for i in 0..number_of_nonces as usize {
        let result =
            //shabal256_deadline_fast(&data[i * SCOOP_SIZE..i * SCOOP_SIZE + SCOOP_SIZE], &gensig);
            shabal256_deadline_fast(
                &data[i * NONCE_SIZE + scoop as usize * SCOOP_SIZE..i*NONCE_SIZE + scoop as usize * SCOOP_SIZE + HASH_SIZE],
                &data[i * NONCE_SIZE + mirror_scoop as usize * SCOOP_SIZE + HASH_SIZE..i * NONCE_SIZE + mirror_scoop as usize * SCOOP_SIZE + SCOOP_SIZE],
                &gensig
            );
        if result < best_deadline {
            best_deadline = result;
            best_offset = i;
        }
    }
    (best_deadline, best_offset as u64)
}

// cache:		    cache to save to
// local_num:		thread number
// numeric_id:		numeric account id
// loc_startnonce	nonce to start generation at
// local_nonces: 	number of nonces to generate
pub fn noncegen_rust(cache: &mut [u8], numeric_id: u64, local_startnonce: u64, local_nonces: u64) {
    let numeric_id: [u32; 2] = unsafe { std::mem::transmute(numeric_id.to_be()) };

    //let mut buffer = [0u8; NONCE_SIZE];
    let mut final_buffer = [0u8; HASH_SIZE];

    // prepare termination strings
    let mut t1 = [0u32; MESSAGE_SIZE];
    t1[0..2].clone_from_slice(&numeric_id);
    t1[4] = 0x80;

    let mut t2 = [0u32; MESSAGE_SIZE];
    t2[8..10].clone_from_slice(&numeric_id);
    t2[12] = 0x80;

    let mut t3 = [0u32; MESSAGE_SIZE];
    t3[0] = 0x80;

    for n in 0..local_nonces {
        // generate nonce numbers & change endianness
        let nonce: [u32; 2] = unsafe { std::mem::transmute((local_startnonce + n).to_be()) };
        // store nonce numbers in relevant termination strings
        t1[2..4].clone_from_slice(&nonce);
        t2[10..12].clone_from_slice(&nonce);

        // start shabal rounds

        // 3 cases: first 128 rounds uses case 1 or 2, after that case 3
        // case 1: first 128 rounds, hashes are even: use termination string 1
        // case 2: first 128 rounds, hashes are odd: use termination string 2
        // case 3: round > 128: use termination string 3
        // round 1
        let hash = shabal256_hash_fast(&[], &t1);

        cache[n as usize * NONCE_SIZE + NONCE_SIZE - HASH_SIZE
            ..n as usize * NONCE_SIZE + NONCE_SIZE]
            .clone_from_slice(&hash);
        let hash = unsafe { std::mem::transmute::<[u8; 32], [u32; 8]>(hash) };

        // store first hash into smart termination string 2
        t2[0..8].clone_from_slice(&hash);
        // round 2 - 128
        for i in (NONCE_SIZE - HASH_CAP + HASH_SIZE..=NONCE_SIZE - HASH_SIZE)
            .rev()
            .step_by(HASH_SIZE)
        {
            // check if msg can be divided into 512bit packages without a
            // remainder
            if i % 64 == 0 {
                // last msg = seed + termination
                let hash = &shabal256_hash_fast(
                    &cache[n as usize * NONCE_SIZE + i..n as usize * NONCE_SIZE + NONCE_SIZE],
                    &t1,
                );
                cache[n as usize * NONCE_SIZE + i - HASH_SIZE..n as usize * NONCE_SIZE + i]
                    .clone_from_slice(hash);
            } else {
                // last msg = 256 bit data + seed + termination
                let hash = &shabal256_hash_fast(
                    &cache[n as usize * NONCE_SIZE + i..n as usize * NONCE_SIZE + NONCE_SIZE],
                    &t2,
                );
                cache[n as usize * NONCE_SIZE + i - HASH_SIZE..n as usize * NONCE_SIZE + i]
                    .clone_from_slice(hash);
            }
        }

        // round 128-8192
        for i in (HASH_SIZE..=NONCE_SIZE - HASH_CAP).rev().step_by(HASH_SIZE) {
            let hash = &shabal256_hash_fast(
                &cache[n as usize * NONCE_SIZE + i..n as usize * NONCE_SIZE + i + HASH_CAP],
                &t3,
            );
            cache[n as usize * NONCE_SIZE + i - HASH_SIZE..n as usize * NONCE_SIZE + i]
                .clone_from_slice(hash);
        }

        // generate final hash
        final_buffer.clone_from_slice(&shabal256_hash_fast(
            &cache[n as usize * NONCE_SIZE + 0..n as usize * NONCE_SIZE + NONCE_SIZE],
            &t1,
        ));

        // XOR with final
        for i in 0..NONCE_SIZE {
            cache[n as usize * NONCE_SIZE + i] ^= final_buffer[i % HASH_SIZE];
        }
    }
}
