use crate::buffer::PageAlignedByteBuffer;
use crate::miner::NonceData;
use crate::poc_hashing::find_best_deadline_rust;
use crate::poc_hashing::noncegen_rust;
use crate::poc_hashing::NONCE_SIZE;
use crate::scheduler::{HasherMessage, RoundInfo};
use crossbeam_channel::Sender;
use futures::sync::mpsc;
use libc::{c_void, uint64_t};
use std::u64;

#[derive(Debug, Clone)]
pub enum SimdExtension {
    AVX512f,
    AVX2,
    AVX,
    SSE2,
    None,
}

extern "C" {
    pub fn init_shabal_sse2();
    pub fn init_shabal_avx();
    pub fn init_shabal_avx2();
    pub fn init_shabal_avx512f();
    pub fn noncegen_sse2(
        cache: *mut c_void,
        numeric_ID: uint64_t,
        local_startnonce: uint64_t,
        local_nonces: uint64_t,
    );
    pub fn noncegen_avx(
        cache: *mut c_void,
        numeric_ID: uint64_t,
        local_startnonce: uint64_t,
        local_nonces: uint64_t,
    );
    pub fn noncegen_avx2(
        cache: *mut c_void,
        numeric_ID: uint64_t,
        local_startnonce: uint64_t,
        local_nonces: uint64_t,
    );
    pub fn noncegen_avx512f(
        cache: *mut c_void,
        numeric_ID: uint64_t,
        local_startnonce: uint64_t,
        local_nonces: uint64_t,
    );
    pub fn find_best_deadline_avx512f(
        data: *const c_void,
        scoop: uint64_t,
        nonce_count: uint64_t,
        gensig: *const c_void,
        best_deadline: *mut uint64_t,
        best_offset: *mut uint64_t,
    ) -> ();
    pub fn find_best_deadline_avx2(
        data: *const c_void,
        scoop: uint64_t,
        nonce_count: uint64_t,
        gensig: *const c_void,
        best_deadline: *mut uint64_t,
        best_offset: *mut uint64_t,
    ) -> ();
    pub fn find_best_deadline_avx(
        data: *const c_void,
        scoop: uint64_t,
        nonce_count: uint64_t,
        gensig: *const c_void,
        best_deadline: *mut uint64_t,
        best_offset: *mut uint64_t,
    ) -> ();
    pub fn find_best_deadline_sse2(
        data: *const c_void,
        scoop: uint64_t,
        nonce_count: uint64_t,
        gensig: *const c_void,
        best_deadline: *mut uint64_t,
        best_offset: *mut uint64_t,
    ) -> ();
}

pub struct CpuTask {
    pub numeric_id: u64,
    pub local_startnonce: u64,
    pub local_nonces: u64,
    pub round: RoundInfo,
}

#[derive(Clone)]
pub struct DeadlineHashingTask {
    pub number_of_nonces: uint64_t,
    pub gensig: [u8; 32],
    pub numeric_id: u64,
    pub height: u64,
    pub base_target: u64,
    pub tx_hash_reply: mpsc::Sender<NonceData>,
    pub start_warp: u64,
    pub number_of_warps: u64,
}

pub fn init_cpu_extensions() -> SimdExtension {
    if is_x86_feature_detected!("avx512f") {
        unsafe {
            init_shabal_avx512f();
        }
        SimdExtension::AVX512f
    } else if is_x86_feature_detected!("avx2") {
        unsafe {
            init_shabal_avx2();
        }
        SimdExtension::AVX2
    } else if is_x86_feature_detected!("avx") {
        unsafe {
            init_shabal_avx();
        }
        SimdExtension::AVX
    } else if is_x86_feature_detected!("sse2") {
        unsafe {
            init_shabal_sse2();
        }
        SimdExtension::SSE2
    } else {
        SimdExtension::None
    }
}

pub fn hash_cpu(
    tx: Sender<HasherMessage>,
    hasher_task: CpuTask,
    simd_ext: SimdExtension,
) -> impl FnOnce() {
    move || {
        // alloc
        let buffer = PageAlignedByteBuffer::new(hasher_task.local_nonces as usize * NONCE_SIZE);
        let bs = buffer.get_buffer();
        let mut bs = bs.lock().unwrap();
        unsafe {
            match simd_ext {
                SimdExtension::AVX512f => noncegen_avx512f(
                    bs.as_mut_ptr() as *mut c_void,
                    hasher_task.numeric_id,
                    hasher_task.local_startnonce,
                    hasher_task.local_nonces,
                ),
                SimdExtension::AVX2 => noncegen_avx2(
                    bs.as_mut_ptr() as *mut c_void,
                    hasher_task.numeric_id,
                    hasher_task.local_startnonce,
                    hasher_task.local_nonces,
                ),
                SimdExtension::AVX => noncegen_avx(
                    bs.as_mut_ptr() as *mut c_void,
                    hasher_task.numeric_id,
                    hasher_task.local_startnonce,
                    hasher_task.local_nonces,
                ),
                SimdExtension::SSE2 => noncegen_sse2(
                    bs.as_mut_ptr() as *mut c_void,
                    hasher_task.numeric_id,
                    hasher_task.local_startnonce,
                    hasher_task.local_nonces,
                ),
                _ => noncegen_rust(
                    &mut bs[..],
                    hasher_task.numeric_id,
                    hasher_task.local_startnonce,
                    hasher_task.local_nonces,
                ),
            }
        }

        // calc best deadline
        #[allow(unused_assignments)]
        let mut deadline: u64 = u64::MAX;
        #[allow(unused_assignments)]
        let mut offset: u64 = 0;

        unsafe {
            match simd_ext {
                SimdExtension::AVX512f => find_best_deadline_avx512f(
                    bs.as_ptr() as *const c_void,
                    hasher_task.round.scoop,
                    hasher_task.local_nonces,
                    hasher_task.round.gensig.as_ptr() as *const c_void,
                    &mut deadline,
                    &mut offset,
                ),
                SimdExtension::AVX2 => find_best_deadline_avx2(
                    bs.as_ptr() as *const c_void,
                    hasher_task.round.scoop,
                    hasher_task.local_nonces,
                    hasher_task.round.gensig.as_ptr() as *const c_void,
                    &mut deadline,
                    &mut offset,
                ),
                SimdExtension::AVX => find_best_deadline_avx(
                    bs.as_ptr() as *const c_void,
                    hasher_task.round.scoop,
                    hasher_task.local_nonces,
                    hasher_task.round.gensig.as_ptr() as *const c_void,
                    &mut deadline,
                    &mut offset,
                ),
                SimdExtension::SSE2 => find_best_deadline_sse2(
                    bs.as_ptr() as *const c_void,
                    hasher_task.round.scoop,
                    hasher_task.local_nonces,
                    hasher_task.round.gensig.as_ptr() as *const c_void,
                    &mut deadline,
                    &mut offset,
                ),
                _ => {
                    let result = find_best_deadline_rust(
                        &bs,
                        hasher_task.round.scoop,
                        hasher_task.local_nonces,
                        &hasher_task.round.gensig,
                    );
                    deadline = result.0;
                    offset = result.1;
                }
            }
        }

        // report hashing done
        tx.send(HasherMessage::SubmitDeadline((
            hasher_task.round.height,
            hasher_task.local_startnonce + offset,
            deadline,
        )))
        .expect("GPU task can't communicate with scheduler thread.");

        tx.send(HasherMessage::NoncesProcessed(hasher_task.local_nonces))
            .expect("GPU task can't communicate with scheduler thread.");
        tx.send(HasherMessage::CpuRequestForWork)
            .expect("GPU task can't communicate with scheduler thread.");
    }
}
