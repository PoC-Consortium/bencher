use crate::cpu_hasher::{hash_cpu, CpuTask, SimdExtension};
#[cfg(feature = "opencl")]
use crate::gpu_hasher::{create_gpu_hasher_thread, GpuTask};
use crate::miner::NonceData;
#[cfg(feature = "opencl")]
use crate::ocl::gpu_init;
use crate::ocl::GpuConfig;
use chrono::Local;
use crossbeam_channel::{unbounded, Receiver};
use futures::sync::mpsc::UnboundedSender;
use std::cmp::min;
#[cfg(feature = "opencl")]
use std::thread;
use std::u64;
use stopwatch::Stopwatch;

#[derive(Clone)]
pub struct RoundInfo {
    pub gensig: [u8; 32],
    pub base_target: u64,
    pub scoop: u64,
    pub height: u64,
}

pub enum HasherMessage {
    CpuRequestForWork,
    GpuRequestForWork(usize),
    NoncesProcessed(u64),
    SubmitDeadline((u64, u64, u64)), //(height, nonce, deadline)
}

pub fn create_scheduler_thread(
    numeric_id: u64,
    start_nonce: u64,
    cpu_threads: u8,
    cpu_task_size: u64,
    simd_ext: SimdExtension,
    gpus: Vec<GpuConfig>,
    blocktime: u64,
    rx_rounds: Receiver<RoundInfo>,
    tx_nonce: UnboundedSender<NonceData>,
) -> impl FnOnce() {
    move || {
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(cpu_threads as usize)
            .build()
            .unwrap();

        let (tx, rx) = unbounded();

        // create gpu threads and channels
        #[cfg(feature = "opencl")]
        let gpu_contexts = if gpus.len() > 0 {
            Some(gpu_init(&gpus))
        } else {
            None
        };

        #[cfg(feature = "opencl")]
        let gpus = match gpu_contexts {
            Some(x) => x,
            None => Vec::new(),
        };
        #[cfg(feature = "opencl")]
        let mut gpu_threads = Vec::new();
        #[cfg(feature = "opencl")]
        let mut gpu_channels = Vec::new();

        #[cfg(feature = "opencl")]
        for (i, gpu) in gpus.iter().enumerate() {
            gpu_channels.push(unbounded());
            gpu_threads.push(thread::spawn({
                create_gpu_hasher_thread(
                    i,
                    gpu.clone(),
                    tx.clone(),
                    gpu_channels.last().unwrap().1.clone(),
                )
            }));
        }

        let mut sw = Stopwatch::start_new();

        for round in &rx_rounds {
            sw.restart();
            let nonces_to_hash = u64::MAX - start_nonce;
            let mut requested = 0u64;
            let mut processed = 0u64;

            // kickoff first gpu and cpu runs
            #[cfg(feature = "opencl")]
            for (i, gpu) in gpus.iter().enumerate() {
                // schedule next gpu task
                let task_size = min(gpu.worksize as u64, nonces_to_hash - requested);
                if task_size > 0 {
                    gpu_channels[i]
                        .0
                        .send(Some(GpuTask {
                            numeric_id,
                            local_startnonce: start_nonce + requested,
                            local_nonces: task_size,
                            round: round.clone(),
                        }))
                        .unwrap();
                }
                requested += task_size;
            }

            // kickoff first cpu runs
            for _ in 0..cpu_threads {
                let task_size = min(cpu_task_size, nonces_to_hash - requested);
                if task_size > 0 {
                    let task = hash_cpu(
                        tx.clone(),
                        CpuTask {
                            numeric_id,
                            local_startnonce: start_nonce + requested,
                            local_nonces: task_size,
                            round: round.clone(),
                        },
                        simd_ext.clone(),
                    );
                    thread_pool.spawn(task);
                }
                requested += task_size;
            }

            // control loop
            let rx = &rx;
            for msg in rx {
                match msg {
                    // schedule next cpu task
                    HasherMessage::CpuRequestForWork => {
                        let task_size = min(cpu_task_size, nonces_to_hash - requested);
                        if task_size > 0 {
                            let task = hash_cpu(
                                tx.clone(),
                                CpuTask {
                                    numeric_id: numeric_id,
                                    local_startnonce: start_nonce + requested,
                                    local_nonces: task_size,
                                    round: round.clone(),
                                },
                                simd_ext.clone(),
                            );
                            thread_pool.spawn(task);
                        }
                        requested += task_size;
                        print_status(processed, &sw, blocktime)
                    }
                    // schedule next gpu task
                    HasherMessage::GpuRequestForWork(id) => {
                        #[cfg(feature = "opencl")]
                        let task_size = min(gpus[id].worksize as u64, nonces_to_hash - requested);
                        #[cfg(not(feature = "opencl"))]
                        let task_size = 0;
                        #[cfg(feature = "opencl")]
                        gpu_channels[id]
                            .0
                            .send(Some(GpuTask {
                                numeric_id: numeric_id,
                                local_startnonce: start_nonce + requested,
                                local_nonces: task_size,
                                round: round.clone(),
                            }))
                            .unwrap();
                        requested += task_size;
                        print_status(processed, &sw, blocktime)
                    }
                    HasherMessage::NoncesProcessed(nonces) => {
                        processed += nonces;
                    }
                    HasherMessage::SubmitDeadline((height, nonce, deadline)) => {
                        // calc capcaity
                        let capacity = requested * 250 * blocktime / 1024 / (1 + sw.elapsed_ms()) as u64;
                        tx_nonce
                            .clone()
                            .unbounded_send(NonceData {
                                numeric_id,
                                nonce,
                                height,
                                deadline,
                                deadline_adjusted: deadline / round.base_target,
                                capacity,
                            })
                            .expect("failed to send nonce data");
                    }
                }
                if rx_rounds.len() > 0 {
                    break;
                }
            }
        }
    }
}

fn print_status(processed: u64, sw: &Stopwatch, blocktime: u64) {
    let datetime = Local::now();
    print!(
        "\r{} [STATS] nonces generated: {}, nonces/minute: {:.2}, emulated size={:.2} GiB",
        datetime.format("%H:%M:%S"),
        sw.elapsed_ms(),
        processed as f64 * 1000.0 * 60.0 / (1 + sw.elapsed_ms()) as f64,
        processed as f64 * 1000.0 / 4.0 / 1024.0 / (1 + sw.elapsed_ms()) as f64 * blocktime as f64,
    );
}
