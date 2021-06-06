use crate::ocl::{gpu_hash, GpuContext};
use crate::scheduler::{HasherMessage, RoundInfo};
use crossbeam_channel::{Receiver, Sender};
use std::sync::Arc;

pub struct GpuTask {
    pub numeric_id: u64,
    pub local_startnonce: u64,
    pub local_nonces: u64,
    pub round: RoundInfo,
}

pub fn create_gpu_hasher_thread(
    gpu_id: usize,
    gpu_context: Arc<GpuContext>,
    tx: Sender<HasherMessage>,
    rx_hasher_task: Receiver<Option<GpuTask>>,
) -> impl FnOnce() {
    move || {
        for task in rx_hasher_task {
            // check if new task or termination
            match task {
                // new task
                Some(task) => {
                    // gpu generate nonces
                    let (deadline, offset) = gpu_hash(&gpu_context, &task);

                    // report hashing done
                    tx.send(HasherMessage::NoncesProcessed(task.local_nonces))
                        .expect("GPU task can't communicate with scheduler thread.");

                    tx.send(HasherMessage::SubmitDeadline((
                        task.round.height,
                        task.local_startnonce + offset,
                        deadline,
                        task.round.block,
                    )))
                    .expect("GPU task can't communicate with scheduler thread.");

                    tx.send(HasherMessage::GpuRequestForWork(gpu_id))
                        .expect("GPU task can't communicate with scheduler thread.");
                }
                // termination
                None => {
                    break;
                }
            }
        }
    }
}
