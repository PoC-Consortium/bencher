use crate::com::api::MiningInfoResponse as MiningInfo;
use crate::config::Cfg;
use crate::cpu_hasher::SimdExtension;
use crate::future::interval::Interval;
#[cfg(feature = "opencl")]
use crate::ocl::GpuConfig;
use crate::poc_hashing;
use crate::request::RequestHandler;
use crate::scheduler::create_scheduler_thread;
use crate::scheduler::RoundInfo;
use crossbeam_channel::unbounded;
use futures::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use std::u64;
use tokio::prelude::*;
use tokio::runtime::TaskExecutor;
use std::cmp::{max};

const GENESIS_BASE_TARGET: u64 = 4_398_046_511_104;

pub struct Miner {
    executor: TaskExecutor,
    request_handler: RequestHandler,
    cpu_threads: usize,
    cpu_worker_task_size: u64,
    simd_extensions: SimdExtension,
    numeric_id: u64,
    start_nonce: u64,
    target_deadline: u64,
    blocktime: u64,
    gpus: Vec<GpuConfig>,
    get_mining_info_interval: u64,
}

pub struct State {
    generation_signature: String,
    generation_signature_bytes: [u8; 32],
    base_target: u64,
    height: u64,
    block: u64,
    server_target_deadline: u64,
    first: bool,
    outage: bool,
    best_deadline: u64,
    scoop: u32,
    capacity: u64,
}

impl State {
    fn new() -> Self {
        Self {
            generation_signature: "".to_owned(),
            generation_signature_bytes: [0; 32],
            base_target: 1,
            height: 0,
            block: 0,
            server_target_deadline: u64::MAX,
            first: true,
            outage: false,
            best_deadline: u64::MAX,
            scoop: 0,
            capacity: 0,
        }
    }

    fn update_mining_info(&mut self, mining_info: &MiningInfo) {
        self.best_deadline= u64::MAX;
        self.height = mining_info.height;
        self.block += 1;
        self.base_target = mining_info.base_target;
        self.server_target_deadline = mining_info.target_deadline;

        self.generation_signature_bytes =
            poc_hashing::decode_gensig(&mining_info.generation_signature);
        self.generation_signature = mining_info.generation_signature.clone();

        let scoop =
            poc_hashing::calculate_scoop(mining_info.height, &self.generation_signature_bytes);
        info!(
            "{: <80}",
            format!(
                "new block: height={}, scoop={}, netdiff={}",
                mining_info.height,
                scoop,
                GENESIS_BASE_TARGET / 240 / mining_info.base_target,
            )
        );
        self.scoop = scoop;
    }
}

#[derive(Clone)]
pub struct NonceData {
    pub numeric_id: u64,
    pub nonce: u64,
    pub height: u64,
    pub block: u64,
    pub deadline: u64,
    pub deadline_adjusted: u64,
    pub capacity: u64,
    pub base_target: u64
}

impl Miner {
    pub fn new(
        cfg: Cfg,
        simd_extensions: SimdExtension,
        cpu_threads: usize,
        executor: TaskExecutor,
    ) -> Miner {
        info!("server: {}", cfg.url);
        let request_handler = RequestHandler::new(
            cfg.url,
            cfg.secret_phrase,
            cfg.timeout,
            cfg.send_proxy_details,
            cfg.additional_headers,
            executor.clone(),
        );

        Miner {
            executor,
            request_handler,
            cpu_threads,
            cpu_worker_task_size: cfg.cpu_worker_task_size,
            simd_extensions,
            numeric_id: cfg.numeric_id,
            start_nonce: cfg.start_nonce,
            target_deadline: cfg.target_deadline,
            blocktime: cfg.blocktime,
            gpus: cfg.gpus,
            get_mining_info_interval: max(1000, cfg.get_mining_info_interval),
        }
    }

    pub fn run(self) {
        // create channels
        let (tx_rounds, rx_rounds) = unbounded();
        let (tx_nonce_data, rx_nonce_data) = mpsc::unbounded();

        // create hasher thread
        thread::spawn(create_scheduler_thread(
            self.numeric_id,
            self.start_nonce,
            self.cpu_threads as u8,
            self.cpu_worker_task_size,
            self.simd_extensions.clone(),
            self.gpus,
            self.blocktime,
            rx_rounds.clone(),
            tx_nonce_data.clone(),
        ));

        let state = Arc::new(Mutex::new(State::new()));

        let request_handler = self.request_handler.clone();
        let inner_state = state.clone();
        let inner_tx_rounds = tx_rounds.clone();
        let get_mining_info_interval = self.get_mining_info_interval;
        // run main mining loop on core
        self.executor.clone().spawn(
            Interval::new_interval(Duration::from_millis(get_mining_info_interval))
                .for_each(move |_| {
                    let state = inner_state.clone();
                    let state2 = inner_state.clone();
                    let state2 = state2.lock().unwrap();
                    let capacity = state2.capacity;
                    drop(state2);
                    let tx_rounds = inner_tx_rounds.clone();
                    request_handler.get_mining_info(capacity).then(move |mining_info| {
                        match mining_info {
                            Ok(mining_info) => {
                                let mut state = state.lock().unwrap();
                                state.first = false;
                                if state.outage {
                                    error!("{: <80}", "outage resolved.");
                                    state.outage = false;
                                }
                                if mining_info.generation_signature != state.generation_signature {
                                    state.update_mining_info(&mining_info);
                                   
                                    // communicate new round hasher
                                    tx_rounds
                                        .send(RoundInfo {
                                            gensig: state.generation_signature_bytes,
                                            base_target: state.base_target,
                                            scoop: state.scoop.into(),
                                            height: state.height,
                                            block: state.block,
                                        })
                                        .expect("main thread can't communicate with hasher thread");
                                }
                            }
                            _ => {
                                let mut state = state.lock().unwrap();
                                if state.first {
                                    error!(
                                        "{: <80}",
                                        "error getting mining info, please check server config"
                                    );
                                    state.first = false;
                                    state.outage = true;
                                } else {
                                    if !state.outage {
                                        error!(
                                            "{: <80}",
                                            "error getting mining info => connection outage..."
                                        );
                                    }
                                    state.outage = true;
                                }
                            }
                        }
                        future::ok(())
                    })
                })
                .map_err(|e| panic!("interval errored: err={:?}", e)),
        );

        let target_deadline = self.target_deadline;
        let request_handler = self.request_handler.clone();
        let state = state.clone();
        self.executor.clone().spawn(
            rx_nonce_data
                .for_each(move |nonce_data| {
                    let mut state = state.lock().unwrap();
                    state.capacity = nonce_data.capacity;
                    let deadline = nonce_data.deadline / nonce_data.base_target;
                    if state.block == nonce_data.block {
                        if state.best_deadline > nonce_data.deadline_adjusted
                            && nonce_data.deadline_adjusted < target_deadline
                        {
                            state.best_deadline = nonce_data.deadline_adjusted;
                            request_handler.submit_nonce(
                                nonce_data.numeric_id,
                                nonce_data.nonce,
                                nonce_data.height,
                                nonce_data.block,
                                nonce_data.deadline,
                                deadline,
                                state.generation_signature_bytes,
                            );
                        }
                    }
                    Ok(())
                })
                .map_err(|e| panic!("interval errored: err={:?}", e)),
        );
    }
}
