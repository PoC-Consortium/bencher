#![warn(unused_extern_crates)]
#[macro_use]
extern crate serde_derive;
#[macro_use]
extern crate clap;
#[macro_use]
extern crate log;

mod com;
mod future;
mod buffer;
mod config;
mod cpu_hasher;
#[cfg(feature = "opencl")]
mod gpu_hasher;
mod logger;
mod miner;
#[cfg(feature = "opencl")]
mod ocl;
mod poc_hashing;
mod request;
mod scheduler;
mod shabal256;

use crate::config::load_cfg;
use crate::cpu_hasher::{init_cpu_extensions, SimdExtension};
use crate::miner::Miner;
use clap::{App, Arg};
use futures::Future;
use std::cmp::min;
use std::process;
use tokio::runtime::Builder;

#[cfg(feature = "opencl")]
use crate::ocl::gpu_get_info;

fn main() {
    let arg = App::new("Bencher - a PoW PoC miner")
        .version(crate_version!())
        .author(crate_authors!())
        .about(crate_description!())
        .arg(
            Arg::with_name("config")
                .short("c")
                .long("config")
                .value_name("FILE")
                .help("Location of the config file")
                .takes_value(true)
                .default_value("config.yaml"),
        );
    #[cfg(feature = "opencl")]
    let arg = arg.arg(
        Arg::with_name("opencl")
            .short("ocl")
            .long("opencl")
            .help("Display OpenCL platforms and devices")
            .takes_value(false),
    );

    let matches = &arg.get_matches();
    let config = matches.value_of("config").unwrap();

    let cfg_loaded = load_cfg(config);
    logger::init_logger(&cfg_loaded);

    info!("bencher v.{}", crate_version!());

    if matches.is_present("opencl") {
        #[cfg(feature = "opencl")]
        ocl::platform_info();
        process::exit(0);
    }

    let simd_extension = init_cpu_extensions();
    let cpuid = raw_cpuid::CpuId::new();
    let cpu_name = cpuid.get_extended_function_info().unwrap();
    let cpu_name = cpu_name.processor_brand_string().unwrap().trim();

    #[cfg(not(feature = "opencl"))]
    let cpu_threads = if cfg_loaded.cpu_threads == 0 {
        num_cpus::get()
    } else {
        min(cfg_loaded.cpu_threads, 2 * num_cpus::get()) // 2x just in case num_cpus doesnt cope with multi cpu
    };

    // special case: dont use cpu if only a gpu is defined
    #[cfg(feature = "opencl")]
    let cpu_threads = if matches.occurrences_of("gpu") > 0 && matches.occurrences_of("cpu") == 0 {
        0
    } else {
        min(cfg_loaded.cpu_threads, 2 * num_cpus::get()) // 2x just in case num_cpus doesnt cope with multi cpu
    };

    info!(
        "cpu: {} [using {} of {} cores{}{:?}]",
        cpu_name,
        cpu_threads,
        num_cpus::get(),
        if let SimdExtension::None = &simd_extension {
            ""
        } else {
            " + "
        },
        &simd_extension
    );

    let mut cpu_string = format!(
        "cpu: {} [using {} of {} cores{}{:?}]",
        cpu_name,
        cpu_threads,
        num_cpus::get(),
        if let SimdExtension::None = &simd_extension {
            ""
        } else {
            " + "
        },
        &simd_extension
    );

    cpu_string.push_str(&", ".to_owned());

    #[cfg(not(feature = "opencl"))]
    let gpu_string = "".to_owned();
    #[cfg(not(feature = "opencl"))]
    let gpu_mem_needed = 0u64;    
    #[cfg(feature = "opencl")]
    let (_gpu_mem_needed, gpu_string) = if !cfg_loaded.gpus.is_empty() {
        gpu_get_info(&cfg_loaded.gpus, false)
    } else {
        (0,"".to_owned())
    };
    cpu_string.push_str(&gpu_string);
        
    #[cfg(feature = "opencl")]
    info!("gpu extensions: OpenCL");

    info!("numeric_id: {}", cfg_loaded.numeric_id);
    info!("start_nonce: {}", cfg_loaded.start_nonce);
    info!("target_deadline: {}", cfg_loaded.target_deadline);
    info!(
        "mode: {}",
        if cfg_loaded.secret_phrase != "".to_owned() {
            "solo"
        } else {
            "pool"
        }
    );

    let rt = Builder::new().core_threads(1).build().unwrap();
    let m = Miner::new(cfg_loaded, simd_extension, cpu_threads, rt.executor(), cpu_string);
    m.run();
    rt.shutdown_on_idle().wait().unwrap();
}
