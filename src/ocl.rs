use self::core::{
    ArgVal, ContextProperties, DeviceInfo, Event, KernelWorkGroupInfo, PlatformInfo, Status,
};
use crate::gpu_hasher::GpuTask;
use crate::poc_hashing::NONCE_SIZE;
use ocl_core as core;
use std::cmp::min;
use std::ffi::CString;
use std::process;
use std::sync::Arc;
use std::u64;

static SRC: &'static str = include_str!("ocl/kernel.cl");

const GPU_HASHES_PER_RUN: usize = 32;

// convert the info or error to a string for printing:
macro_rules! to_string {
    ($expr:expr) => {
        match $expr {
            Ok(info) => info.to_string(),
            Err(err) => match err.api_status() {
                Some(Status::CL_KERNEL_ARG_INFO_NOT_AVAILABLE) => "Not available".into(),
                _ => err.to_string(),
            },
        }
    };
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GpuConfig {
    platform_id: usize,
    device_id: usize,
    cores: usize,
}

//#[allow(dead_code)]
pub struct GpuContext {
    queue: core::CommandQueue,
    kernel0: core::Kernel,
    ldim0: [usize; 3],
    gdim0: [usize; 3],
    kernel1: core::Kernel,
    kernel2: core::Kernel,
    buffer_gpu: core::Mem,
    gensig_gpu: core::Mem,
    pub worksize: usize,
    deadlines_gpu: core::Mem,
    best_deadline_gpu: core::Mem,
    best_offset_gpu: core::Mem,
}

// Ohne Gummi im Bahnhofsviertel... das wird noch Konsequenzen haben
unsafe impl Sync for GpuContext {}

impl GpuContext {
    pub fn new(gpu_platform: usize, gpu_id: usize, cores: usize) -> GpuContext {
        let platform_ids = core::get_platform_ids().unwrap();
        let platform_id = platform_ids[gpu_platform];
        let device_ids = core::get_device_ids(&platform_id, None, None).unwrap();
        let device_id = device_ids[gpu_id];
        let context_properties = ContextProperties::new().platform(platform_id);
        let context =
            core::create_context(Some(&context_properties), &[device_id], None, None).unwrap();
        let src_cstring = CString::new(SRC).unwrap();
        let program = core::create_program_with_source(&context, &[src_cstring]).unwrap();
        core::build_program(
            &program,
            None::<&[()]>,
            &CString::new("").unwrap(),
            None,
            None,
        )
        .unwrap();
        let queue = core::create_command_queue(&context, &device_id, None).unwrap();
        let kernel0 = core::create_kernel(&program, "noncegen").unwrap();
        let kernel_workgroup_size = get_kernel_work_group_size(&kernel0, device_id);
        let workgroup_count = cores;
        let worksize = kernel_workgroup_size * workgroup_count;
        let gdim0 = [worksize, 1, 1];
        let ldim0 = [kernel_workgroup_size, 1, 1];
        let kernel1 = core::create_kernel(&program, "calculate_deadlines").unwrap();
        let kernel2 = core::create_kernel(&program, "find_min").unwrap();

        // create buffers
        let buffer_gpu = unsafe {
            core::create_buffer::<_, u8>(
                &context,
                core::MEM_READ_WRITE,
                (NONCE_SIZE as usize) * worksize,
                None,
            )
            .expect("can't create gpu buffer")
        };

        let gensig_gpu = unsafe {
            core::create_buffer::<_, u8>(&context, core::MEM_READ_ONLY, 32, None).unwrap()
        };

        let deadlines_gpu = unsafe {
            core::create_buffer::<_, u64>(&context, core::MEM_READ_WRITE, gdim0[0], None).unwrap()
        };

        let best_offset_gpu = unsafe {
            core::create_buffer::<_, u64>(&context, core::MEM_READ_WRITE, 1, None).unwrap()
        };

        let best_deadline_gpu = unsafe {
            core::create_buffer::<_, u64>(&context, core::MEM_READ_WRITE, 1, None).unwrap()
        };

        GpuContext {
            queue,
            kernel0,
            ldim0,
            gdim0,
            kernel1,
            kernel2,
            buffer_gpu,
            gensig_gpu,
            worksize,
            deadlines_gpu,
            best_deadline_gpu,
            best_offset_gpu,
        }
    }
}

pub fn platform_info() {
    let platform_ids = core::get_platform_ids().unwrap();
    for (i, platform_id) in platform_ids.iter().enumerate() {
        info!(
            "OCL: platform {}, {} - {}",
            i,
            to_string!(core::get_platform_info(&platform_id, PlatformInfo::Name)),
            to_string!(core::get_platform_info(&platform_id, PlatformInfo::Version))
        );
        let device_ids = core::get_device_ids(&platform_id, None, None).unwrap();
        let context_properties = ContextProperties::new().platform(*platform_id);
        for (j, device_id) in device_ids.iter().enumerate() {
            info!(
                "OCL:   device {}, {} - {}",
                j,
                to_string!(core::get_device_info(device_id, DeviceInfo::Vendor)),
                to_string!(core::get_device_info(device_id, DeviceInfo::Name))
            );

            let context =
                core::create_context(Some(&context_properties), &[*device_id], None, None).unwrap();
            let src_cstring = CString::new(SRC).unwrap();
            let program = core::create_program_with_source(&context, &[src_cstring]).unwrap();
            core::build_program(
                &program,
                None::<&[()]>,
                &CString::new("").unwrap(),
                None,
                None,
            )
            .unwrap();
            let kernel = core::create_kernel(&program, "noncegen").unwrap();
            let cores = get_cores(*device_id) as usize;
            let kernel_workgroup_size = get_kernel_work_group_size(&kernel, *device_id);
            info!(
                "OCL:     cores={},kernel_workgroupsize={}",
                cores, kernel_workgroup_size
            );
        }
        info!("OCL:");
    }
}

fn get_cores(device: core::DeviceId) -> u32 {
    match core::get_device_info(device, DeviceInfo::MaxComputeUnits).unwrap() {
        core::DeviceInfoResult::MaxComputeUnits(mcu) => mcu,
        _ => panic!("Unexpected error"),
    }
}

pub fn gpu_get_info(gpus: &[GpuConfig], quiet: bool) -> u64 {
    let mut total_mem_needed = 0u64;
    for gpu in gpus.iter() {
        let platform_ids = core::get_platform_ids().unwrap();
        if gpu.platform_id >= platform_ids.len() {
            println!("Error: Selected OpenCL platform doesn't exist.");
            println!("Shutting down...");
            process::exit(0);
        }
        let platform = platform_ids[gpu.platform_id];
        let device_ids = core::get_device_ids(&platform, None, None).unwrap();
        if gpu.device_id >= device_ids.len() {
            println!("Error: Selected OpenCL device doesn't exist");
            println!("Shutting down...");
            process::exit(0);
        }
        let device = device_ids[gpu.device_id];
        let max_compute_units =
            match core::get_device_info(&device, DeviceInfo::MaxComputeUnits).unwrap() {
                core::DeviceInfoResult::MaxComputeUnits(mcu) => mcu,
                _ => panic!("Unexpected error. Can't obtain number of GPU cores."),
            };
        let mem = match core::get_device_info(&device, DeviceInfo::GlobalMemSize).unwrap() {
            core::DeviceInfoResult::GlobalMemSize(gms) => gms,
            _ => panic!("Unexpected error. Can't obtain GPU memory size."),
        };

        // get work_group_size for kernel
        let context_properties = ContextProperties::new().platform(platform);
        let context =
            core::create_context(Some(&context_properties), &[device], None, None).unwrap();
        let src_cstring = CString::new(SRC).unwrap();
        let program = core::create_program_with_source(&context, &[src_cstring]).unwrap();
        core::build_program(
            &program,
            None::<&[()]>,
            &CString::new("").unwrap(),
            None,
            None,
        )
        .unwrap();
        let kernel = core::create_kernel(&program, "noncegen").unwrap();
        let kernel_workgroup_size = get_kernel_work_group_size(&kernel, device);

        let gpu_cores = if gpu.cores == 0 {
            max_compute_units as usize
        } else {
            min(gpu.cores, max_compute_units as usize)
        };
        let mem_needed = gpu_cores * kernel_workgroup_size * 256 * 1024;

        if mem_needed > mem as usize {
            println!("Error: Not enough GPU-memory. Please reduce number of cores.");
            println!("Shutting down...");
            process::exit(0);
        }

        if !quiet {
            info!(
                "gpu: {} - {} [using {} of {} cores]",
                to_string!(core::get_device_info(&device, DeviceInfo::Vendor)),
                to_string!(core::get_device_info(&device, DeviceInfo::Name)),
                gpu_cores,
                max_compute_units
            );
        }
        if !quiet {
            info!(
                "gpu-ram: Total={:.2} MiB, Usage={:.2} MiB",
                mem / 1024 / 1024,
                mem_needed / 1024 / 1024,
            );
        }

        total_mem_needed += mem_needed as u64 / 2;
    }
    total_mem_needed
}

pub fn gpu_init(gpus: &[GpuConfig]) -> Vec<Arc<GpuContext>> {
    let mut result = Vec::new();
    for gpu in gpus.iter() {
        let platform_ids = core::get_platform_ids().unwrap();
        if gpu.platform_id >= platform_ids.len() {
            println!("Error: Selected OpenCL platform doesn't exist.");
            println!("Shutting down...");
            process::exit(0);
        }
        let platform = platform_ids[gpu.platform_id];
        let device_ids = core::get_device_ids(&platform, None, None).unwrap();
        if gpu.device_id >= device_ids.len() {
            println!("Error: Selected OpenCL device doesn't exist");
            println!("Shutting down...");
            process::exit(0);
        }
        let device = device_ids[gpu.device_id];
        let max_compute_units =
            match core::get_device_info(&device, DeviceInfo::MaxComputeUnits).unwrap() {
                core::DeviceInfoResult::MaxComputeUnits(mcu) => mcu,
                _ => panic!("Unexpected error. Can't obtain number of GPU cores."),
            };
        let gpu_cores = if gpu.cores == 0 {
            max_compute_units as usize
        } else {
            min(gpu.cores, 2 * max_compute_units as usize)
        };
        result.push(Arc::new(GpuContext::new(
            gpu.platform_id,
            gpu.device_id,
            gpu_cores,
        )));
    }
    result
}

fn get_kernel_work_group_size(x: &core::Kernel, y: core::DeviceId) -> usize {
    let kwg = core::get_kernel_work_group_info(x, y, KernelWorkGroupInfo::WorkGroupSize);
    if let Ok(kwg) = kwg {
        match kwg {
            core::KernelWorkGroupInfoResult::WorkGroupSize(kws) => kws,
            _ => panic!("Unexpected error"),
        }
    } else {
        0
    }
}

pub fn gpu_hash(gpu_context: &Arc<GpuContext>, task: &GpuTask) -> (u64, u64) {
    let numeric_id_be: u64 = task.numeric_id.to_be();

    let mut start;
    let mut end;

    core::set_kernel_arg(&gpu_context.kernel0, 0, ArgVal::mem(&gpu_context.buffer_gpu)).unwrap();
    core::set_kernel_arg(
        &gpu_context.kernel0,
        1,
        ArgVal::primitive(&task.local_startnonce),
    )
    .unwrap();
    core::set_kernel_arg(
        &gpu_context.kernel0,
        5,
        ArgVal::primitive(&task.local_nonces),
    )
    .unwrap();
    core::set_kernel_arg(&gpu_context.kernel0, 2, ArgVal::primitive(&numeric_id_be)).unwrap();

    for i in (0..8192).step_by(GPU_HASHES_PER_RUN) {
        if i + GPU_HASHES_PER_RUN < 8192 {
            start = i;
            end = i + GPU_HASHES_PER_RUN - 1;
        } else {
            start = i;
            end = i + GPU_HASHES_PER_RUN;
        }

        core::set_kernel_arg(&gpu_context.kernel0, 3, ArgVal::primitive(&(start as i32))).unwrap();
        core::set_kernel_arg(&gpu_context.kernel0, 4, ArgVal::primitive(&(end as i32))).unwrap();

        unsafe {
            core::enqueue_kernel(
                &gpu_context.queue,
                &gpu_context.kernel0,
                1,
                None,
                &gpu_context.gdim0,
                Some(gpu_context.ldim0),
                None::<Event>,
                None::<&mut Event>,
            )
            .unwrap();
        }
    }
    core::finish(&gpu_context.queue).unwrap();

    upload_gensig(&gpu_context, task.round.gensig, true);

    // calc deadline

    core::set_kernel_arg(
        &gpu_context.kernel1,
        0,
        ArgVal::mem(&gpu_context.gensig_gpu),
    )
    .unwrap();
    core::set_kernel_arg(&gpu_context.kernel1, 1, ArgVal::mem(&gpu_context.buffer_gpu)).unwrap();
    core::set_kernel_arg(
        &gpu_context.kernel1,
        2,
        ArgVal::mem(&gpu_context.deadlines_gpu),
    )
    .unwrap();

    core::set_kernel_arg(
        &gpu_context.kernel1,
        3,
        ArgVal::primitive(&task.round.scoop),
    )
    .unwrap();

    unsafe {
        core::enqueue_kernel(
            &gpu_context.queue,
            &gpu_context.kernel1,
            1,
            None,
            &gpu_context.gdim0,
            Some(gpu_context.ldim0),
            None::<Event>,
            None::<&mut Event>,
        )
        .unwrap();
    }

    core::set_kernel_arg(
        &gpu_context.kernel2,
        0,
        ArgVal::mem(&gpu_context.deadlines_gpu),
    )
    .unwrap();
    core::set_kernel_arg(
        &gpu_context.kernel2,
        1,
        ArgVal::primitive(&(task.local_nonces)),
    )
    .unwrap();
    core::set_kernel_arg(
        &gpu_context.kernel2,
        2,
        ArgVal::local::<u32>(&gpu_context.ldim0[0]),
    )
    .unwrap();
    core::set_kernel_arg(
        &gpu_context.kernel2,
        3,
        ArgVal::mem(&gpu_context.best_offset_gpu),
    )
    .unwrap();
    core::set_kernel_arg(
        &gpu_context.kernel2,
        4,
        ArgVal::mem(&gpu_context.best_deadline_gpu),
    )
    .unwrap();

    unsafe {
        core::enqueue_kernel(
            &gpu_context.queue,
            &gpu_context.kernel2,
            1,
            None,
            &gpu_context.gdim0,
            Some(gpu_context.ldim0),
            None::<Event>,
            None::<&mut Event>,
        )
        .unwrap();
    }
    
    get_result(&gpu_context)

}

pub fn get_result(gpu_context: &Arc<GpuContext>) -> (u64, u64) {
    let mut best_offset = vec![0u64; 1];
    let mut best_deadline = vec![0u64; 1];

    unsafe {
        core::enqueue_read_buffer(
            &gpu_context.queue,
            &gpu_context.best_offset_gpu,
            true,
            0,
            &mut best_offset,
            None::<Event>,
            None::<&mut Event>,
        )
        .unwrap();
    }
    unsafe {
        core::enqueue_read_buffer(
            &gpu_context.queue,
            &gpu_context.best_deadline_gpu,
            true,
            0,
            &mut best_deadline,
            None::<Event>,
            None::<&mut Event>,
        )
        .unwrap();
    }

    (best_deadline[0], best_offset[0])
}

fn upload_gensig(gpu_context: &Arc<GpuContext>, gensig: [u8; 32], blocking: bool) {
    unsafe {
        core::enqueue_write_buffer(
            &gpu_context.queue,
            &gpu_context.gensig_gpu,
            blocking,
            0,
            &gensig,
            None::<Event>,
            None::<&mut Event>,
        )
        .unwrap();
    }
}