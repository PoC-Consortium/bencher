use crate::ocl::GpuConfig;
use rand::Rng;
use std::collections::HashMap;
use std::fs;
use std::u32;
use url::Url;

#[derive(Debug, Serialize, Deserialize)]
pub struct Cfg {
    #[serde(default = "default_numeric_id")]
    pub numeric_id: u64,

    #[serde(default = "default_start_nonce")]
    pub start_nonce: u64,

    #[serde(default = "default_secret_phrase")]
    pub secret_phrase: String,

    #[serde(default = "default_blocktime")]
    pub blocktime: u64,

    #[serde(with = "url_serde")]
    pub url: Url,

    #[serde(default = "default_gpus")]
    pub gpus: Vec<GpuConfig>,

    #[serde(default = "default_cpu_threads")]
    pub cpu_threads: usize,

    #[serde(default = "default_cpu_task_size")]
    pub cpu_worker_task_size: u64,

    #[serde(default = "default_cpu_thread_pinning")]
    pub cpu_thread_pinning: bool,

    #[serde(default = "default_target_deadline")]
    pub target_deadline: u64,

    #[serde(default = "default_get_mining_info_interval")]
    pub get_mining_info_interval: u64,

    #[serde(default = "default_timeout")]
    pub timeout: u64,

    #[serde(default = "default_send_proxy_details")]
    pub send_proxy_details: bool,

    #[serde(default = "default_additional_headers")]
    pub additional_headers: HashMap<String, String>,

    #[serde(default = "default_console_log_level")]
    pub console_log_level: String,

    #[serde(default = "default_logfile_log_level")]
    pub logfile_log_level: String,

    #[serde(default = "default_logfile_max_count")]
    pub logfile_max_count: u32,

    #[serde(default = "default_logfile_max_size")]
    pub logfile_max_size: u64,

    #[serde(default = "default_console_log_pattern")]
    pub console_log_pattern: String,

    #[serde(default = "default_logfile_log_pattern")]
    pub logfile_log_pattern: String,
}

fn default_numeric_id() -> u64 {
    //hi bold!
    7900104405094198526
}

fn default_start_nonce() -> u64 {
    let mut rng = rand::thread_rng();
    u64::from(rng.gen::<u32>())
}

fn default_secret_phrase() -> String {
    "".to_owned()
}

fn default_blocktime() -> u64 {
    240
}

fn default_cpu_threads() -> usize {
    0
}

fn default_cpu_task_size() -> u64 {
    64
}

fn default_cpu_thread_pinning() -> bool {
    false
}

fn default_gpus() -> Vec<GpuConfig> {
    Vec::new()
}

fn default_target_deadline() -> u64 {
    u64::from(u32::MAX)
}

fn default_get_mining_info_interval() -> u64 {
    3000
}

fn default_timeout() -> u64 {
    5000
}

fn default_send_proxy_details() -> bool {
    false
}

fn default_additional_headers() -> HashMap<String, String> {
    HashMap::new()
}

fn default_console_log_level() -> String {
    "Info".to_owned()
}

fn default_logfile_log_level() -> String {
    "Warn".to_owned()
}

fn default_logfile_max_count() -> u32 {
    10
}

fn default_logfile_max_size() -> u64 {
    20
}

fn default_console_log_pattern() -> String {
    "\r{d(%H:%M:%S.%3f%z)} [{h({l}):<5}] [{T}] [{t}] - {M}:{m}{n}".to_owned()
}

fn default_logfile_log_pattern() -> String {
    "\r{d(%Y-%m-%dT%H:%M:%S.%3f%z)} [{h({l}):<5}] [{T}] [{f}:{L}] [{t}] - {M}:{m}{n}".to_owned()
}

pub fn load_cfg(config: &str) -> Cfg {
    let cfg_str =
        fs::read_to_string(config).expect(&format!("failed to open config, config={}", config));
    let cfg: Cfg = serde_yaml::from_str(&cfg_str).expect("failed to parse config");
    cfg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_cfg() {
        let cfg = load_cfg("config.yaml");
        assert_eq!(cfg.timeout, 3000);
    }
}
