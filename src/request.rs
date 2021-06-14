use crate::com::api::{FetchError, MiningInfoResponse};
use crate::com::client::{Client, ProxyDetails, SubmissionParameters};
use crate::future::prio_retry::PrioRetry;
use futures::future::Future;
use futures::stream::Stream;
use futures::sync::mpsc;
use std::collections::HashMap;
use std::time::Duration;
use std::u64;
use tokio;
use tokio::runtime::TaskExecutor;
use url::Url;
use stopwatch::Stopwatch;
use std::sync::Arc;

#[derive(Clone)]
pub struct RequestHandler {
    client: Client,
    tx_submit_data: mpsc::UnboundedSender<SubmissionParameters>,
}

impl RequestHandler {
    pub fn new(
        base_uri: Url,
        secret_phrase: String,
        timeout: u64,
        send_proxy_details: bool,
        additional_headers: Arc<HashMap<String, String>>,
        executor: TaskExecutor,
    ) -> RequestHandler {
        // TODO
        let proxy_details = if send_proxy_details {
            ProxyDetails::Enabled
        } else {
            ProxyDetails::Disabled
        };

        let client = Client::new(
            base_uri,
            secret_phrase,
            timeout,
            proxy_details,
            additional_headers,
        );

        let (tx_submit_data, rx_submit_nonce_data) = mpsc::unbounded();
        RequestHandler::handle_submissions(
            client.clone(),
            rx_submit_nonce_data,
            tx_submit_data.clone(),
            executor,
        );

        RequestHandler {
            client,
            tx_submit_data,
        }
    }

    fn handle_submissions(
        client: Client,
        rx: mpsc::UnboundedReceiver<SubmissionParameters>,
        tx_submit_data: mpsc::UnboundedSender<SubmissionParameters>,
        executor: TaskExecutor,
    ) {
        let stream = PrioRetry::new(rx, Duration::from_secs(3))
            .and_then(move |submission_params| {
                let tx_submit_data = tx_submit_data.clone();
                let mut sw = Stopwatch::new();
                sw.start();
                client
                    .clone()
                    .submit_nonce(&submission_params)
                    .then(move |res| {
                        sw.stop();
                        match res {
                            Ok(res) => {
                                if submission_params.deadline != res.deadline {
                                    log_deadline_mismatch(
                                        submission_params.height,
                                        submission_params.account_id,
                                        submission_params.nonce,
                                        submission_params.deadline,
                                        res.deadline,
                                        sw.elapsed_ms()
                                    );
                                } else {
                                    log_submission_accepted(
                                        submission_params.height,
                                        submission_params.account_id,
                                        submission_params.nonce,
                                        submission_params.deadline,
                                        sw.elapsed_ms()
                                    );
                                }
                            }
                            Err(FetchError::Pool(e)) => {
                                // Very intuitive, if some pools send an empty message they are
                                // experiencing too much load expect the submission to be resent later.
                                if e.message.is_empty() || e.message == "limit exceeded" {
                                    log_pool_busy(
                                        submission_params.height,
                                        submission_params.account_id,
                                        submission_params.nonce,
                                        submission_params.deadline,
                                        sw.elapsed_ms()
                                    );
                                    let res = tx_submit_data.unbounded_send(submission_params);
                                    if let Err(e) = res {
                                        error!("can't send submission params: {}", e);
                                    }
                                } else {
                                    log_submission_not_accepted(
                                        submission_params.height,
                                        submission_params.account_id,
                                        submission_params.nonce,
                                        submission_params.deadline,
                                        sw.elapsed_ms(),
                                        e.code,
                                        &e.message,
                                    );
                                }
                            }
                            Err(FetchError::Http(x)) => {
                                log_submission_failed(
                                    submission_params.height,
                                    submission_params.account_id,
                                    submission_params.nonce,
                                    submission_params.deadline,
                                    &x.to_string(),
                                );
                                let res = tx_submit_data.unbounded_send(submission_params);
                                if let Err(e) = res {
                                    error!("can't send submission params: {}", e);
                                }
                            }
                        };
                        Ok(())
                    })
            })
            .for_each(|_| Ok(()))
            .map_err(|e| error!("can't handle submission params: {:?}", e));
        executor.spawn(stream);
    }

    pub fn get_mining_info(&self, capacity: u64,  additional_headers: Arc<HashMap<String, String>>) -> impl Future<Item = MiningInfoResponse, Error = FetchError> {
        self.client.get_mining_info(capacity, additional_headers)
    }

    pub fn submit_nonce(
        &self,
        account_id: u64,
        nonce: u64,
        height: u64,
        block: u64,
        deadline_unadjusted: u64,
        deadline: u64,
        gen_sig: [u8; 32],
    ) {
        let res = self.tx_submit_data.unbounded_send(SubmissionParameters {
            account_id,
            nonce,
            height,
            block,
            deadline_unadjusted,
            deadline,
            gen_sig,
        });
        if let Err(e) = res {
            error!("can't send submission params: {}", e);
        }
    }
}

fn log_deadline_mismatch(
    height: u64,
    account_id: u64,
    nonce: u64,
    deadline: u64,
    deadline_pool: u64,
    latency: i64,
) {
    error!(
        "dl mismatch: height={}, id={}, nonce={}, \
         dl_miner={}, dl_pool={}, latency={}ms",
        height, account_id, nonce, deadline, deadline_pool, latency
    );
}

fn log_submission_failed(height: u64, account_id: u64, nonce: u64, deadline: u64, err: &str) {
    warn!(
        "{: <80}",
        format!(
            "submission failed, retrying: height={}, id={}, nonce={}, dl={}, response={}",
            height, account_id, nonce, deadline, err
        )
    );
}

fn log_submission_not_accepted(
    height: u64,
    account_id: u64,
    nonce: u64,
    deadline: u64,
    latency: i64,
    err_code: i32,
    msg: &str,
) {
    error!(
        "dl rejected: height={}, id={}, nonce={}, \
         dl={}, latency={}ms\n\tcode: {}\n\tmessage: {}",
        height, account_id, nonce, deadline, latency, err_code, msg,
    );
}

fn log_submission_accepted(height: u64, account_id: u64, nonce: u64, deadline: u64, latency: i64) {
    info!(
        "dl accepted: height={}, id={}, nonce={}, dl={}, latency={}ms",
        height, account_id, nonce, deadline, latency
    );
}

fn log_pool_busy(height: u64, account_id: u64, nonce: u64, deadline: u64, latency: i64) {
    info!(
        "pool busy, retrying: height={}, id={}, nonce={}, dl={}, latency={}ms",
        height, account_id, nonce, deadline, latency
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    static BASE_URL: &str = "http://94.130.178.37:31000";

    #[test]
    fn test_submit_nonce() {
        let rt = tokio::runtime::Runtime::new().expect("can't create runtime");

        let request_handler = RequestHandler::new(
            BASE_URL.parse().unwrap(),
            HashMap::new(),
            3,
            12,
            true,
            HashMap::new(),
            rt.executor(),
        );

        request_handler.submit_nonce(1337, 12, 111, 0, 7123, 1193, [0; 32]);

        rt.shutdown_on_idle();
    }
}
