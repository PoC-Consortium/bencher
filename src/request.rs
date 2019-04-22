use crate::miner::NonceData;
use bytes::Buf;
use futures::future::Future;
use futures::stream::Stream;
use futures::{future, stream};
use reqwest::header::HeaderName;
use reqwest::r#async::{Chunk, ClientBuilder, Decoder, Request};
use serde::de::{self, DeserializeOwned};
use std::collections::HashMap;
use std::fmt;
use std::mem;
use std::time::Duration;
use std::u64;
use url::form_urlencoded::byte_serialize;
use url::Url;

#[derive(Clone)]
pub struct RequestHandler {
    secret_phrase: String,
    base_uri: Url,
    timeout: Duration,
    timeout_mi: Duration,
    send_proxy_details: bool,
    headers: reqwest::header::HeaderMap,
}

pub enum FetchError {
    Http(reqwest::Error),
    Pool(PoolError),
}

impl From<reqwest::Error> for FetchError {
    fn from(err: reqwest::Error) -> FetchError {
        FetchError::Http(err)
    }
}

impl From<PoolError> for FetchError {
    fn from(err: PoolError) -> FetchError {
        FetchError::Pool(err)
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MiningInfo {
    pub generation_signature: String,

    #[serde(deserialize_with = "from_str_or_int")]
    pub base_target: u64,

    #[serde(deserialize_with = "from_str_or_int")]
    pub height: u64,

    #[serde(
        default = "default_target_deadline",
        deserialize_with = "from_str_or_int"
    )]
    pub target_deadline: u64,
}

fn default_target_deadline() -> u64 {
    u64::MAX
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SubmitNonceResponse {
    pub deadline: u64,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PoolErrorWrapper {
    error: PoolError,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PoolError {
    code: i32,
    message: String,
}

// MOTHERFUCKING pool
fn from_str_or_int<'de, D>(deserializer: D) -> Result<u64, D::Error>
where
    D: de::Deserializer<'de>,
{
    struct StringOrIntVisitor;

    impl<'de> de::Visitor<'de> for StringOrIntVisitor {
        type Value = u64;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("string or int")
        }

        fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
            v.parse::<u64>().map_err(de::Error::custom)
        }

        fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E> {
            Ok(v)
        }
    }

    deserializer.deserialize_any(StringOrIntVisitor)
}

impl RequestHandler {
    pub fn new(
        base_uri: Url,
        mut secret_phrase: String,
        timeout: u64,
        timeout_mi: u64,
        send_proxy_details: bool,
        additional_headers: HashMap<String, String>,
    ) -> RequestHandler {
        let ua = "Bencher/".to_owned() + "1.0.0";

        secret_phrase = byte_serialize(secret_phrase.as_bytes()).collect();
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("User-Agent", ua.to_owned().parse().unwrap());
        if send_proxy_details {
            headers.insert("X-Miner", ua.to_owned().parse().unwrap());
            headers.insert(
                "X-Minername",
                hostname::get_hostname()
                    .unwrap_or_else(|| "".to_owned())
                    .parse()
                    .unwrap(),
            );
            headers.insert(
                "X-Plotfile",
                ("ScavengerProxy/".to_owned()
                    + &*hostname::get_hostname().unwrap_or_else(|| "".to_owned()))
                    .parse()
                    .unwrap(),
            );
        }

        for (key, value) in additional_headers {
            let header_name = HeaderName::from_bytes(&key.into_bytes()).unwrap();
            headers.insert(header_name, value.parse().unwrap());
        }

        RequestHandler {
            secret_phrase,
            base_uri,
            timeout: Duration::from_millis(timeout),
            timeout_mi: Duration::from_millis(timeout_mi),
            send_proxy_details,
            headers,
        }
    }

    fn uri_for(&self, path: &str, query: &str) -> Url {
        let mut url = self.base_uri.clone();
        url.path_segments_mut()
            .map_err(|_| "cannot be base")
            .unwrap()
            .pop_if_empty()
            .push(path);
        url.set_query(Some(query));
        url
    }

    pub fn get_mining_info(&self) -> impl Future<Item = MiningInfo, Error = FetchError> {
        do_req(
            self.uri_for("burst", "requestType=getMiningInfo"),
            reqwest::Method::GET,
            self.headers.clone(),
            self.timeout_mi,
        )
    }

    pub fn submit_nonce(&self, nonce_data: NonceData) -> impl Future<Item = (), Error = ()> {
        let mut query = format!(
            "requestType=submitNonce&accountId={}&nonce={}&secretPhrase={}&blockheight={}",
            nonce_data.numeric_id, nonce_data.nonce, &self.secret_phrase, nonce_data.height
        );
        // if pool mining also send the deadline (usefull for proxies)
        if self.secret_phrase == "" {
            query += &format!("&deadline={}", nonce_data.deadline);
        }

        let url = self.uri_for("burst", &query);
        let timeout = self.timeout;
        let mut headers = self.headers.clone();

        headers.insert("X-Capacity", nonce_data.capacity.to_string().parse().unwrap());

        stream::iter_ok(1..=3)
            .and_then(move |retry| {
                let nonce_data = nonce_data.clone();
                do_req(url.clone(), reqwest::Method::POST, headers.clone(), timeout).then(
                    move |res: Result<SubmitNonceResponse, FetchError>| match res {
                        Ok(res) => {
                            if nonce_data.deadline_adjusted != res.deadline {
                                log_deadline_mismatch(
                                    nonce_data.height,
                                    nonce_data.numeric_id,
                                    nonce_data.nonce,
                                    nonce_data.deadline_adjusted,
                                    res.deadline,
                                );
                            } else {
                                log_submission_accepted(
                                    nonce_data.numeric_id,
                                    nonce_data.nonce,
                                    nonce_data.deadline_adjusted,
                                );
                            }
                            Ok(true)
                        }
                        Err(FetchError::Pool(e)) => {
                            log_submission_not_accepted(
                                nonce_data.height,
                                nonce_data.numeric_id,
                                nonce_data.nonce,
                                nonce_data.deadline_adjusted,
                                e.code,
                                &e.message,
                            );
                            Ok(true)
                        }
                        Err(_) => {
                            log_submission_failed(
                                retry,
                                nonce_data.numeric_id,
                                nonce_data.nonce,
                                nonce_data.deadline_adjusted,
                            );
                            Ok(false)
                        }
                    },
                )
            })
            .take_while(|success| future::ok(!success))
            .for_each(|_| Ok(()))
    }
}

fn log_deadline_mismatch(
    height: u64,
    account_id: u64,
    nonce: u64,
    deadline: u64,
    deadline_pool: u64,
) {
    error!(
        "submit: deadlines mismatch, height={}, account={}, nonce={}, \
         deadline_miner={}, deadline_pool={}",
        height, account_id, nonce, deadline, deadline_pool
    );
}

fn log_submission_failed(retry: u8, account_id: u64, nonce: u64, deadline: u64) {
    if retry < 3 {
        warn!(
            "{: <80}",
            format!(
                "submission failed:, attempt={}, account={}, nonce={}, deadline={}",
                retry, account_id, nonce, deadline
            )
        );
    } else {
        error!(
            "{: <80}",
            format!(
                "submission retries exhausted: account={}, nonce={}, deadline={}",
                account_id, nonce, deadline
            )
        );
    }
}

fn log_submission_not_accepted(
    height: u64,
    account_id: u64,
    nonce: u64,
    deadline: u64,
    err_code: i32,
    msg: &str,
) {
    error!(
        "submission not accepted: height={}, account={}, nonce={}, \
         deadline={}\n\tcode: {}\n\tmessage: {}",
        height, account_id, nonce, deadline, err_code, msg,
    );
}

fn log_submission_accepted(account_id: u64, nonce: u64, deadline: u64) {
    info!(
        "deadline accepted: account={}, nonce={}, deadline={}",
        account_id, nonce, deadline
    );
}

fn parse_json_result<T: DeserializeOwned>(body: &Chunk) -> Result<T, PoolError> {
    match serde_json::from_slice(body.bytes()) {
        Ok(x) => Ok(x),
        _ => match serde_json::from_slice::<PoolErrorWrapper>(body.bytes()) {
            Ok(x) => Err(x.error),
            _ => {
                let v = body.to_vec();
                Err(PoolError {
                    code: 0,
                    message: String::from_utf8_lossy(&v).to_string(),
                })
            }
        },
    }
}

fn do_req<T: DeserializeOwned>(
    url: Url,
    method: reqwest::Method,
    headers: reqwest::header::HeaderMap,
    timeout: Duration,
) -> impl Future<Item = T, Error = FetchError> {
    let mut req = Request::new(method, url);
    req.headers_mut().extend(headers);

    ClientBuilder::new()
        .timeout(timeout)
        .build()
        .unwrap()
        .execute(req)
        .and_then(|mut res| {
            let body = mem::replace(res.body_mut(), Decoder::empty());
            body.concat2()
        })
        .from_err::<FetchError>()
        .and_then(|body| match parse_json_result(&body) {
            Ok(x) => Ok(x),
            Err(e) => Err(e.into()),
        })
}
