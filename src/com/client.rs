use crate::com::api::*;
use futures::stream::Stream;
use futures::Future;
use reqwest::header::{HeaderMap, HeaderName};
use reqwest::r#async::{Client as InnerClient, ClientBuilder, Decoder};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::mem;
use std::sync::Arc;
use std::time::Duration;
use url::form_urlencoded::byte_serialize;
use url::Url;

/// A client for communicating with Pool/Proxy/Wallet.
#[derive(Clone, Debug)]
pub struct Client {
    inner: InnerClient,
    secret_phrase: Arc<String>,
    base_uri: Url,
    headers: Arc<HeaderMap>,
}

/// Parameters ussed for nonce submission.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubmissionParameters {
    pub account_id: u64,
    pub nonce: u64,
    pub height: u64,
    pub block: u64,
    pub deadline_unadjusted: u64,
    pub deadline: u64,
    pub gen_sig: [u8; 32],
}

/// Usefull for deciding which submission parameters are the newest and best.
/// We always cache the currently best submission parameters and on fail
/// resend them with an exponential backoff. In the meantime if we get better
/// parameters the old ones need to be replaced.
impl Ord for SubmissionParameters {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.block < other.block {
            Ordering::Less
        } else if self.block > other.block {
            Ordering::Greater
        } else if self.gen_sig == other.gen_sig {
            // on the same chain, best deadline wins
            if self.deadline <= other.deadline {
                Ordering::Greater
            } else {
                Ordering::Less
            }
        } else {
            // switched to a new chain
            Ordering::Less
        }
    }
}

impl PartialOrd for SubmissionParameters {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Whether to send additional data for Proxies.
#[derive(Clone, PartialEq, Debug)]
pub enum ProxyDetails {
    /// Send additional data like capacity, miner name, ...
    Enabled,
    /// Don't send any additional data:
    Disabled,
}

impl Client {
    fn ua() -> String {
        "Bencher/".to_owned() + crate_version!()
    }

    fn submit_nonce_headers(
        proxy_details: ProxyDetails,
        additional_headers: Arc<HashMap<String, String>>,
    ) -> HeaderMap {
        let ua = Client::ua();
        let mut headers = HeaderMap::new();
        headers.insert("User-Agent", ua.to_owned().parse().unwrap());
        if proxy_details == ProxyDetails::Enabled {
            // It's amazing how a user agent is just not enough.
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

        for (key, value) in &*additional_headers {
            let header_name = HeaderName::from_bytes(&key.clone().into_bytes()).unwrap();
            headers.insert(header_name, value.parse().unwrap());
        }

        headers
    }

    /// Create a new client communicating with Pool/Proxy/Wallet.
    pub fn new(
        base_uri: Url,
        secret_phrase: String,
        timeout: u64,
        proxy_details: ProxyDetails,
        additional_headers: Arc<HashMap<String, String>>,
    ) -> Self {
        let secret_phrase_encoded = byte_serialize(secret_phrase.as_bytes()).collect();

        let headers =
            Client::submit_nonce_headers(proxy_details, additional_headers);

        let client = ClientBuilder::new()
            .timeout(Duration::from_millis(timeout))
            .build()
            .unwrap();

        Self {
            inner: client,
            secret_phrase: Arc::new(secret_phrase_encoded),
            base_uri,
            headers: Arc::new(headers),
        }
    }

    /// Get current mining info.
    pub fn get_mining_info(&self, capacity: u64, additional_headers: Arc<HashMap<String, String>>, xpu_string : Arc<String>) -> impl Future<Item = MiningInfoResponse, Error = FetchError> {
        let mut headers = (*self.headers).clone();
        headers.insert(
            "X-Capacity",
            capacity.to_string().parse().unwrap(),
        );
        headers.insert(
            "X-Xpu",
            xpu_string.parse().unwrap(),
        );
        for (key, value) in &*additional_headers {
            let header_name = HeaderName::from_bytes(&key.clone().into_bytes()).unwrap();
            headers.insert(header_name, value.parse().unwrap());
        }
        self.inner
            .get(self.uri_for("burst"))
            .headers(headers)          
            .query(&GetMiningInfoRequest {
                request_type: &"getMiningInfo",
            })
            .send()
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

    pub fn uri_for(&self, path: &str) -> Url {
        let mut url = self.base_uri.clone();
        url.path_segments_mut()
            .map_err(|_| "cannot be base")
            .unwrap()
            .pop_if_empty()
            .push(path);
        url
    }

    /// Submit nonce to the pool and get the corresponding deadline.
    pub fn submit_nonce(
        &self,
        submission_data: &SubmissionParameters,
    ) -> impl Future<Item = SubmitNonceResponse, Error = FetchError> {

        let mut query = format!(
            "requestType=submitNonce&accountId={}&nonce={}&secretPhrase={}&blockheight={}",
            submission_data.account_id, submission_data.nonce, self.secret_phrase, submission_data.height
        );

        // If we don't have a secret phrase then we most likely talk to a pool or a proxy.
        // Both can make use of the deadline, e.g. a proxy won't validate deadlines but still
        // needs to rank the deadlines.
        // The best thing is that legacy proxies use the unadjusted deadlines so...
        // yay another parameter!
        if *self.secret_phrase == "" {
            query += &format!("&deadline={}", submission_data.deadline_unadjusted);
        }        

        // Some "Extrawurst" for the CreepMiner proxy (I think?) which needs the deadline inside
        // the "X-Deadline" header.
        let mut headers = (*self.headers).clone();
        headers.insert(
            "X-Deadline",
            submission_data.deadline.to_string().parse().unwrap(),
        );

        let mut uri = self.uri_for("burst");
        uri.set_query(Some(&query));

        self.inner
            .post(uri)
            .headers(headers)
            .send()
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    static BASE_URL: &str = "https://wallet.burstcoin.ro/";

    #[test]
    fn test_submit_params_cmp() {
        let submit_params_1 = SubmissionParameters {
            account_id: 1337,
            nonce: 12,
            height: 112,
            block: 0,
            deadline_unadjusted: 7123,
            deadline: 1193,
            gen_sig: [0; 32],
        };

        let mut submit_params_2 = submit_params_1.clone();
        submit_params_2.block += 1;
        assert!(submit_params_1 < submit_params_2);

        let mut submit_params_2 = submit_params_1.clone();
        submit_params_2.deadline -= 1;
        assert!(submit_params_1 < submit_params_2);

        let mut submit_params_2 = submit_params_1.clone();
        submit_params_2.gen_sig[0] = 1;
        submit_params_2.deadline += 1;
        assert!(submit_params_1 < submit_params_2);

        let mut submit_params_2 = submit_params_1.clone();
        submit_params_2.deadline += 1;
        assert!(submit_params_1 > submit_params_2);
    }

    #[test]
    fn test_requests() {
        let mut rt = tokio::runtime::Runtime::new().expect("can't create runtime");

        let mut secret = HashMap::new();
        secret.insert(1337u64,"secret".to_owned());
        let client = Client::new(
            BASE_URL.parse().unwrap(),
            secret,
            5000,
            12,
            ProxyDetails::Enabled,
            HashMap::new(),
        );

        let height = match rt.block_on(client.get_mining_info()) {
            Err(e) => panic!("can't get mining info: {:?}", e),
            Ok(mining_info) => mining_info.height,
        };

        // this fails if pinocchio switches to a new block height in the meantime
        let nonce_submission_response = rt.block_on(client.submit_nonce(&SubmissionParameters {
            account_id: 1337,
            nonce: 12,
            height,
            block: 1,
            deadline_unadjusted: 7123,
            deadline: 1193,
            gen_sig: [0; 32],
        }));

        if let Err(e) = nonce_submission_response {
            assert!(false, "can't submit nonce: {:?}", e);
        }
    }
}
