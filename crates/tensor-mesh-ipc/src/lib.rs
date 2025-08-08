use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedTensor {
    pub id: u64,
    pub offset: u64,
    pub len: u64,
}

#[derive(Debug)]
pub struct LeaseToken {
    id: u64,
}

impl Drop for LeaseToken {
    fn drop(&mut self) {
        // TODO: send release RPC to broker
    }
}

pub async fn connect_broker(_path: &str) -> std::io::Result<()> {
    Ok(())
}