use crate::DType;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub type TensorId = u64;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor {
    pub id: TensorId,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub data: Vec<u8>,
}

impl Tensor {
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn nbytes(&self) -> usize {
        self.len() * self.dtype.size_bytes()
    }

    pub fn dummy(id: TensorId, shape: Vec<usize>, dtype: DType) -> Self {
        let len = shape.iter().product::<usize>();
        let data = vec![0u8; len * dtype.size_bytes()];
        Self {
            id,
            shape,
            dtype,
            data,
        }
    }

    pub fn into_arc(self) -> Arc<[u8]> {
        self.data.into()
    }
}