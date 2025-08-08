use serde::{Deserialize, Serialize};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    F16,
    F32,
    I8,
}

impl DType {
    pub fn size_bytes(&self) -> usize {
        match self {
            DType::F16 => 2,
            DType::F32 => 4,
            DType::I8  => 1,
        }
    }
}