use tensor_mesh_core::{DType, Graph, Tensor};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum KernelError {
    #[error("unsupported dtype {0:?}")]
    UnsupportedDType(DType),
    #[error("kernel launch failed: {0}")]
    Launch(String),
}

pub trait DeviceKernel: Send + Sync {
    fn dispatch(&self, graph: &Graph, node_idx: usize) -> Result<Tensor, KernelError>;
}

/// Re-export backend modules.
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "metal")]
pub mod metal;
#[cfg(feature = "cpu")]
pub mod cpu;
