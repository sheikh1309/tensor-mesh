#[cfg(feature = "cpu")]
use {
    rayon::prelude::*,
    tensor_mesh_core::{DType, Tensor},
};

#[cfg(feature = "cpu")]
pub struct CpuKernel;

#[cfg(feature = "cpu")]
impl super::DeviceKernel for CpuKernel {
    fn dispatch(&self, _graph: &tensor_mesh_core::Graph, _node_idx: usize) -> Result<tensor_mesh_core::Tensor, super::KernelError> {
        todo!("CPU kernel dispatch stub")
    }
}