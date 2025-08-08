#[cfg(feature = "metal")]
use {
    metal_rs::{Device, CommandQueue},
    tensor_mesh_core::{DType, Tensor},
};

#[cfg(feature = "metal")]
pub struct MetalKernel {
    device: Device,
    queue:  CommandQueue,
}

#[cfg(feature = "metal")]
impl super::DeviceKernel for MetalKernel {
    fn dispatch(&self, _graph: &tensor_mesh_core::Graph, _node_idx: usize) -> Result<tensor_mesh_core::Tensor, super::KernelError> {
        todo!("Metal kernel dispatch stub")
    }
}