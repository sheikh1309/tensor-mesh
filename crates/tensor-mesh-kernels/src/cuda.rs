#[cfg(feature = "cuda")]
use {
    cudarc::{driver::CudaDevice, nvrtc::Ptx},
    tensor_mesh_core::{DType, Tensor},
    std::sync::Arc,
};

#[cfg(feature = "cuda")]
pub struct CudaKernel {
    dev: Arc<CudaDevice>,
}

#[cfg(feature = "cuda")]
impl super::DeviceKernel for CudaKernel {
    fn dispatch(&self, _graph: &tensor_mesh_core::Graph, _node_idx: usize) -> Result<tensor_mesh_core::Tensor, super::KernelError> {
        todo!("CUDA kernel dispatch stub")
    }
}