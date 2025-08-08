use crate::{Tensor, TensorId};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub inputs: Vec<TensorId>,
    pub outputs: Vec<TensorId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Node {
    MatMul(MatMul),
    RmsNorm(RmsNorm),
    RoPE(RoPE),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatMul {
    pub lhs: TensorId,
    pub rhs: TensorId,
    pub out: TensorId,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RmsNorm {
    pub x: TensorId,
    pub weight: TensorId,
    pub eps: f32,
    pub out: TensorId,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoPE {
    pub x: TensorId,
    pub pos: TensorId,
    pub out: TensorId,
}