use model::tokenize_input;
use onnxruntime::ndarray::{self, Array};
use onnxruntime::GraphOptimizationLevel;
use onnxruntime::{environment::Environment, tensor::OrtOwnedTensor};
use std::collections::HashMap;

mod model;

fn main() -> anyhow::Result<()> {
    let environment = Environment::builder().build()?;

    let mut start_session = environment
        .new_session_builder()?
        .with_optimization_level(GraphOptimizationLevel::Basic)?
        .with_number_threads(1)?
        .with_model_from_file("models/model_start.onnx")?;

    let example_input = "endbr64 push r15 push r14 push r13";

    println!("Function start prediction (Rust): {}", start_prediction);

    Ok(())
}
