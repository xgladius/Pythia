use onnxruntime::ndarray::{self, Array};
use onnxruntime::GraphOptimizationLevel;
use onnxruntime::{environment::Environment, tensor::OrtOwnedTensor};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;

fn main() -> anyhow::Result<()> {
    let mut tokenizer_file = File::open("models/tokenizer.json")?;
    let mut tokenizer_json = String::new();
    tokenizer_file.read_to_string(&mut tokenizer_json)?;

    let tokenizer: serde_json::Value = serde_json::from_str(&tokenizer_json)?;
    let word_index_str = tokenizer["config"]["word_index"].as_str().unwrap();
    let word_index: HashMap<String, f32> = serde_json::from_str(word_index_str)?;

    let environment = Environment::builder().build()?;

    let mut start_session = environment
        .new_session_builder()?
        .with_optimization_level(GraphOptimizationLevel::Basic)?
        .with_number_threads(1)?
        .with_model_from_file("models/model_start.onnx")?;

    let example_input = "endbr64 push r15 push r14 push r13";
    let mut tokenized_input: Vec<f32> = tokenize_input(&word_index, example_input);

    let input_shape = &start_session.inputs[0].dimensions;
    if let Some(expected_length) = input_shape.get(1).and_then(|dim| dim.to_owned()) {
        if tokenized_input.len() < expected_length as usize {
            tokenized_input.resize(expected_length as usize, 0.0); // Pad with zeros if necessary
        } else if tokenized_input.len() > expected_length as usize {
            tokenized_input.truncate(expected_length as usize); // Trim the input if necessary
        }
    }

    let input_array = Array::from_shape_vec((1, tokenized_input.len()), tokenized_input)?;

    let start_output_vec = start_session.run(vec![input_array.clone()])?;
    let start_output: &OrtOwnedTensor<f32, _> = &start_output_vec[0];

    let start_prediction = start_output.index_axis(ndarray::Axis(0), 0)[0];

    println!("Function start prediction (Rust): {}", start_prediction);

    Ok(())
}

fn tokenize_input(tokenizer: &HashMap<String, f32>, input: &str) -> Vec<f32> {
    let words: Vec<&str> = input.split_whitespace().collect();
    let mut tokens = Vec::new();
    for word in words {
        if let Some(token) = tokenizer.get(word) {
            tokens.push(token.clone());
        } else {
            println!("fail {}", word);
            tokens.push(0.0);
        }
    }

    tokens
}
