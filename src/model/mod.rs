pub mod predictor;

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;

use onnxruntime::environment::Environment;
use onnxruntime::ndarray::{self, Array};
use onnxruntime::session::Session;
use onnxruntime::tensor::OrtOwnedTensor;
use onnxruntime::GraphOptimizationLevel;

pub fn load_session<'a>(environment: &'a Environment, path: String) -> anyhow::Result<Session<'a>> {
    let session = environment
        .new_session_builder()?
        .with_optimization_level(GraphOptimizationLevel::Basic)?
        .with_number_threads(1)?
        .with_model_from_file(path)?;
    Ok(session)
}

pub fn get_word_index(path: &str) -> anyhow::Result<HashMap<String, f32>> {
    let mut tokenizer_file = File::open(path)?;
    let mut tokenizer_json = String::new();
    tokenizer_file.read_to_string(&mut tokenizer_json)?;

    let tokenizer: serde_json::Value = serde_json::from_str(&tokenizer_json)?;
    let word_index_str = tokenizer["config"]["word_index"].as_str().unwrap();

    Ok(serde_json::from_str(word_index_str)?)
}

pub fn tokenize_input(tokenizer: &HashMap<String, f32>, input: &str) -> Vec<f32> {
    let words: Vec<&str> = input.split_whitespace().collect();
    let mut tokens = Vec::new();
    for word in words {
        if let Some(token) = tokenizer.get(word) {
            tokens.push(*token);
        } else {
            tokens.push(0.0);
        }
    }

    tokens
}

pub fn predict(
    session: &mut Session<'_>,
    word_index: &HashMap<String, f32>,
    input: String,
    threshold: f32,
) -> (bool, f32) {
    let mut tokenized_input: Vec<f32> = tokenize_input(word_index, &input);

    let input_shape = &session.inputs[0].dimensions;
    if let Some(expected_length) = input_shape.get(1).copied() {
        match tokenized_input
            .len()
            .cmp(&(expected_length.unwrap() as usize))
        {
            std::cmp::Ordering::Less => {
                tokenized_input.resize(expected_length.unwrap() as usize, 0.0);
            }
            std::cmp::Ordering::Greater => {
                tokenized_input.truncate(expected_length.unwrap() as usize);
            }
            _ => {}
        }
    }

    let input_array = Array::from_shape_vec((1, tokenized_input.len()), tokenized_input).unwrap();

    let start_output_vec = session.run(vec![input_array.clone()]).unwrap();
    let start_output: &OrtOwnedTensor<f32, _> = &start_output_vec[0];
    let confidence = start_output.index_axis(ndarray::Axis(0), 0)[0];

    (confidence > threshold, confidence)
}
