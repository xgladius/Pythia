use std::collections::HashMap;
use std::fs::File;
use std::io::Read;

use onnxruntime::ndarray::{self, Array};
use onnxruntime::session::Session;
use onnxruntime::tensor::OrtOwnedTensor;

pub fn predict(
    session: &mut Session<'_>,
    word_index: HashMap<String, f32>,
    input: String,
    threshold: f32,
) -> bool {
    let mut tokenized_input: Vec<f32> = tokenize_input(&word_index, &input);

    let input_shape = &session.inputs[0].dimensions;
    if let Some(expected_length) = input_shape.get(1).and_then(|dim| dim.to_owned()) {
        if tokenized_input.len() < expected_length as usize {
            tokenized_input.resize(expected_length as usize, 0.0);
        } else if tokenized_input.len() > expected_length as usize {
            tokenized_input.truncate(expected_length as usize);
        }
    }

    let input_array = Array::from_shape_vec((1, tokenized_input.len()), tokenized_input).unwrap();

    let start_output_vec = session.run(vec![input_array.clone()]).unwrap();
    let start_output: &OrtOwnedTensor<f32, _> = &start_output_vec[0];

    start_output.index_axis(ndarray::Axis(0), 0)[0] > threshold
}

pub fn get_word_index() -> anyhow::Result<HashMap<String, f32>> {
    let mut tokenizer_file = File::open("models/tokenizer.json")?;
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
            tokens.push(token.clone());
        } else {
            println!("fail {}", word);
            tokens.push(0.0);
        }
    }

    tokens
}
