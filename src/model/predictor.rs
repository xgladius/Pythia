use std::collections::HashMap;

use iced_x86::{Formatter, Instruction, IntelFormatter};
use onnxruntime::{environment::Environment, session::Session};

use crate::disasm::MyFormatterOutput;

use super::{get_word_index, load_session, predict};

pub struct Predictor<'a> {
    prologue_session: Session<'a>,
    epilogue_session: Session<'a>,
    word_index: HashMap<String, f32>,
}

impl<'a> Predictor<'a> {
    pub fn new(
        environment: &'a Environment,
        prologue_path: String,
        epilogue_path: String,
        word_index_path: String,
    ) -> anyhow::Result<Self> {
        let prologue_session = load_session(environment, prologue_path)?;
        let epilogue_session = load_session(environment, epilogue_path)?;
        let word_index = get_word_index(&word_index_path)?;

        Ok(Predictor {
            prologue_session,
            epilogue_session,
            word_index,
        })
    }

    pub fn is_function_start(
        &mut self,
        instructions: &[Instruction],
        formatter: &mut IntelFormatter,
        confidence: f32,
    ) -> (bool, f32) {
        Self::do_prediction(
            &mut self.prologue_session,
            &self.word_index,
            instructions,
            formatter,
            confidence,
        )
    }

    pub fn is_function_end(
        &mut self,
        instructions: &[Instruction],
        formatter: &mut IntelFormatter,
        confidence: f32,
    ) -> (bool, f32) {
        Self::do_prediction(
            &mut self.epilogue_session,
            &self.word_index,
            instructions,
            formatter,
            confidence,
        )
    }

    fn do_prediction(
        session: &mut Session<'_>,
        word_index: &HashMap<String, f32>,
        instructions: &[Instruction],
        formatter: &mut IntelFormatter,
        confidence: f32,
    ) -> (bool, f32) {
        let mut tokenized_input = String::new();
        for instr in instructions.iter() {
            let mut output = String::new();
            formatter.format(instr, &mut MyFormatterOutput::new(&mut output));
            tokenized_input.push_str(&output);
            tokenized_input.push(' ');
        }

        predict(session, word_index, tokenized_input, confidence)
    }
}
