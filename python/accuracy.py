import os
import json
import numpy as np
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re

samples_dir = "samples"
BATCH_SIZE = 1024  # Adjust based on your system's memory and performance

# Load the tokenizer from the JSON file
with open('tokenizer.json') as f:
    tokenizer_json = f.read()
    tokenizer = tokenizer_from_json(tokenizer_json)

# Load the trained models
model_start = load_model('model_start.keras')
model_end = load_model('model_end.keras')

def load_all_samples(samples_dir):
    functions = []
    negatives = []
    for filename in os.listdir(samples_dir):
        if filename.endswith(".json"):
            sample_file = os.path.join(samples_dir, filename)
            with open(sample_file, 'r') as f:
                data = json.load(f)
                functions.extend(data.get('functions', []))
                negatives.extend(data.get('negative_examples', []))
    return functions, negatives

def process_operands(operands):
    # Remove numeric values but keep the rest of the operand string
    return re.sub(r'\b\d+\b', '', operands).strip()

def prepare_sequences(instruction_list, max_len):
    processed_instructions = [
        " ".join([f"{instr['mnemonic']} {process_operands(operand)}" for instr in instr_list for operand in instr['operands'] if operand])
        for instr_list in instruction_list
    ]
    sequences = tokenizer.texts_to_sequences(processed_instructions)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return padded_sequences

def batch_predict(model, sequences):
    predictions = []
    for i in range(0, len(sequences), BATCH_SIZE):
        batch_sequences = sequences[i:i+BATCH_SIZE]
        batch_preds = model.predict(batch_sequences, verbose=0)
        predictions.extend(batch_preds.flatten())
    return np.array(predictions)

def evaluate_model():
    functions, negatives = load_all_samples(samples_dir)
    max_len = model_start.input_shape[1]  # Adapt input length from the model

    # Prepare data for function starts
    prologue_info = [func['prologue_info'] for func in functions if func.get('prologue_info')]
    prologue_sequences = prepare_sequences(prologue_info, max_len)
    prologue_labels = np.ones(len(prologue_sequences))

    # Prepare data for function ends
    epilogue_info = [func['epilogue_info'] for func in functions if func.get('epilogue_info')]
    epilogue_sequences = prepare_sequences(epilogue_info, max_len)
    epilogue_labels = np.ones(len(epilogue_sequences))

    # Prepare data for negatives (both start and end)
    negative_info = [neg['instructions'] for neg in negatives if neg.get('instructions')]
    negative_sequences = prepare_sequences(negative_info, max_len)
    negative_labels = np.zeros(len(negative_sequences))

    # Combine positive and negative samples for start predictions
    start_sequences = np.concatenate((prologue_sequences, negative_sequences), axis=0)
    start_labels = np.concatenate((prologue_labels, negative_labels), axis=0)

    # Combine positive and negative samples for end predictions
    end_sequences = np.concatenate((epilogue_sequences, negative_sequences), axis=0)
    end_labels = np.concatenate((epilogue_labels, negative_labels), axis=0)

    # Predict function starts
    start_predictions = batch_predict(model_start, start_sequences)
    start_predictions_binary = (start_predictions >= 0.5).astype(int)

    # Predict function ends
    end_predictions = batch_predict(model_end, end_sequences)
    end_predictions_binary = (end_predictions >= 0.5).astype(int)

    # Calculate metrics for function start predictions
    print("Function Start Prediction Metrics:")
    print(f"Accuracy : {accuracy_score(start_labels, start_predictions_binary):.4f}")
    print(f"Precision: {precision_score(start_labels, start_predictions_binary):.4f}")
    print(f"Recall   : {recall_score(start_labels, start_predictions_binary):.4f}")
    print(f"F1 Score : {f1_score(start_labels, start_predictions_binary):.4f}\n")

    # Calculate metrics for function end predictions
    print("Function End Prediction Metrics:")
    print(f"Accuracy : {accuracy_score(end_labels, end_predictions_binary):.4f}")
    print(f"Precision: {precision_score(end_labels, end_predictions_binary):.4f}")
    print(f"Recall   : {recall_score(end_labels, end_predictions_binary):.4f}")
    print(f"F1 Score : {f1_score(end_labels, end_predictions_binary):.4f}")

if __name__ == '__main__':
    evaluate_model()
