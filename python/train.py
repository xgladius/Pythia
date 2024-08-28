import json
import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, preprocessing
from sklearn.metrics import accuracy_score
import tf2onnx

# Directory containing JSON files
directory = 'samples'

# Step 1: Load the JSON data from all files in the directory
all_data = []
for filename in os.listdir(directory):
    if filename.endswith('.json'):
        with open(os.path.join(directory, filename), 'r') as f:
            data = json.load(f)
            all_data.append(data)

# Step 2: Prepare instruction information and labels for training
X_start = []
X_end = []
y_start = []
y_end = []

# Flatten the data and collect instruction information
for file_data in all_data:
    # Process function data
    for entry in file_data.get("functions", []):
        prologue_info = entry["prologue_info"]
        epilogue_info = entry["epilogue_info"]

        if prologue_info == epilogue_info or any(insn["mnemonic"] == "endbr64" for insn in epilogue_info):
            continue

        X_start.append(" ".join([f"{insn['mnemonic']} {','.join(insn['operands'])}" for insn in prologue_info]))
        X_end.append(" ".join([f"{insn['mnemonic']} {','.join(insn['operands'])}" for insn in epilogue_info]))
        y_start.append(1)  # Positive example for function start
        y_end.append(1)    # Positive example for function end

    # Process negative examples
    for neg_example in file_data.get("negative_examples", []):
        instructions = neg_example["instructions"]
        instruction_str = " ".join([f"{insn['mnemonic']} {','.join(insn['operands'])}" for insn in instructions])
        X_start.append(instruction_str)
        X_end.append(instruction_str)
        y_start.append(neg_example["label_start"])  # Negative example for function start
        y_end.append(neg_example["label_end"])      # Negative example for function end

# Step 3: Tokenize the instruction information
tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(X_start + X_end)

X_start_seq = tokenizer.texts_to_sequences(X_start)
X_end_seq = tokenizer.texts_to_sequences(X_end)

# Pad sequences to ensure consistent input shape
max_len = max(max(len(seq) for seq in X_start_seq), max(len(seq) for seq in X_end_seq))
X_start_seq = preprocessing.sequence.pad_sequences(X_start_seq, maxlen=max_len, padding='post')
X_end_seq = preprocessing.sequence.pad_sequences(X_end_seq, maxlen=max_len, padding='post')

X_start_seq = np.array(X_start_seq)
X_end_seq = np.array(X_end_seq)
y_start = np.array(y_start)
y_end = np.array(y_end)

# Step 4: Split the data into training and test sets
X_train_start, X_test_start, y_train_start, y_test_start = train_test_split(X_start_seq, y_start, test_size=0.1, random_state=42)
X_train_end, X_test_end, y_train_end, y_test_end = train_test_split(X_end_seq, y_end, test_size=0.1, random_state=42)


# Step 5: Build an RNN model
def create_rnn_model(input_shape, vocab_size):
    model = models.Sequential()
    model.add(layers.Embedding(input_dim=vocab_size, output_dim=64, input_length=input_shape[1]))
    model.add(layers.LSTM(64, return_sequences=False))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create models for function start and end
vocab_size = len(tokenizer.word_index) + 1  # +1 because tokenizer indexes start at 1
model_start = create_rnn_model(X_train_start.shape, vocab_size)
model_end = create_rnn_model(X_train_end.shape, vocab_size)

# Step 6: Train the models
model_start.fit(X_train_start, y_train_start, epochs=10, batch_size=32, validation_split=0.1)
model_end.fit(X_train_end, y_train_end, epochs=10, batch_size=32, validation_split=0.1)

# Step 7: Evaluate the models
loss_start, accuracy_start = model_start.evaluate(X_test_start, y_test_start)
loss_end, accuracy_end = model_end.evaluate(X_test_end, y_test_end)

print(f"RNN model accuracy for function start: {accuracy_start:.2f}")
print(f"RNN model accuracy for function end: {accuracy_end:.2f}")

# Step 8: Save the trained models
model_start.save('models/model_start.keras')
model_end.save('models/model_end.keras')

print("RNN models saved to 'model_start.keras' and 'model_end.keras'")

tokenizer_json = tokenizer.to_json()
with open('models/tokenizer.json', 'w') as f:
    f.write(tokenizer_json)
print("Tokenizer saved to 'tokenizer.json'")
