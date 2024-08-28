import onnxruntime as ort
import numpy as np
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras import preprocessing

# Load the tokenizer
with open('models/tokenizer.json') as f:
    tokenizer_json = f.read()
    tokenizer = tokenizer_from_json(tokenizer_json)

# Load the ONNX model
session = ort.InferenceSession('models/model_start.onnx')

# Example input to test
example_input = "endbr64 push r15 push r14 push r13"

# Tokenize the input text
example_seq = tokenizer.texts_to_sequences([example_input])
print(f"Tokenized input (Python): {example_seq}")

# Pad the sequence to the required length
max_len = session.get_inputs()[0].shape[1]  # Get the input shape of the ONNX model
example_padded = preprocessing.sequence.pad_sequences(example_seq, maxlen=max_len, padding='post')
print(f"Padded input (Python): {example_padded}")

# Convert to numpy array
example_padded = np.array(example_padded, dtype=np.float32)
print(f"Input array (Python): {example_padded}")

# Prepare input for ONNX model
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Run the model to get the prediction
start_prediction = session.run([output_name], {input_name: example_padded})[0]

# Output the prediction
print(f"Function start prediction (ONNX): {start_prediction[0][0]}")
