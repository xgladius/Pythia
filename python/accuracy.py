import os
import json
import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

BATCH_SIZE = 4096
SAMPLES_DIR = "samples"
MODEL_START_PATH = 'models/model_start.keras'
MODEL_END_PATH = 'models/model_end.keras'
TOKENIZER_PATH = 'models/tokenizer.json'
GRAPH_DIR = "graphs"

def load_tokenizer(tokenizer_path):
    with open(tokenizer_path) as f:
        tokenizer_json = f.read()
    return tokenizer_from_json(tokenizer_json)

def load_models(model_start_path, model_end_path):
    model_start = load_model(model_start_path)
    model_end = load_model(model_end_path)
    return model_start, model_end

def load_all_samples(samples_dir):
    functions, negatives = [], []
    for filename in os.listdir(samples_dir):
        if filename.endswith(".json"):
            with open(os.path.join(samples_dir, filename), 'r') as f:
                data = json.load(f)
                functions.extend(data.get('functions', []))
                negatives.extend(data.get('negative_examples', []))
    return functions, negatives

def process_operands(operands):
    return re.sub(r'\b\d+\b', '', operands).strip()

def prepare_sequences(instruction_list, tokenizer, max_len):
    processed_instructions = [
        " ".join([f"{instr['mnemonic']} {process_operands(operand)}" 
                  for instr in instr_list for operand in instr['operands'] if operand])
        for instr_list in instruction_list
    ]
    sequences = tokenizer.texts_to_sequences(processed_instructions)
    return pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

def batch_predict(model, sequences, batch_size):
    predictions = []
    for i in range(0, len(sequences), batch_size):
        batch_sequences = sequences[i:i+batch_size]
        batch_preds = model.predict(batch_sequences, verbose=0)
        predictions.extend(batch_preds.flatten())
    return np.array(predictions)

def plot_confusion_matrix(cm, title, filename):
    plt.figure(figsize=(10, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(os.path.join(GRAPH_DIR, filename))
    plt.close()

# Plot ROC curve
def plot_roc_curve(labels, predictions, title, filename):
    fpr, tpr, _ = roc_curve(labels, predictions)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(GRAPH_DIR, filename))
    plt.close()

def plot_metrics_bar_chart(metrics, title, filename):
    ax = metrics.set_index('Metric').plot(kind='bar', figsize=(10, 6))
    plt.title(title)
    plt.ylabel('Score')
    plt.xticks(rotation=0)
    
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.4f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points')
    plt.savefig(os.path.join(GRAPH_DIR, filename))
    plt.close()

def evaluate_model():
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    model_start, model_end = load_models(MODEL_START_PATH, MODEL_END_PATH)
    functions, negatives = load_all_samples(SAMPLES_DIR)
    max_len = model_start.input_shape[1]

    # Prepare sequences and labels
    prologue_sequences = prepare_sequences(
        [func['prologue_info'] for func in functions if func.get('prologue_info')], tokenizer, max_len)
    epilogue_sequences = prepare_sequences(
        [func['epilogue_info'] for func in functions if func.get('epilogue_info')], tokenizer, max_len)
    negative_sequences = prepare_sequences(
        [neg['instructions'] for neg in negatives if neg.get('instructions')], tokenizer, max_len)

    start_labels = np.concatenate((np.ones(len(prologue_sequences)), np.zeros(len(negative_sequences))))
    end_labels = np.concatenate((np.ones(len(epilogue_sequences)), np.zeros(len(negative_sequences))))

    start_sequences = np.concatenate((prologue_sequences, negative_sequences))
    end_sequences = np.concatenate((epilogue_sequences, negative_sequences))

    start_predictions = batch_predict(model_start, start_sequences, BATCH_SIZE)
    end_predictions = batch_predict(model_end, end_sequences, BATCH_SIZE)

    start_predictions_binary = (start_predictions >= 0.5).astype(int)
    end_predictions_binary = (end_predictions >= 0.5).astype(int)

    metrics_data = calculate_and_print_metrics("Function Start", start_labels, start_predictions_binary)
    metrics_data.update(calculate_and_print_metrics("Function End", end_labels, end_predictions_binary))

    plot_confusion_matrix(confusion_matrix(start_labels, start_predictions_binary), 
                          "Confusion Matrix for Function Start Predictions", 
                          "confusion_matrix_start.png")
    
    plot_confusion_matrix(confusion_matrix(end_labels, end_predictions_binary), 
                          "Confusion Matrix for Function End Predictions", 
                          "confusion_matrix_end.png")

    metrics_df = pd.DataFrame(metrics_data)
    plot_metrics_bar_chart(metrics_df, f'Model Performance Metrics\n(Tested on {len(functions)} Functions and {len(negatives)} Negatives)',
                           "model_performance_metrics.png")

    plot_roc_curve(start_labels, start_predictions, "ROC Curve for Function Start Predictions", "roc_curve_start.png")
    plot_roc_curve(end_labels, end_predictions, "ROC Curve for Function End Predictions", "roc_curve_end.png")

def calculate_and_print_metrics(prefix, labels, predictions):
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    roc_auc_value = auc(*roc_curve(labels, predictions)[:2])

    print(f"\n{prefix} Prediction Metrics:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc_value:.4f}")

    return {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'],
        f'{prefix}': [accuracy, precision, recall, f1, roc_auc_value]
    }

if __name__ == '__main__':
    if not os.path.exists(GRAPH_DIR):
        os.makedirs(GRAPH_DIR)
    evaluate_model()
