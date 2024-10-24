import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import matplotlib.pyplot as plt
from models import LNN, DNN  # Assuming models.py defines LNN and DNN
from train import data_file, transform, BATCH_SIZE  # Assuming train.py defines data_file and transform for loading data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the test dataset
test_set = data_file(split='test', transform=transform, download=False)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=False)

# Function to load model and generate predictions
def load_model_and_predict(model_arch, model_path):
    if model_arch == 'LNN':
        model = LNN(16, 19, 2, 32).to(device)
    elif model_arch == 'DNN':
        model = DNN(28 * 28).to(device)
    else:
        raise ValueError("Unknown model architecture")
    
    # Load saved model weights
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Generate predictions
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)
    y_output_labels = torch.tensor([]).to(device)
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).squeeze().long()
            
            outputs = model(images)
            outputs = outputs.softmax(dim=-1)
            y_true = torch.cat((y_true.long(), labels), 0)
            y_score = torch.cat((y_score, outputs), 0)
            y_output_labels = torch.cat((y_output_labels, torch.argmax(outputs, dim=1)), 0)
    
    return y_true.cpu().numpy(), y_score.cpu().numpy(), y_output_labels.cpu().numpy()

# Calculate metrics for each model
def calculate_metrics(y_true, y_score, y_output_labels):
    auc = roc_auc_score(y_true, y_score[:, 1])
    accuracy = accuracy_score(y_true, y_output_labels)
    fscore = f1_score(y_true, y_output_labels)
    return auc, accuracy, fscore

# Load models and get predictions
y_true, y_score_model1, y_output_labels_model1 = load_model_and_predict('LNN', 'saved_models/LNN_SAVE_898590')
_, y_score_model2, y_output_labels_model2 = load_model_and_predict('DNN', 'saved_models/DNN_SAVE_898590')

# Calculate metrics for LNN
auc_model1, acc_model1, f1_model1 = calculate_metrics(y_true, y_score_model1, y_output_labels_model1)

# Calculate metrics for DNN
auc_model2, acc_model2, f1_model2 = calculate_metrics(y_true, y_score_model2, y_output_labels_model2)

# Create a bar chart to compare AUC, accuracy, and F1-Score for both models
metrics = ['AUC', 'Accuracy', 'F1-Score']
lnn_scores = [auc_model1, acc_model1, f1_model1]
dnn_scores = [auc_model2, acc_model2, f1_model2]

# Plot bar chart
bar_width = 0.35
index = range(len(metrics))

fig, ax = plt.subplots(figsize=(8, 6))
bar1 = plt.bar(index, lnn_scores, bar_width, label='LNN', color='blue')
bar2 = plt.bar([i + bar_width for i in index], dnn_scores, bar_width, label='DNN', color='green')

# Labeling
plt.xlabel('Metrics')
plt.ylabel('Scores')
plt.title('Comparison of AUC, Accuracy, and F1-Score for LNN and DNN')
plt.xticks([i + bar_width / 2 for i in index], metrics)
plt.legend()

# Display plot
plt.tight_layout()
plt.show()

# Print out the metrics for reference
print(f'LNN: AUC={auc_model1:.4f}, Accuracy={acc_model1:.4f}, F1-Score={f1_model1:.4f}')
print(f'DNN: AUC={auc_model2:.4f}, Accuracy={acc_model2:.4f}, F1-Score={f1_model2:.4f}')
