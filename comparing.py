
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from models import LNN, DNN  # Import the model classes (assuming models.py defines LNN and DNN)
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
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).squeeze().long()
            
            outputs = model(images)
            outputs = outputs.softmax(dim=-1)
            y_true = torch.cat((y_true.long(), labels), 0)
            y_score = torch.cat((y_score, outputs), 0)

    return y_true.cpu().numpy(), y_score.cpu().numpy()[:, 1]  # Return true labels and probabilities for class 1

# Function to plot and compare ROC curves
def plot_roc_curve(y_true, y_score_1, y_score_2, model1_name='Model 1', model2_name='Model 2'):
    # Compute ROC curve and ROC area for both models
    fpr_1, tpr_1, _ = roc_curve(y_true, y_score_1)
    fpr_2, tpr_2, _ = roc_curve(y_true, y_score_2)

    auc_1 = roc_auc_score(y_true, y_score_1)
    auc_2 = roc_auc_score(y_true, y_score_2)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_1, tpr_1, color='blue', label=f'{model1_name} (AUC = {auc_1:.4f})')
    plt.plot(fpr_2, tpr_2, color='green', label=f'{model2_name} (AUC = {auc_2:.4f})')

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line for random guessing
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

    print(f"{model1_name} AUC: {auc_1:.4f}")
    print(f"{model2_name} AUC: {auc_2:.4f}")

# Example usage:
# Load models and get predictions
y_true, y_score_model1 = load_model_and_predict('LNN', 'saved_models/LNN_SAVE_898590')
_, y_score_model2 = load_model_and_predict('DNN', 'saved_models/DNN_SAVE_898590')

# Compare the ROC curves and AUC for LNN and DNN
plot_roc_curve(y_true, y_score_model1, y_score_model2, model1_name='LNN', model2_name='DNN')
