import os
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from dataset import WasteDataset
from config import Config

def get_model(name, num_classes):
    if name == 'resnet18':
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == 'mobilenetv2':
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == 'densenet121':
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == 'vgg16':
        model = models.vgg16_bn(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    else:
        raise ValueError("Model not supported")
    return model

def train_and_eval(model, train_loader, val_loader, device, epochs=50):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}")

        # Evaluate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        val_accuracies.append(acc)
        print(f"Validation Accuracy: {acc:.4f}")

    return train_losses, val_accuracies

def save_model(model, model_name):
    model_path = os.path.join(Config.model_save_path, f"{model_name}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model {model_name} to {model_path}")

def plot_training_process(train_losses, val_accuracies, model_name):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.title(f'Training Loss for {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='orange')
    plt.title(f'Validation Accuracy for {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.plot_save_path, f"{model_name}_training_process.png"))
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_names = ['resnet18', 'resnet50', 'mobilenetv2', 'densenet121', 'efficientnet_b0', 'vgg16']
    num_classes = 24  # Adjust based on your dataset
    results = {}

    for name in model_names:
        print(f"\n===== Training model: {name} =====")
        model = get_model(name, num_classes)
        
        # Assuming train_loader and val_loader are defined elsewhere
        train_losses, val_accuracies = train_and_eval(model, train_loader, val_loader, device)
        
        save_model(model, name)
        plot_training_process(train_losses, val_accuracies, name)
        
        results[name] = {
            'train_losses': train_losses,
            'val_accuracies': val_accuracies
        }

    with open('training_results.json', 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()