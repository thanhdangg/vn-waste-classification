def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return all_preds, all_labels

def calculate_metrics(all_labels, all_preds):
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return acc, f1

def evaluate_models(models, data_loader, device):
    results = {}
    for model_name, model in models.items():
        print(f"Evaluating model: {model_name}")
        all_preds, all_labels = evaluate_model(model, data_loader, device)
        acc, f1 = calculate_metrics(all_labels, all_preds)
        results[model_name] = {'accuracy': acc, 'f1_score': f1}
        print(f"Model: {model_name}, Accuracy: {acc:.4f}, F1-score: {f1:.4f}")
    return results