import itertools
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import logging
import os


# Setup logger
def setup_logger(log_file):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.INFO)

    # Create file handler for logging to a file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Create console handler for logging to stdout
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a formatter and attach it to both handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def train_and_evaluate_model(hyperparameter_1_batch_size=8,
                             hyperparameter_2_learning_rate=0.01,
                             hyperparameter_3_num_epochs=100,
                             dataset_choice=6,
                             image_resolution=224,
                             output_dir="./output/"):
    """
    Train and evaluate a MobileNetV2 model on a selected dataset.
    """

    # Setup the logger
    log_file = os.path.join(output_dir, 'training.log')
    logger = setup_logger(log_file)

    # Step 1: Set up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Step 2: Define data preprocessing
    transform = transforms.Compose([
        transforms.Resize((image_resolution, image_resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    input_dir = ""
    if dataset_choice == 1:
        input_dir = "../tiny_datasets/BeeOrWasp/"
    elif dataset_choice == 2:
        input_dir = "../tiny_datasets/InsectRecognition/"
    elif dataset_choice == 3:
        input_dir = "../tiny_datasets/ImagesCV/"
    elif dataset_choice == 4:
        input_dir = "../organized_datasets/BeeOrWasp/"
    elif dataset_choice == 5:
        input_dir = "../organized_datasets/InsectRecognition/"
    elif dataset_choice == 6:
        input_dir = "../organized_datasets/ImagesCV/"

    #output_dir = os.path.join(output_dir, str(dataset_choice))

    # Step 3: Load the data
    logger.info(f"Loading data from {input_dir}")
    train_data = datasets.ImageFolder(root=os.path.join(input_dir, 'training_data'), transform=transform)
    val_data = datasets.ImageFolder(root=os.path.join(input_dir, 'validation_data'), transform=transform)
    test_data = datasets.ImageFolder(root=os.path.join(input_dir, 'testing_data'), transform=transform)

    train_loader = DataLoader(train_data, batch_size=hyperparameter_1_batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=hyperparameter_1_batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=hyperparameter_1_batch_size, shuffle=False)

    # Step 4: Initialize the model from scratch
    model = models.resnet18(weights=None)
    num_classes = len(train_data.classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # Step 5: Define loss and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameter_2_learning_rate)

    # Step 6: Train and validate, with tracking for plotting
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    def train(model, loader, loss_function, optimizer):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(loader.dataset)
        accuracy = correct / total
        return epoch_loss, accuracy

    def evaluate(model, loader, loss_function):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(loader.dataset)
        accuracy = correct / total
        return epoch_loss, accuracy

    # Training loop with tracking
    for epoch in range(hyperparameter_3_num_epochs):
        train_loss, train_acc = train(model, train_loader, loss_function, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, loss_function)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        logger.info(f"Epoch {epoch + 1}/{hyperparameter_3_num_epochs}, "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Step 7: Evaluate on test data
    test_loss, test_acc = evaluate(model, test_loader, loss_function)
    logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # Step 8: Plot training and validation accuracy/loss
    epochs = range(1, hyperparameter_3_num_epochs + 1)

    # Plot Loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'g', label='Training Loss')
    plt.plot(epochs, val_losses, 'b', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'g', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'b', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig(os.path.join(output_dir, "accuracyAndLoss.png"))

    # Save the model
    torch.save(model.state_dict(), os.path.join(output_dir, "mobilenet_v2_model.pth"))
    logger.info("Model saved as mobilenet_v2_model.pth")

    # Step 9: Generate t-SNE Plot
    def plot_tsne(model, loader, device):
        model.eval()
        features, labels = [], []
        with torch.no_grad():
            for inputs, lbls in loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                features.append(outputs.cpu().numpy())
                labels.extend(lbls.numpy())

        features = np.concatenate(features, axis=0)
        labels = np.array(labels)

        # Standardize features before t-SNE
        features = StandardScaler().fit_transform(features)

        # Adjust perplexity based on number of samples
        n_samples = features.shape[0]
        perplexity = min(30, n_samples - 1)  # Ensures perplexity is less than n_samples

        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        tsne_results = tsne.fit_transform(features)

        # Plot t-SNE results
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter)
        plt.title("t-SNE visualization of Model Outputs")
        plt.savefig(os.path.join(output_dir, "tsne_plot.png"))

    # Step 10: Confusion Matrix and Classification Report
    def evaluate_model_performance(model, data_loader, device):
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = outputs.max(1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=train_data.classes,
                    yticklabels=train_data.classes)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))

        # Classification Report
        report = classification_report(y_true, y_pred, target_names=train_data.classes, output_dict=True)

        # Saving classification metrics as an image
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.table(cellText=[list(report[class_name].values()) for class_name in train_data.classes],
                 colLabels=["Precision", "Recall", "F1-score", "Support"],
                 rowLabels=train_data.classes,
                 loc="center")
        plt.savefig(os.path.join(output_dir, "classification_report.png"))

    # Execute Reporting
    plot_tsne(model, test_loader, device)
    evaluate_model_performance(model, test_loader, device)

    logger.info("t-SNE plot, confusion matrix, and classification report have been saved as PNG files.")


def run_experiments():
    """
    Run experiments with different combinations of batch size, learning rate, and dataset choice.
    Organize the outputs and logs into separate directories.
    """

    # Hyperparameters and dataset choices
    batch_sizes = [8, 16, 32, 64]
    learning_rates = [0.01, 0.001, 0.0001, 0.00001]
    dataset_choices = [1, 2, 3, 4, 5, 6]

    # Output base directory
    base_output_dir = "./pretrained_experiments_results/"

    # Create the base output directory if it does not exist
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)

    # Set up the log file for the current experiment
    log_file = os.path.join(base_output_dir, 'overview_training.log')
    logger = setup_logger(log_file)

    # Iterate over all combinations of hyperparameters and dataset choices
    for batch_size, learning_rate, dataset_choice in itertools.product(batch_sizes, learning_rates, dataset_choices):

        # Create a directory for the current combination of parameters
        output_dir = os.path.join(base_output_dir, f"batch_{batch_size}_lr_{learning_rate}_dataset_{dataset_choice}")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        logger.info(
            f"Running experiment with batch_size={batch_size}, learning_rate={learning_rate}, dataset_choice={dataset_choice}")

        # Call the train_and_evaluate_model function with the current parameters
        train_and_evaluate_model(
            hyperparameter_1_batch_size=batch_size,
            hyperparameter_2_learning_rate=learning_rate,
            hyperparameter_3_num_epochs=100,  # Set the number of epochs as required
            dataset_choice=dataset_choice,
            image_resolution=224,  # Set image resolution as required
            output_dir=output_dir  # Save outputs and logs in the appropriate folder
        )
        logger.info(
            f"Experiment completed for batch_size={batch_size}, learning_rate={learning_rate}, dataset_choice={dataset_choice}")

run_experiments()