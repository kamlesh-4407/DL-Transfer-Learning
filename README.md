# DL- Developing a Neural Network Classification Model using Transfer Learning

## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## Problem Statement and Dataset
Problem Statement:
Develop an image classification system using transfer learning with the VGG19 convolutional neural network to automatically classify chip images into two categories: defect and notdefect. The pretrained VGG19 model is used to extract useful visual features from the images, while its final fully connected layer is modified to perform binary classification. The model is trained and evaluated using training and testing images, and its performance is measured using loss, accuracy, confusion matrix, and classification report.

Dataset:
The dataset contains chip images divided into two classes: defect and notdefect. The images are organized into separate training and testing folders. The training data is used to train the modified VGG19 model, while the testing data is used to evaluate its classification performance.


## Neural Network Model
<img width="1145" height="908" alt="image" src="https://github.com/user-attachments/assets/373e1881-1615-4bde-9761-014d2377e94e" />


## DESIGN STEPS
### STEP 1: 

Load and preprocess the image dataset and resize the images to 224 × 224 pixels.

### STEP 2: 

Load the pretrained VGG19 model with ImageNet weights.

### STEP 3: 

Freeze the pretrained layers to preserve the learned features.
### STEP 4: 
  
Replace the final fully connected layer with a new layer containing two output classes, defect and notdefect.

### STEP 5: 

Define the Cross-Entropy loss function and Adam optimizer and train the model.

### STEP 6: 

Evaluate the trained model using loss curves, confusion matrix, classification report, and new sample prediction.

## PROGRAM

### Name: Kamlesh Y

### Register Number: 212224100039
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models, datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
## Step 1: Load and Preprocess Data
# Define transformations for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images for pre-trained model input
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Standard normalization for pre-trained models
])
# Load dataset from a folder (structured as: dataset/class_name/images)
dataset_path = "C:\\Users\\varal\\Downloads\\Deep Learning\\Deep Learning\\chip_data\\dataset"
train_dataset = datasets.ImageFolder(root=f"{dataset_path}/train", transform=transform)
test_dataset = datasets.ImageFolder(root=f"{dataset_path}/test", transform=transform)
# Display some input images
def show_sample_images(dataset, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(5, 5))
    for i in range(num_images):
        image, label = dataset[i]
        image = image.permute(1, 2, 0)  # Convert tensor format (C, H, W) to (H, W, C)
        axes[i].imshow(image)
        axes[i].set_title(dataset.classes[label])
        axes[i].axis("off")
    plt.show()
# Show sample images from the training dataset
show_sample_images(train_dataset)
# Get the total number of samples in the training dataset
print(f"Total number of training samples: {len(train_dataset)}")
print(f"Total number of testing samples: {len(test_dataset)}")
# Get the shape of the first image in the dataset
first_image, label = train_dataset[0]
print(f"Shape of the first image: {first_image.shape}")
# Create DataLoader for batch processing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

num_classes = len(train_dataset.classes)
print("Number of classes:", num_classes)
# Load Pretrained Model
model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
# Freeze all layers
for param in model.parameters():
    param.requires_grad = False
# Replace final classifier layer
model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer (train only last layer)
optimizer = optim.Adam(model.classifier[6].parameters(), lr=0.001)
# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
from torchsummary import summary
# Print model summary
summary(model, input_size=(3, 224, 224))
def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Compute validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_losses.append(val_loss / len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    
    # Plot training and validation loss
   
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
## Step 4: Test the Model and Compute Confusion Matrix & Classification Report
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Print classification report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))
train_model(model, train_loader,test_loader)
test_model(model, test_loader)
def predict_image(model, image_index, dataset):
    model.eval()
    image, label = dataset[image_index]
    with torch.no_grad():
        image_tensor = image.unsqueeze(0).to(device)
        output = model(image_tensor)
        _,pred=torch.max(output,1)
        # Apply sigmoid to get probability, threshold at 0.5
        prob = torch.sigmoid(pred)
        predicted = (prob > 0.5).int().item()

    class_names = class_names = dataset.classes
    # Display the image
    image_to_display = transforms.ToPILImage()(image)
    plt.figure(figsize=(4, 4))
    plt.imshow(image_to_display)
    plt.title(f'Actual: {class_names[label]}\nPredicted: {class_names[predicted]}')
    plt.axis("off")
    plt.show()
    print(f'Actual: {class_names[label]}, Predicted: {class_names[predicted]}')
#Example Prediction
predict_image(model, image_index=25, dataset=test_dataset)












```


### OUTPUT

## Training Loss, Validation Loss Vs Iteration Plot

<img width="1118" height="602" alt="Screenshot 2026-08-20 155526" src="https://github.com/user-attachments/assets/509be02b-8656-4acc-b48d-444d452a5c39" />


## Confusion Matrix
<img width="943" height="636" alt="Screenshot 2026-08-20 155508" src="https://github.com/user-attachments/assets/811d75fe-3427-40aa-9ec2-1bdcb197eeb2" />



## Classification Report


<img width="637" height="250" alt="Screenshot 2026-08-20 155050" src="https://github.com/user-attachments/assets/dd651af9-c3b6-4949-ba71-b1e58771a857" />

### New Sample Data Prediction

<img width="772" height="627" alt="Screenshot 2026-08-20 155425" src="https://github.com/user-attachments/assets/dd559ec1-cc5b-46dd-a68a-b62c4ad4544d" />


## RESULT
Thus the python program to develop an image classification model using transfer learning with VGG19 architecture is executed successfully.
