
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import argparse

# Define a function to load and preprocess the data
def load_data(data_dir):
    # Define the transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    valid_dataset = datasets.ImageFolder(data_dir + '/valid', transform=valid_transforms)

    # Define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32)

    return train_loader, valid_loader

# Define a function to build the model
def build_model(arch, hidden_units):
    # Load a pre-trained model
    model = models.__dict__[arch](pretrained=True)

    # Freeze the parameters of the pre-trained model
    for param in model.parameters():
        param.requires_grad = False

    # Replace the classifier with a custom classifier
    classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )
    model.classifier = classifier

    return model

# Define a function to train the model
def train_model(model, train_loader, valid_loader, criterion, optimizer, device, epochs):
    model.to(device)

    for epoch in range(epochs):
        train_loss = 0.0
        valid_loss = 0.0
        accuracy = 0.0

        # Training loop
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        # Validation loop
        model.eval()
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                valid_loss += loss.item() * inputs.size(0)

                ps = torch.exp(outputs)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        train_loss = train_loss / len(train_loader.dataset)
        valid_loss = valid_loss / len(valid_loader.dataset)
        accuracy = accuracy / len(valid_loader)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Valid Loss: {valid_loss:.4f}")
        print(f"Valid Accuracy: {accuracy:.4f}")
        print()

# Parse the command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--flag', action='store_true', help='use flag')
parser.add_argument('data_dir', type=str,default='flowers', help='data_dir')
parser.add_argument('--arch', type=str, default='vgg16', help='model architecture')
parser.add_argument('--hidden_units', type=int, default=512, help='number of hidden units')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--gpu', action='store_true', help='use GPU for training')
args = parser.parse_args()

# Load and preprocess the data
train_loader, valid_loader = load_data(args.data_dir)

# Build the model
model = build_model(args.arch, args.hidden_units)

# Define the criterion and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

# Train the model
device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
train_model(model, train_loader, valid_loader, criterion, optimizer, device, args.epochs)

# Save the trained model checkpoint
model.class_to_idx = train_loader.dataset.class_to_idx
checkpoint = {
    'model': model,
    'state_dict': model.state_dict(),
    'class_to_idx': model.class_to_idx,
    'arch': args.arch,
    'hidden_units': args.hidden_units
}
torch.save(checkpoint, 'checkpoint.pth')
