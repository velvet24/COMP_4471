import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets

def train(data_dir, num_classes):

    model = models.resnet18(weights="ResNet18_Weights.DEFAULT")
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    batch_size = 32

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {
        'train': datasets.ImageFolder(f'{data_dir}/train', data_transforms['train']),
        'val': datasets.ImageFolder(f'{data_dir}/val', data_transforms['val']),
        'test': datasets.ImageFolder(f'{data_dir}/test', data_transforms['val'])
    }

    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True),
        'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=True),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False)
    }

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(20):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            print(f'{phase} Epoch {epoch+1} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
    torch.save(model.state_dict(), 'resnet18_custom.pth')

if __name__ == "__main__":
    train("./datasets/fishNet_split", 372)