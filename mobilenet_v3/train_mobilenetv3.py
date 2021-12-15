
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='path dataset')
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--project', default='', help='path save model')
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--img-size', type=int, default=224)
args = vars(parser.parse_args())

input_path = args['dataset']
batch_size = args['batch_size']
n_worker = 16
input_size = 112

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
image_datasets = {
    'train':
    datasets.ImageFolder(input_path + 'train',
    transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])),
    'validation':
    datasets.ImageFolder(input_path + 'val',
    transforms.Compose([
        transforms.Resize(int(input_size/0.875)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize,
    ]))}

train_loader = torch.utils.data.DataLoader(
    image_datasets['train'],
    batch_size=batch_size, shuffle=True,
    num_workers=n_worker, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    image_datasets['validation'],
    batch_size=batch_size, shuffle=False,
    num_workers=n_worker, pin_memory=True)

dataloaders = {
    'train':
    train_loader,
    'validation':
    val_loader
}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v3_small(pretrained=True).to(device)


criterion = nn.CrossEntropyLoss()
model.classifier = nn.Sequential(
    nn.Linear(576, 128),
    nn.Hardswish(inplace=True),
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(128, 3)).to(device)
optimizer = optim.SGD(model.classifier.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-5)
def train_model(model, criterion, optimizer, num_epochs=3):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                #print(preds)
            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                        epoch_loss,
                                                       epoch_acc))
        torch.save(model.state_dict(), args['project'] + 'classifier.h5')
    return model
model_trained = train_model(model, criterion, optimizer, num_epochs=args['epochs'])
torch.save(model_trained.state_dict(), args['project'] + 'classifier.h5')
print('save model at ', args['project'])