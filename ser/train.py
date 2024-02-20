from torch import optim
import torch
import torch.nn.functional as F

from ser.model import Net

def train(run_path, params, train_dataloader, val_dataloader, device):
    # load model
    model = Net().to(device)

    # setup params
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

    # train
    for epoch in range(params['epochs']):
        batch_train(model, train_dataloader, optimizer, epoch, device)
        batch_val(model, val_dataloader, epoch, device)

    # save model and save model params
    torch.save(model, run_path / "model.pt")




def batch_train(model, dataloader, optimizer, epoch, device):
    for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            model.train()
            optimizer.zero_grad()
            output = model(images)
            loss = F.nll_loss(output, labels)
            loss.backward()
            optimizer.step()
            print(
                f"Train Epoch: {epoch} | Batch: {i}/{len(dataloader)} "
                f"| Loss: {loss.item():.4f}"
            )




@torch.no_grad()
def batch_val(model, dataloader, epoch, device):
    # validate
    val_loss = 0
    correct = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        model.eval()
        output = model(images)
        val_loss += F.nll_loss(output, labels, reduction="sum").item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()
        val_loss /= len(dataloader.dataset)
        val_acc = correct / len(dataloader.dataset)

        print(
            f"Val Epoch: {epoch} | Avg Loss: {val_loss:.4f} | Accuracy: {val_acc}"
        )
