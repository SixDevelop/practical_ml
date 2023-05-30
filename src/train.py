import torch
from torch import nn
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.checkpoint import CheckpointSaver
from src.utils import Precision, Recall
from src.model import NeuralNetwork


def train(
    dataloader:DataLoader,
    model: NeuralNetwork,
    loss_fn: CrossEntropyLoss,
    optimizer: SGD,
    writer: SummaryWriter
):
    size = len(dataloader.dataset)
    model.train()
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs, targets = inputs.to(device), targets.to(device)

        # Compute prediction error
        preds = model(inputs)
        loss = loss_fn(preds, targets)
        writer.add_scalar("Loss/train", loss)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            loss, current = loss.item(), (batch_idx + 1) * len(inputs)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(
    dataloader: DataLoader,
    model: nn.Module, 
    loss_fn: nn.Module,
    checkpoint_saver: CheckpointSaver, 
    writer: SummaryWriter,
    step: int = 0
) -> int:
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, num_correct = 0, 0
    precision = Precision(num_classes=10)
    recall = Recall(num_classes=10)
    with torch.no_grad():
        for inputs, targets in dataloader:
            step += 1
            preds = model(inputs)
            loss = loss_fn(preds, targets).item()
            writer.add_scalar("Loss/test", loss)
            num_correct += (preds.argmax(1) == targets).type(torch.float).sum().item()
            precision(preds, targets)
            recall(preds, targets)
            
    test_loss /= num_batches
    num_correct /= size
    print(f"Test Error: \n Accuracy: {(100*num_correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(f"Test Error: \n Precision: {(100*precision.compute()):>0.1f}%, Recall: {(100*recall.compute()):>0.1f} \n")
    writer.add_scalar("Loss/precision", precision.compute())
    writer.add_scalar("Loss/recall", recall.compute())
    checkpoint_saver.get_checkpoint(model, precision.compute(), step )
    return step