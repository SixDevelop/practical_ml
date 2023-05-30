import hydra
import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
from src.dataset import FashionMNISTDataset
from torch.utils.tensorboard import SummaryWriter

from src.checkpoint import CheckpointSaver
from src.model import NeuralNetwork
from train import test, train


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    train_data = FashionMNISTDataset(
        "data/FashionMNIST/raw", train=True)  # ваш код
    test_data = FashionMNISTDataset(
        "data/FashionMNIST/raw", train=False)  # ваш код
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NeuralNetwork().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.settings.learning_rate)
    writer = SummaryWriter()
    checkpoint_saver = CheckpointSaver(model.__class__.__name__, should_minimize=False)
    step = 0
    for t in range(cfg.settings.epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        print("Train => ")
        train(train_dataloader, model, loss_fn, optimizer, writer)
        print("Test => ")
        step = test(test_dataloader, model, loss_fn, checkpoint_saver, writer, step)


if __name__ == '__main__':
    main()