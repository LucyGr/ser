from pathlib import Path
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from ser.data import train_dataloader, validate_dataloader
from ser.train import train as trainer


import typer

main = typer.Typer()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


@main.command()
def train(
    name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to save under."
    ),
    epochs: int = typer.Option(
        2, "-e", "--epoch", help="Number of epochs."
    ),
    batch_size: int = typer.Option(
        1000, "-bs", "--batch_size", help="Batch size."
    ),
    learning_rate: float = typer.Option(
        0.01, "-lr", "--learning_rate", help="Learning rate."
    ),
):
    print(f"Running experiment {name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # save the parameters!
    ####################### need to do this #############
    params = (name, epochs, batch_size, learning_rate)

    # torch transforms
    ts = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

# data loaders were here now in data.py
    training_dataloader = train_dataloader(batch_size, ts)
    validation_dataloader = validate_dataloader(batch_size, ts)
    
    save_here = PROJECT_ROOT / "saved stuff"
    # train
    trainer(save_here, params, training_dataloader, validation_dataloader,device )



@main.command()
def infer():
    print("This is where the inference code will go")
