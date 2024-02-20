from torch.utils.data import DataLoader
from torchvision import datasets

    # dataloaders

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# define these now
training_dataloader = DataLoader(
    datasets.MNIST(root=DATA_DIR, download=True, train=True, transform=ts),
    batch_size=batch_size,
    shuffle=True,
    num_workers=1,
)

validation_dataloader = DataLoader(
    datasets.MNIST(root=DATA_DIR, download=True, train=False, transform=ts),
    batch_size=batch_size,
    shuffle=False,
    num_workers=1,
)