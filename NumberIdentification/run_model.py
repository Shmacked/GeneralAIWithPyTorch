import sys
from pathlib import Path

from simple_mlp import SimpleMLP
from simple_resnet import SimpleResNet
from helpers.model import predict
from helpers.clean_up import clean_up
import torch
import torchvision.transforms as transforms


def _model_path(filename: str) -> Path:
    if getattr(sys, "frozen", False):
        beside_exe = Path(sys.executable).resolve().parent / filename
        if beside_exe.is_file():
            return beside_exe
        if hasattr(sys, "_MEIPASS"):
            return Path(sys._MEIPASS) / filename
    return Path(__file__).resolve().parent / filename


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_dict1 = torch.load(
    _model_path("model1.pth"), weights_only=True, map_location=device
)
state_dict2 = torch.load(
    _model_path("model2.pth"), weights_only=True, map_location=device
)

simple_mlp = SimpleMLP().to(device)
simple_resnet = SimpleResNet().to(device)

simple_mlp.load_state_dict(state_dict1)
simple_resnet.load_state_dict(state_dict2)


def run_model(image):
    image = image.unsqueeze(0).to(device)
    output1 = predict(simple_mlp, image, device)
    output2 = predict(simple_resnet, image, device)
    return output1, output2


if __name__ == "__main__":
    from torchvision import datasets
    from torch.utils.data import DataLoader

    test_transform = transforms.Compose(
        [
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=test_transform
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    # plot_confusion_matrix(simple_mlp, test_loader, device)
    # plot_confusion_matrix(simple_resnet, test_loader, device)
    clean_up(
        [
            simple_mlp,
            simple_resnet,
            test_loader,
            test_dataset,
            test_transform,
            state_dict1,
            state_dict2,
        ],
        globals(),
    )
