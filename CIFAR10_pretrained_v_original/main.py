import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader

from torchvision import datasets, transforms
from transformers import CLIPProcessor, CLIPModel

from PIL import Image
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

root = Path(__file__).parent / 'datasets' / 'cifar10'
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.CIFAR10(root=str(root), train=False, download=True, transform=transform)
testing_dl = DataLoader(test_dataset)

subset_indices = list(range(10))
test_subset = Subset(test_dataset, subset_indices)
testing_dl = DataLoader(test_dataset, shuffle=False, batch_size=8)

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

class CIFARAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))


cifar_autoencoder = CIFARAutoencoder().to(device)

p = Path(__file__).parent/ "models" / "pretrained" / "cifar_autoencoder.pt"
state = torch.load(str(p), weights_only=True, map_location=device)
cifar_autoencoder.load_state_dict(state)

cifar_autoencoder.eval()

batched_recons = []
true_labels = []
for index, (image, label) in enumerate(testing_dl):
    image = image.to(device)
    output = cifar_autoencoder(image)
    batched_recons.append(output.cpu())
    true_labels.extend(label.tolist())

recon_tensor = torch.cat(batched_recons, dim=0)

text_descriptions = [f"a photo of a {name}" for name in cifar10_classes]

def zero_shot_classification(image_tensor, text_descriptions, clip_model, clip_processor):
    image_tensor = transforms.ToPILImage()(image_tensor)
    _device = next(clip_model.parameters()).device
    inputs = clip_processor(text=text_descriptions, images=image_tensor, return_tensors="pt", padding=True)
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    outputs = clip_model(**inputs)
    scores = outputs.logits_per_image[0]
    probs = scores.softmax(-1)
    return scores, probs


recon_pred = []
for image in recon_tensor:
    scores, probs = zero_shot_classification(image, text_descriptions, clip_model, clip_processor)
    pred = probs.argmax().item()
    recon_pred.append(pred)


origin_pred = []
for index, (images, label) in enumerate(testing_dl):
    for image in images:
        scores, probs = zero_shot_classification(image, text_descriptions, clip_model, clip_processor)
        pred = probs.argmax().item()
        origin_pred.append(pred)

constr_accuracy = accuracy_score(true_labels, recon_pred)
origin_accuracy = accuracy_score(true_labels, origin_pred)
print(constr_accuracy, origin_accuracy)

constr_report = classification_report(true_labels, recon_pred, target_names=cifar10_classes)
origin_report = classification_report(true_labels, origin_pred, target_names=cifar10_classes)
print(constr_report, origin_report)


constr_report_df = pd.DataFrame(classification_report(true_labels, recon_pred, target_names=cifar10_classes, output_dict=True)).transpose()
origin_report_df = pd.DataFrame(classification_report(true_labels, origin_pred, target_names=cifar10_classes, output_dict=True)).transpose()

constr_hm = sns.heatmap(constr_report_df)
origin_hm = sns.heatmap(origin_report_df)

constr_metrics = constr_report_df.loc[cifar10_classes, ["precision", "recall", "f1-score"]]
origin_metrics = origin_report_df.loc[cifar10_classes, ["precision", "recall", "f1-score"]]

plt.figure(figsize=(10, 6))
sns.heatmap(constr_metrics, annot=True, cmap="Blues")
plt.title("Reconstructed Images - Classification Metrics")
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(origin_metrics, annot=True, cmap="Greens")
plt.title("Original Images - Classification Metrics")
plt.show()


diff_df = (origin_metrics - constr_metrics)
plt.figure(figsize=(10, 6))
sns.heatmap(diff_df, annot=True, cmap="Blues")
plt.title("Difference - Classification Metrics")
plt.show()


# Assuming testing_dl is your DataLoader and recon_tensor is your model's output
# We'll grab one batch to get the original images
dataiter = iter(testing_dl)
original_images, _ = next(dataiter)

# Select the first 10
orig_plot = original_images[:10]
recon_plot = recon_tensor[:10]

num_to_display = min(10, orig_plot.shape[0])

fig, axes = plt.subplots(nrows=2, ncols=num_to_display, figsize=(15, 4))

for i in range(num_to_display):
    # Process Original
    img_orig = orig_plot[i].cpu().numpy().transpose((1, 2, 0))
    # Process Original
    img_orig = orig_plot[i].cpu().numpy().transpose((1, 2, 0))
    # Undo normalization if applied (example: mean=0.5, std=0.5)
    img_orig = img_orig * 0.5 + 0.5 
    
    # Process Reconstructed
    img_recon = recon_plot[i].detach().cpu().numpy().transpose((1, 2, 0))
    img_recon = img_recon * 0.5 + 0.5
    
    # Top row: Originals
    axes[0, i].imshow(np.clip(img_orig, 0, 1))
    axes[0, i].axis('off')
    if i == 0: axes[0, i].set_title("Originals")

    # Bottom row: Reconstructions
    axes[1, i].imshow(np.clip(img_recon, 0, 1))
    axes[1, i].axis('off')
    if i == 0: axes[1, i].set_title("Reconstructed")

plt.tight_layout()
plt.show()
