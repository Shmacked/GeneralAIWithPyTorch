import torch

def profile_model_size(model, input_size, device):
    dummy_input = torch.randn(input_size).to(device)
    with torch.no_grad():
        model(dummy_input)
    params = sum(p.numel() for p in model.parameters())
    return params