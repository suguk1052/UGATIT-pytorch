import torch
from torchvision import models

# Provide a custom Identity for older PyTorch versions
if hasattr(torch.nn, "Identity"):
    Identity = torch.nn.Identity
else:
    class Identity(torch.nn.Module):
        def forward(self, x):
            return x


def load_inception():
    """Load torchvision Inception V3 with feature outputs.

    The loader works with both old and new torchvision versions.
    """
    if hasattr(models, "Inception_V3_Weights"):
        weights = models.Inception_V3_Weights.DEFAULT
        inception = models.inception_v3(weights=weights)
    else:
        inception = models.inception_v3(pretrained=True)
    inception.fc = Identity()
    inception.eval()
    return inception


if __name__ == "__main__":
    # Smoke test for manual execution
    net = load_inception()
    print("Loaded Inception V3")
