import torch
from model import UNet

checkpoint_path = "models/best_unet.pth"

model = UNet(in_channels=14, out_channels=1)

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Model loaded successfully!")
print("Training epoch:", checkpoint["epoch"])
print("Validation loss:", checkpoint["val_loss"])

# Test input: batch=1, channels=14, height=128, width=128
dummy_input = torch.randn(1, 14, 128, 128)

with torch.no_grad():
    output = model(dummy_input)

print("Output shape:", output.shape)