
from pathlib import Path
import torch
import segmentation_models_pytorch as smp

# Where to store pretrained encoder weights
checkpoint_dir = Path(
    "/scratch0/f89601470/torch/hub/checkpoints"
)
checkpoint_dir.mkdir(parents=True, exist_ok=True)

backbones = [
    "mit_b0",
    "mit_b1",
    "mit_b2",
    "mit_b3",
    "mit_b4",
]

for backbone in backbones:

    checkpoint_path = checkpoint_dir / f"{backbone}.pth"

    if checkpoint_path.exists():
        print(f"Already exists: {checkpoint_path}")
        continue

    print(f"Loading pretrained {backbone}...")

    # SMP downloads pretrained weights if not already cached
    model = smp.FPN(
        encoder_name=backbone,
        encoder_weights="imagenet",
        in_channels=3,
        classes=5,
    )

    # Save ONLY encoder weights in the format expected by your model
    torch.save(
        model.encoder.state_dict(),
        checkpoint_path,
    )

    print(f"Saved: {checkpoint_path}")

    del model

print("All encoder checkpoints are ready.")