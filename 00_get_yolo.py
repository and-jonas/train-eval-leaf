import shutil
from pathlib import Path
from ultralytics import YOLO

checkpoint_dir = Path("/scratch0/f89601470/torch/hub/checkpoints")
checkpoint_dir.mkdir(parents=True, exist_ok=True)

for size in ["n", "s", "m", "l"]:
    checkpoint_path = checkpoint_dir / f"yolo11{size}-pose.pt"

    if checkpoint_path.exists():
        print(f"Already exists: {checkpoint_path}")
        continue

    print(f"Downloading YOLO11{size}-pose...")

    model = YOLO(f"yolo11{size}-pose.pt")

    downloaded_path = Path(f"yolo11{size}-pose.pt")

    shutil.move(str(downloaded_path), str(checkpoint_path))

    print(f"Saved: {checkpoint_path}")