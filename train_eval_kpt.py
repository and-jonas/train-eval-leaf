from pathlib import Path

import hydra
import pandas as pd
import numpy as np
from omegaconf import DictConfig, OmegaConf
from ultralytics import YOLO, settings
from torch.utils.tensorboard import SummaryWriter

settings.update({"tensorboard": True})

from pathlib import Path
import pandas as pd
import yaml


def prepare_yolo_fold(cfg, fold, run_dir, splits):

    # Select fold
    fold_splits = splits.loc[
        splits["fold"] == fold
    ].copy()

    if fold_splits.empty:
        raise ValueError(f"No samples found for fold {fold}")

    # Validate split assignments
    if fold_splits["sample_UID"].isna().any():
        raise ValueError(f"Missing sample_UID in fold {fold}")

    if fold_splits["sample_UID"].duplicated().any():
        raise ValueError(
            f"Duplicate sample_UID assignments in fold {fold}"
        )

    # Locate images
    original_yaml = Path(cfg.dataset.data_yaml).resolve()
    with open(original_yaml) as f:
        data_cfg = yaml.safe_load(f)
    if "kpt_shape" not in data_cfg:
        raise ValueError(f"Missing kpt_shape in {original_yaml}")
    data_root = Path(data_cfg.get("path", original_yaml.parent)).resolve()
    image_dir = data_root / "images"

    image_paths = {}

    for path in image_dir.rglob("*"):

        if path.suffix.lower() not in {
            ".jpg", ".jpeg", ".png"
        }:
            continue

        if path.stem in image_paths:
            raise ValueError(
                f"Duplicate image identifier: {path.stem}\n"
                f"{image_paths[path.stem]}\n"
                f"{path}"
            )

        image_paths[path.stem] = path.resolve()

    # Create directory for split definitions
    split_dir = run_dir / "split"
    split_dir.mkdir(parents=True, exist_ok=True)

    # Generate image lists
    for split_name, filename in [
        ("train", "train.txt"),
        ("validation", "val.txt"),
    ]:

        sample_ids = fold_splits.loc[
            fold_splits["split"] == split_name,
            "sample_UID",
        ].tolist()

        if not sample_ids:
            raise ValueError(
                f"No {split_name} samples in fold {fold}"
            )

        missing = set(sample_ids) - set(image_paths)

        if missing:
            raise ValueError(
                f"{len(missing)} missing images for "
                f"{split_name}, fold {fold}: "
                f"{sorted(missing)[:10]}"
            )

        paths = [
            str(image_paths[sid])
            for sid in sample_ids
        ]

        (split_dir / filename).write_text(
            "\n".join(paths) + "\n"
        )

        print(
            f"Fold {fold} | {split_name}: "
            f"{len(paths)} images"
        )

    # Load original YOLO dataset configuration
    with open(cfg.dataset.data_yaml) as f:
        data_cfg = yaml.safe_load(f)

    # Replace train/validation paths
    data_cfg["train"] = str(
        (split_dir / "train.txt").resolve()
    )
    data_cfg["val"] = str(
        (split_dir / "val.txt").resolve()
    )

    data_cfg["path"] = str(data_root.resolve())

    # Save fold-specific YAML
    fold_yaml = split_dir / "dataset.yaml"

    with open(fold_yaml, "w") as f:
        yaml.safe_dump(data_cfg, f)

    return fold_yaml


class PoseTensorBoardLogger:
    """Log class-specific pose metrics after each validation epoch."""

    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=str(log_dir))
        self.epoch = 0

    def on_train_epoch_start(self, trainer):
        self.epoch = trainer.epoch + 1

    def on_val_end(self, validator):

        print(
            f"[TensorBoard] Validation callback, epoch {self.epoch}",
            flush=True,
        )

        pose = validator.metrics.pose

        if pose is None:
            print("[TensorBoard] Pose metrics unavailable", flush=True)
            return

        confidence_values = np.asarray(pose.px)
        f1_values = np.asarray(pose.f1_curve)
        p_values = np.asarray(pose.p_curve)
        r_values = np.asarray(pose.r_curve)

        if f1_values.size == 0:
            print("[TensorBoard] F1 curves unavailable", flush=True)
            return

        for i, class_id in enumerate(pose.ap_class_index):

            class_name = validator.names[int(class_id)]

            _, _, ap50, ap50_95 = pose.class_result(i)

            best_idx = int(np.argmax(f1_values[i]))

            self.writer.add_scalar(
                f"{class_name}/mAP50",
                float(ap50),
                self.epoch,
            )

            self.writer.add_scalar(
                f"{class_name}/mAP50_95",
                float(ap50_95),
                self.epoch,
            )

            self.writer.add_scalar(
                f"{class_name}/F1",
                float(f1_values[i, best_idx]),
                self.epoch,
            )

            self.writer.add_scalar(
                f"{class_name}/Confidence",
                float(confidence_values[best_idx]),
                self.epoch,
            )

            self.writer.add_scalar(
                f"{class_name}/Precision",
                float(p_values[i, best_idx]),
                self.epoch,
            )

            self.writer.add_scalar(
                f"{class_name}/Recall",
                float(r_values[i, best_idx]),
                self.epoch,
            )

        self.writer.flush()

        print(
            f"[TensorBoard] Logged pose metrics for epoch {self.epoch}",
            flush=True,
        )

    def close(self):
        self.writer.close()


def run_fold(fold, cfg, splits, output_dir):

    dataset_cfg = cfg.dataset
    model_cfg = cfg.model
    opt_cfg = cfg.optimization

    # Unique output directory for every configuration
    run_dir = (
        output_dir
        / str(dataset_cfg.dataset_name)
        / f"yolo11{model_cfg.size}"
        / f"seed_{cfg.seed}"
        / f"fold_{fold}"
    ).resolve()

    run_dir.mkdir(parents=True, exist_ok=True)

    results_path = run_dir / "validation_metrics.csv"

    # Skip completed runs
    if results_path.exists():
        print(f"Already completed: {run_dir}")
        return

    # Save Hydra configuration
    OmegaConf.save(cfg, run_dir / "config.yaml", resolve=True)

    # Generate fold-specific YOLO dataset configuration
    if splits is not None:
        data_yaml = prepare_yolo_fold(
            cfg=cfg,
            fold=fold,
            run_dir=run_dir,
            splits=splits,
        )
        # extract n_train and n_val
        with open(data_yaml, "r") as f:
            data_config = yaml.safe_load(f)
        def count_images(file_path):
            with open(file_path, "r") as f:
                return sum(1 for line in f if line.strip())
        n_train = count_images(data_config["train"])
        n_val = count_images(data_config["val"])
    else:
        data_yaml = cfg.dataset.data_yaml

        with open(data_yaml, "r") as f:
            data_config = yaml.safe_load(f)

        root = Path(data_config["path"])

        train_dir = root / data_config["train"]
        val_dir = root / data_config["val"]

        def count_images(directory):
            return sum(
                1 for p in directory.iterdir()
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )

        n_train = count_images(train_dir)
        n_val = count_images(val_dir)



    print(f"Fold {fold}: {n_train} training images, {n_val} validation images")

    # Use pretrained segmentation model
    model = YOLO(str(model_cfg.weights))

    # log validation metrics to tensorboard
    tb_logger = PoseTensorBoardLogger(log_dir=run_dir)
    model.add_callback(
        "on_train_epoch_start",
        tb_logger.on_train_epoch_start,
    )
    model.add_callback(
        "on_val_end",
        tb_logger.on_val_end,
    )

    # Resume interrupted training when possible
    last_ckpt = run_dir / "weights" / "last.pt"

    if last_ckpt.exists():
        print(f"Resuming from {last_ckpt}")
        model = YOLO(str(last_ckpt))
        model.train(resume=True, device=0)

    else:
        model.train(
            data=str(data_yaml),
            epochs=int(opt_cfg.epochs),
            imgsz=1024,
            batch=int(opt_cfg.batch_size),
            workers=int(opt_cfg.n_workers),
            seed=int(cfg.seed),
            device=0,
            max_det=1500,
            project=str(run_dir.parent),
            name=run_dir.name,
            exist_ok=True,
            pretrained=True,
            patience=int(opt_cfg.early_stopping.patience) if opt_cfg.early_stopping.enabled else 0,
            save=True,
            save_period=-1,
            plots=False,
            verbose=False,
        )

    # Validate the best or the final checkpoint
    early_stopping_enabled = bool(opt_cfg.early_stopping.enabled)
    checkpoint_type = "best" if early_stopping_enabled else "last"
    selected_ckpt = run_dir / "weights" / f"{checkpoint_type}.pt"
    if not selected_ckpt.exists():
        raise FileNotFoundError(selected_ckpt)

    evaluation_model = YOLO(str(selected_ckpt))

    metrics = evaluation_model.val(
        data=str(data_yaml),
        split="val",
        device=0,
        imgsz=1024,
        max_det=1500,
        plots=False,
        project=str(run_dir),
        name="validation",
        exist_ok=True,
    )

    # Overall pose metrics
    pose = metrics.pose

    result = {
        "experiment": str(cfg.run_name),
        "split_strategy": (str(cfg.split.name) if cfg.get("split") is not None else "pre-defined"),
        "fold": int(fold),
        "seed": int(cfg.seed),
        "n_train": n_train,
        "n_val": n_val,    
        "backbone": f"yolo11{model_cfg.size}",
        "dataset": str(dataset_cfg.dataset_name),
        "epochs_max": int(opt_cfg.epochs),
        "epochs_trained": int(YOLO(str(last_ckpt)).ckpt["epoch"]) + 1,
        "best_epoch": int(evaluation_model.ckpt["epoch"]) + 1,  
        "checkpoint_type": checkpoint_type,

        # Overall metrics
        "pose_precision": float(pose.mp),
        "pose_recall": float(pose.mr),
        "pose_mAP50": float(pose.map50),
        "pose_mAP50_95": float(pose.map),
    }

    # Confidence thresholds and F1 curves
    confidence_values = np.asarray(pose.px)
    f1_values = np.asarray(pose.f1_curve)
    p_values = np.asarray(pose.p_curve)
    r_values = np.asarray(pose.r_curve)

    # Class-specific metrics
    for i, class_id in enumerate(pose.ap_class_index):

        class_name = metrics.names[int(class_id)]

        precision, recall, ap50, ap50_95 = pose.class_result(i)

        # Confidence threshold maximizing F1
        best_idx = int(np.argmax(f1_values[i]))

        result[f"{class_name}_mAP50"] = float(ap50)
        result[f"{class_name}_mAP50_95"] = float(ap50_95)

        result[f"{class_name}_F1"] = float(
            f1_values[i, best_idx]
        )

        result[f"{class_name}_confidence"] = float(
            confidence_values[best_idx]
        )

        result[f"{class_name}_precision"] = float(
            p_values[i, best_idx]
        )

        result[f"{class_name}_recall"] = float(
            r_values[i, best_idx]
        )

    # Save results
    pd.DataFrame([result]).to_csv(
        results_path,
        index=False,
    )

    print(f"Results saved to: {results_path}")

    return result


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="experiment/02_kpt_architecture"
)
def main(cfg: DictConfig):

    if "split" in cfg and cfg.split is not None:
        split_name = cfg.split.name

        # Read split assignments
        splits = pd.read_csv(
            cfg.split.path,
            dtype={"sample_UID": str},
        )

        # Select splitting strategy
        split_experiment = cfg.split.get(
            "experiment", cfg.split.name
        )

        splits = splits[
            splits["experiment"] == split_experiment
        ].copy()

        if splits.empty:
            raise ValueError(
                f"No splits found for '{cfg.split.name}' "
                f"in {cfg.split.path}"
            )

        # Determine available folds
        available_folds = sorted(
            int(f) for f in splits["fold"].unique()
        )

        if cfg.split.folds == "all":
            folds = available_folds
        else:
            folds = [int(f) for f in cfg.split.folds]

        # Optional single-fold override
        if cfg.fold is not None:
            folds = [int(cfg.fold)]

        if not set(folds).issubset(set(available_folds)):
            raise ValueError(
                f"Requested folds {folds} not found. "
                f"Available folds: {available_folds}"
            )

    else:

        # Predefined train/validation split in dataset YAML
        with open(cfg.dataset.data_yaml, "r") as f:
            data_config = yaml.safe_load(f)

        if "train" not in data_config or "val" not in data_config:
            raise ValueError(
                "No split configuration provided, and dataset YAML "
                "does not define both 'train' and 'val'."
            )

        split_name = "pre-defined"
        splits = None
        folds = [0]

        if cfg.fold is not None and int(cfg.fold) != 0:
            raise ValueError(
                "Predefined train/validation datasets only support fold 0"
            )

    # Output directory
    output_dir = (
        Path("experiments")
        / str(cfg.task)
        / str(cfg.run_name)
        / str(split_name)
    ).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment: {cfg.run_name}")
    print(f"Dataset: {cfg.dataset.dataset_name}")
    print(f"Split strategy: {split_name}")
    print(f"Folds: {folds}")

    # Run cross-validation
    for fold in folds:

        run_fold(
            fold=fold,
            cfg=cfg,
            splits=splits,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()