from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig
import hydra

from dataset import SegmentationDataset
from transforms_pytorch import (
    build_train_transform,
    build_val_transform,
)

from lightning_model import SegmentationModel

import pandas as pd


def run_fold(fold, cfg, splits, output_dir):

    # Access configurations
    dataset_cfg = cfg.dataset
    model_cfg = cfg.model
    optimization_cfg = cfg.optimization
    augmentation_cfg = cfg.data_augmentation

    pl.seed_everything(int(cfg.seed), workers=True)

    # Output directory
    fold_dir = (
        output_dir
        / str(dataset_cfg.dataset_name)
        / str(model_cfg.backbone)
        / f"seed_{cfg.seed}"
        / f"fold_{fold}"
    )
    fold_dir.mkdir(parents=True, exist_ok=True)

    # Datasets
    train_transform = build_train_transform(augmentation_cfg)
    val_transform = build_val_transform(augmentation_cfg)
    if "train" in dataset_cfg and "val" in dataset_cfg:
        train_dataset = SegmentationDataset(
            data_dir=dataset_cfg.train.data,
            labels_dir=dataset_cfg.train.labels,
            transform=train_transform,
        )
        val_dataset = SegmentationDataset(
            data_dir=dataset_cfg.val.data,
            labels_dir=dataset_cfg.val.labels,
            transform=val_transform,
        )
    else:
        fold_splits = splits.loc[ splits["fold"] == fold].copy()
        train_ids = set(fold_splits.loc[fold_splits["split"] == "train", "sample_UID"])
        val_ids = set(fold_splits.loc[fold_splits["split"] == "validation", "sample_UID"])
        assert not train_ids & val_ids
        assert train_ids and val_ids
        train_dataset = SegmentationDataset(
            data_dir=dataset_cfg.data,
            labels_dir=dataset_cfg.labels,
            transform=train_transform,
            sample_ids=train_ids,
        )
        val_dataset = SegmentationDataset(
            data_dir=dataset_cfg.data,
            labels_dir=dataset_cfg.labels,
            transform=val_transform,
            sample_ids=val_ids,
        )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=optimization_cfg.batch_size,
        shuffle=True,
        num_workers=optimization_cfg.n_workers,
        pin_memory=int(optimization_cfg.n_workers) > 0,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=optimization_cfg.batch_size,
        shuffle=False,
        num_workers=optimization_cfg.n_workers,
        pin_memory=int(optimization_cfg.n_workers) > 0,
        persistent_workers=True,
    )

    # Model
    model = SegmentationModel(
        optimizer_name=str(optimization_cfg.optimizer.name),
        learning_rate=float(optimization_cfg.learning_rate),
        weight_decay=float(optimization_cfg.weight_decay),
        strategy=str(optimization_cfg.strategy),
        class_weights=list(optimization_cfg.class_weights),
        num_classes=int(model_cfg.n_classes),
        class_names=model_cfg.class_names,
        metric_classes=model_cfg.metric_classes,
        encoder_name=model_cfg.backbone,
        encoder_weights_path=model_cfg.encoder_weights_path,
    )

    # TensorBoard
    logger = TensorBoardLogger(save_dir=str(fold_dir), name="", version=0)

    # save config
    run_dir = Path(logger.log_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # skip existing
    if (run_dir / "results.csv").exists():
        print(f"Already completed: {run_dir}")
        return

    OmegaConf.save(cfg, run_dir / "config.yaml", resolve=True)

    # Callbacks
    lr_monitor = LearningRateMonitor(
        logging_interval="epoch",
    )
    checkpoint = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename="best-{epoch:02d}",
        monitor=optimization_cfg.early_stopping.monitor,
        mode=optimization_cfg.early_stopping.mode,
        save_top_k=1 if optimization_cfg.early_stopping.enabled else 0,
        save_last=False,
    )
    callbacks = [lr_monitor, checkpoint]
    if optimization_cfg.early_stopping.enabled:
        early_stopping = EarlyStopping(
            monitor=optimization_cfg.early_stopping.monitor,
            mode=optimization_cfg.early_stopping.mode,
            patience=optimization_cfg.early_stopping.patience,
            min_delta=optimization_cfg.early_stopping.min_delta,
        )
        callbacks.append(early_stopping)

    # Verify dataset sizes
    print(f"Train: {len(train_dataset)}")
    print(f"Validation: {len(val_dataset)}")

    # train and monitor on the held-out fold
    trainer = pl.Trainer(
        max_epochs=int(optimization_cfg.epochs),
        accelerator="gpu",
        # devices=[int(optimization_cfg.device)],
        devices=1,
        accumulate_grad_batches=optimization_cfg.accumulate_grad_batches,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        enable_checkpointing=True,
        num_sanity_val_steps=0
    )
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # Select model for final evaluation
    if optimization_cfg.early_stopping.enabled:
        ckpt_path = "best"
        checkpoint_type = "best"
    else:
        ckpt_path = None
        checkpoint_type = "last"

    # Evaluate selected model on the held-out fold
    fold_metrics = trainer.validate(
        model,
        dataloaders=val_loader,
        ckpt_path=ckpt_path,
        weights_only=False,
    )[0]

    # save results
    result = {
        "experiment": cfg.run_name,
        "fold": int(fold),
        "seed": int(cfg.seed),
        "backbone": str(model_cfg.backbone),
        "dataset": str(dataset_cfg.dataset_name),
        "n_train": len(train_dataset),
        "n_val": len(val_dataset),
        "epochs_max": int(optimization_cfg.epochs),
        "epochs_trained": int(trainer.current_epoch),
        "checkpoint_type": checkpoint_type,
        **{
            key: float(value)
            for key, value in fold_metrics.items()
        },
    }
    pd.DataFrame([result]).to_csv(run_dir / "results.csv", index=False)
    print(f"\nResults saved to: {run_dir / 'results.csv'}")

    return result


@hydra.main(
        version_base=None, 
        config_path="config", 
        config_name="experiment/05_seg_org_architecture"
    )
def main(cfg: DictConfig):

    # print(OmegaConf.to_yaml(cfg))

    task = cfg.task
    experiment = cfg.run_name

    # get samples and their metadata for one fold of an experiment
    if "train" in cfg.dataset and "val" in cfg.dataset:
        splits = None
        split_name = "pre-defined"
        folds = [0]
        if cfg.fold is not None and int(cfg.fold) != 0:
            raise ValueError(
                "Predefined train/validation datasets only support fold 0"
            )
    else:
        # get training and validation sample_UIDs

        # Splitting strategy
        split_name = cfg.split.name

        # Read split assignments                                           
        splits = pd.read_csv(cfg.split.path, dtype={"sample_UID": str})
        if splits.empty:
            raise ValueError(f"No splits found for {cfg.dataset.dataset_name}: {cfg.split.name}")

        # Determine which folds to run
        available_folds = sorted(splits["fold"].unique())
        if cfg.split.folds == "all":
            folds = available_folds
        else:
            folds = list(cfg.split.folds)

        # Optional: select one fold via command line
        if cfg.fold is not None:
            folds = [int(cfg.fold)]

        if not set(folds).issubset(set(available_folds)):
            raise ValueError(
                f"Requested folds {folds} not found. "
                f"Available folds: {available_folds}"
            )

    print(f"Experiment: {experiment}")
    print(f"Dataset: {cfg.dataset.dataset_name}")
    print(f"Folds to run: {folds}")

    # make output directory
    output_dir = Path("experiments") / task / experiment / split_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # perform evaluation using the pre-defined resampling procedure
    for fold in folds:
        run_fold(fold=fold, cfg=cfg, splits=splits, output_dir=output_dir)


if __name__ == "__main__":
    main()