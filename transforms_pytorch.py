import albumentations as A


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def build_train_transform(config):
    t = config.train
    transforms = []

    if t.hflip:
        transforms.append(A.HorizontalFlip(p=0.5))

    if t.vflip:
        transforms.append(A.VerticalFlip(p=0.5))

    if t.resize != 0:
        transforms.append(
            A.Resize(height=t.resize, width=t.resize, p=1)
        )

    if t.random_scale != 0:
        transforms.append(
            A.RandomScale(scale_limit=(0, t.random_scale))
        )

    if t.random_rotate != 0:
        transforms.append(
            A.Rotate(limit=t.random_rotate)
        )

    if t.random_crop != 0:
        transforms.append(
            A.RandomCrop(
                height=t.random_crop,
                width=t.random_crop,
            )
        )

    if (
        t.color_jitter.brightness != 0
        or t.color_jitter.contrast != 0
        or t.color_jitter.saturation != 0
        or t.color_jitter.hue != 0
    ):
        transforms.append(
            A.ColorJitter(
                brightness=t.color_jitter.brightness,
                contrast=t.color_jitter.contrast,
                saturation=t.color_jitter.saturation,
                hue=t.color_jitter.hue,
            )
        )

    if t.gaussian_blur != 0:
        transforms.append(
            A.Defocus(radius=t.gaussian_blur)
        )

    # Not adding Gaussian noise for now because the current
    # configuration has gaussian_noise = 0 and Albumentations 2.x
    # changed the GaussNoise API.

    transforms.append(A.Normalize(mean=MEAN, std=STD))

    return A.Compose(transforms)


def build_val_transform(config):
    t = config.val

    return A.Compose([
        A.Resize(height=t.resize, width=t.resize),
        A.Normalize(mean=MEAN, std=STD),
    ])