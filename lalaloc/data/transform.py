import torchvision.transforms as transforms


def build_transform(config, is_train, is_layout=False):
    # TODO: Data augmentation, flip and rotate the camera
    # Needs to be applied to layouts as well
    in_size = config.INPUT.LAYOUT_SIZE if is_layout else config.INPUT.IMG_SIZE

    transform = [
        transforms.Resize(in_size),
        transforms.ToTensor(),
    ]
    if is_layout:
        transform = [transforms.ToPILImage(),] + transform
    elif not config.TRAIN.NO_TRANSFORM:
        transform += [
            transforms.Normalize(
                mean=config.INPUT.NORMALISE_MEAN, std=config.INPUT.NORMALISE_STD
            ),
        ]
    transform = transforms.Compose(transform)
    return transform
