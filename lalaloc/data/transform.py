import torchvision.transforms as transforms

def build_transform(config, is_train, is_layout=False):
    # TODO: Data augmentation, flip and rotate the camera
    # Needs to be applied to layouts as well
    transform = [
        transforms.Resize(config.INPUT.IMG_SIZE),
        transforms.ToTensor(),
    ]
    if is_layout:
        transform = [transforms.ToPILImage(),] + transform
    elif not config.TRAIN.NO_TRANSFORM:
        transform += [
            transforms.Normalize(
                mean=config.INPUT.NORMALISE_MEAN, std=config.INPUT.NORMALISE_MEAN
            ),
        ]
    transform = transforms.Compose(transform)
    return transform