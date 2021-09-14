import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from lalaloc.config import get_cfg_defaults, parse_args
from lalaloc.model import ImageFromLayout, Layout2LayoutDecode


if __name__ == "__main__":
    args = parse_args()

    config = get_cfg_defaults()
    config.merge_from_file(args.config_file)
    config.merge_from_list(args.opts)
    if args.val:
        config.TEST.VAL_AS_TEST = True
    config.freeze()
    print(config)

    pl.seed_everything(config.SEED)

    if args.checkpoint_file:
        resume_path = args.checkpoint_file
    else:
        resume_path = None

    if config.MODEL.QUERY_TYPE == "image":
        model = ImageFromLayout(config)
    elif config.MODEL.QUERY_TYPE == "layout":
        model = Layout2LayoutDecode(config)
    else:
        raise NotImplementedError(
            "The query type, {}, isn't recognised.".format(config.MODEL.QUERY_TYPE)
        )

    logger = loggers.TensorBoardLogger(config.OUT_DIR)
    checkpoint_callback = ModelCheckpoint(save_top_k=-1,)

    trainer = pl.Trainer(
        max_epochs=config.TRAIN.NUM_EPOCHS,
        gpus=config.SYSTEM.NUM_GPUS,
        logger=logger,
        distributed_backend=config.SYSTEM.DISTRIBUTED_BACKEND,
        limit_val_batches=25,
        resume_from_checkpoint=resume_path,
        num_sanity_val_steps=2,
        check_val_every_n_epoch=config.TRAIN.TEST_EVERY,
        callbacks=[checkpoint_callback],
    )
    if args.test_ckpt:
        assert config.SYSTEM.NUM_GPUS == 1
        load = torch.load(args.test_ckpt)
        model.load_state_dict(load["state_dict"], strict=False)
        trainer.test(model)
    else:
        trainer.fit(model)
