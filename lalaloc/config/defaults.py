from yacs.config import CfgNode as CN

_C = CN()
_C.OUT_DIR = "./runs"
_C.SEED = 42

_C.SYSTEM = CN()
_C.SYSTEM.NUM_WORKERS = 0
_C.SYSTEM.NUM_GPUS = 1
_C.SYSTEM.DISTRIBUTED_BACKEND = "ddp"


_C.INPUT = CN()

_C.INPUT.IMG_SIZE = (256, 512)
_C.INPUT.NORMALISE_MEAN = [0.485, 0.456, 0.406]
_C.INPUT.NORMALISE_STD = [0.229, 0.224, 0.225]


_C.DATASET = CN()

_C.DATASET.PATH = "/home/henry/Data/datasets/structured3d/Structured3D"
_C.DATASET.TEST_LIGHTING = ["warm"]
_C.DATASET.TEST_FURNITURE = ["full"]
_C.DATASET.TRAIN_LIGHTING = ["raw", "warm", "cold"]
_C.DATASET.TRAIN_FURNITURE = ["empty", "full", "simple"]
_C.DATASET.AUGMENT_LAYOUTS = False


_C.RENDER = CN()

_C.RENDER.IMG_SIZE = (256, 512)
_C.RENDER.USE_CUDA = True
_C.RENDER.BATCH_SIZE = 256
_C.RENDER.INVALID_VALUE = -1000


_C.MODEL = CN()

_C.MODEL.QUERY_TYPE = "image"
_C.MODEL.PANORAMA_BACKBONE = "resnet18"
_C.MODEL.LAYOUT_BACKBONE = "resnet18"
_C.MODEL.DESC_LENGTH = 128
_C.MODEL.NORMALISE_EMBEDDING = True
_C.MODEL.EMBEDDER_TYPE = "fc"
_C.MODEL.DECODER_RESOLUTION = (32, 64)

_C.POSE_REFINE = CN()
_C.POSE_REFINE.LR = 0.01
_C.POSE_REFINE.SCHEDULER_PATIENCE = 10
_C.POSE_REFINE.SCHEDULER_THRESHOLD = 0.05
_C.POSE_REFINE.SCHEDULER_DECAY = 0.5
_C.POSE_REFINE.CONVERGANCE_THRESHOLD = 0.0001
_C.POSE_REFINE.CONVERGANCE_PATIENCE = 20
_C.POSE_REFINE.MAX_ITERS = 150
_C.POSE_REFINE.RENDER_SIZE = (256, 512)


_C.TRAIN = CN()

_C.TRAIN.NUM_EPOCHS = 10
_C.TRAIN.BATCH_SIZE = 16
_C.TRAIN.TEST_EVERY = 1
_C.TRAIN.LOSS = "triplet"
_C.TRAIN.GT_LAYOUT_LOSS = False

_C.TRAIN.INITIAL_LR = 0.01
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WEIGHT_DECAY = 1e-4
_C.TRAIN.LR_MILESTONES = [5, 8]
_C.TRAIN.LR_GAMMA = 0.1

_C.TRAIN.NEAR_MIN_DIST = 0.0
_C.TRAIN.NEAR_MAX_DIST = 0.5
_C.TRAIN.FAR_MIN_DIST = 2
_C.TRAIN.FAR_MAX_DIST = 10
_C.TRAIN.NUM_NEAR_SAMPLES = 1
_C.TRAIN.NUM_FAR_SAMPLES = 1

_C.TRAIN.APPEND_GT = False
_C.TRAIN.NO_TRANSFORM = False

_C.TRAIN.SOURCE_WEIGHTS = ""
_C.TRAIN.COMPUTE_GT_DIST = True

_C.TRAIN.DISTANCE_LOSS_SCALE = 1.0
_C.TRAIN.LAYOUT_LOSS_SCALE = 1.0
_C.TRAIN.DECODER_LOSS_SCALE = 0.001

_C.TRAIN.CUT_GRAD = True


_C.TEST = CN()

_C.TEST.BATCH_SIZE = 1
_C.TEST.POSE_SAMPLE_STEP = 500
_C.TEST.LAYOUTS_MAX_BATCH = 64
_C.TEST.VOGEL_DISC_REFINE = False
_C.TEST.VOGEL_SAMPLES = 100
_C.TEST.VAL_AS_TEST = False
_C.TEST.LATENT_POSE_OPTIMISATION = False
_C.TEST.DECODE_REFINE = False
_C.TEST.DECODE_USE_GT = False
_C.TEST.METRIC_DUMP = ""


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
