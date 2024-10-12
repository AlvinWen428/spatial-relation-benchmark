from yacs.config import CfgNode as CN

_C = CN()
# -----------------------------------------------------------------------------
# EXPERIMENT
# -----------------------------------------------------------------------------
_C.EXP = CN()
_C.EXP.MODEL_NAME = '2d'
_C.EXP.EXP_ID = ""
_C.EXP.SEED = 0
_C.EXP.OUTPUT_DIR = "./results"
# -----------------------------------------------------------------------------
# DATALOADER
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
_C.DATALOADER.dataset_name = "rel3d"
_C.DATALOADER.batch_size = 128
_C.DATALOADER.num_workers = 4
_C.DATALOADER.datapath = "./data/rel3d/c_0.9_c_0.1.json"
_C.DATALOADER.load_img = False
_C.DATALOADER.crop = False
_C.DATALOADER.norm_data = True
_C.DATALOADER.data_aug_shift = False
_C.DATALOADER.data_aug_color = False
_C.DATALOADER.resize_mask = False
_C.DATALOADER.trans_vec = []
_C.DATALOADER.predicate_dim = 30
_C.DATALOADER.object_dim = 67
# -----------------------------------------------------------------------------
# TRAINING DETAILS
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.num_epochs = 200
_C.TRAIN.optimizer = "adam"
_C.TRAIN.learning_rate = 1e-3
_C.TRAIN.lr_decay_ratio = 0.5
_C.TRAIN.l2 = 0.0
_C.TRAIN.layer_decay = 1.0
_C.TRAIN.scheduler = "plateau"
_C.TRAIN.early_stop = 20
_C.TRAIN.patience = 10
_C.TRAIN.warmup = 0
# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# -----------------------------------------------------------------------------
# Vision Transformer
# -----------------------------------------------------------------------------
_C.MODEL.VISION_TRANSFORMER = CN()
_C.MODEL.VISION_TRANSFORMER.in_chans = 3
_C.MODEL.VISION_TRANSFORMER.patch_size = 16
_C.MODEL.VISION_TRANSFORMER.embed_dim = 768
_C.MODEL.VISION_TRANSFORMER.depth = 12
_C.MODEL.VISION_TRANSFORMER.num_heads = 12
_C.MODEL.VISION_TRANSFORMER.drop_rate = 0.0
_C.MODEL.VISION_TRANSFORMER.drop_path_rate = 0.0
_C.MODEL.VISION_TRANSFORMER.pretrained = ""
_C.MODEL.VISION_TRANSFORMER.readnet_d_hidden = 512
_C.MODEL.VISION_TRANSFORMER.readnet_dropout = 0.0
# -----------------------------------------------------------------------------
# RegionViT
# -----------------------------------------------------------------------------
_C.MODEL.REGIONVIT = CN()
_C.MODEL.REGIONVIT.backbone = 'vit_base'
_C.MODEL.REGIONVIT.hidden_dim = 128
_C.MODEL.REGIONVIT.pretrain_ckp = ""
_C.MODEL.REGIONVIT.box_feature_size = 128
# -----------------------------------------------------------------------------
# CrossAttnViT
# -----------------------------------------------------------------------------
_C.MODEL.CROSSATTNVIT = CN()
_C.MODEL.CROSSATTNVIT.decoder_depth = 1
_C.MODEL.CROSSATTNVIT.decoder_drop_rate = 0.0
_C.MODEL.CROSSATTNVIT.decoder_attn_drop_rate = 0.0
_C.MODEL.CROSSATTNVIT.emb_pool = 'max-in-roi'
# -----------------------------------------------------------------------------
# RelatiViT
# -----------------------------------------------------------------------------
_C.MODEL.RELATIVIT = CN()
_C.MODEL.RELATIVIT.prompt_emb_pool = 'max-in-roi'
_C.MODEL.RELATIVIT.use_attn_mask = True
# -----------------------------------------------------------------------------
# CNNTransformer
# -----------------------------------------------------------------------------
_C.MODEL.CNNTRANSFORMER = CN()
_C.MODEL.CNNTRANSFORMER.backbone = 'resnet50'
_C.MODEL.CNNTRANSFORMER.pretrained_resnet = ''
_C.MODEL.CNNTRANSFORMER.roi_pool_size = 3
_C.MODEL.CNNTRANSFORMER.prompt_emb_pool = 'max'
_C.MODEL.CNNTRANSFORMER.use_attn_mask = True


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
