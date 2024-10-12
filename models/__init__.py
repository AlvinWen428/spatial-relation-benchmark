from .regionvit import RegionViT
from .cnn_transformer import CNNTransformer
from .cross_attention_vit import CrossAttnViT
from .relativit import RelatiViT


def build_model(cfg):
    if cfg.EXP.MODEL_NAME == 'regionvit':
        model = RegionViT(
            predicate_dim=cfg.DATALOADER.predicate_dim,
            **cfg.MODEL.REGIONVIT
        )
    elif cfg.EXP.MODEL_NAME == 'relativit':
        model = RelatiViT(
            predicate_dim=cfg.DATALOADER.predicate_dim,
            **cfg.MODEL.VISION_TRANSFORMER,
            **cfg.MODEL.RELATIVIT
        )
    elif cfg.EXP.MODEL_NAME == 'cnn-transformer':
        model = CNNTransformer(
            predicate_dim=cfg.DATALOADER.predicate_dim,
            **cfg.MODEL.VISION_TRANSFORMER,
            **cfg.MODEL.CNNTRANSFORMER
        )
    elif cfg.EXP.MODEL_NAME == 'cross-attn-vit':
        model = CrossAttnViT(
            predicate_dim=cfg.DATALOADER.predicate_dim,
            **cfg.MODEL.VISION_TRANSFORMER,
            **cfg.MODEL.CROSSATTNVIT,
        )
    else:
        raise ValueError(f"Model {cfg.EXP.MODEL_NAME} not supported")

    return model
