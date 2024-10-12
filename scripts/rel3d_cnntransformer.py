import os


pretrained_resnet_path = "supervised"
pretrained_vit_path = "./data/pretrained_checkpoints/ibot_vit_base_patch16.pth"
for seed in range(5):
    command = f"CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --master_port 10001 --nproc_per_node 4 \
        main.py --exp-config configs/rel3d/cnn_transformer.yaml \
        EXP.SEED {seed} \
        EXP.MODEL_NAME cnn-transformer \
        EXP.EXP_ID rel3d_CNNTransformer_seed{seed} \
        MODEL.CNNTRANSFORMER.pretrained_resnet {pretrained_resnet_path} \
        MODEL.VISION_TRANSFORMER.pretrained {pretrained_vit_path} "

    os.system(command)
