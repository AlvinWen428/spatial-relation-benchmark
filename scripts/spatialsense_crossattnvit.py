import os


pretrained_vit_path = "./data/pretrained_checkpoints/ibot_vit_base_patch16.pth"
for pretrained_model in ["ibot"]:
    for seed in range(5):
        command = f"CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --master_port 10006 --nproc_per_node 4 \
            main.py --exp-config configs/spatialsense/cross_attn_vit.yaml \
            EXP.SEED {seed} \
            EXP.MODEL_NAME cross-attn-vit \
            EXP.EXP_ID spatialsenseplus_CrossAttnViT_seed{seed} \
            DATALOADER.datapath ./data/spatial_sense/annots_spatialsenseplus.json \
            MODEL.VISION_TRANSFORMER.pretrained {pretrained_vit_path} "

        os.system(command)
