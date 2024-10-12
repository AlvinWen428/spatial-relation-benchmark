import os


pretrained_vit_path = "./data/pretrained_checkpoints/ibot_vit_base_patch16.pth"
for seed in range(5):
    command = f"CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --master_port 10004 --nproc_per_node 4 \
        main.py --exp-config configs/spatialsense/regionvit.yaml \
        EXP.SEED {seed} \
        EXP.MODEL_NAME regionvit \
        EXP.EXP_ID spatialsenseplus_RegionViT_seed{seed} \
        DATALOADER.datapath ./data/spatial_sense/annots_spatialsenseplus.json \
        MODEL.REGIONVIT.pretrain_ckp {pretrained_vit_path} "
    os.system(command)
