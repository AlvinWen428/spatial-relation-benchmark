import os
import shutil
from pathlib import Path
import gdown
import argparse


def wgetgdrive(file_id, output_path):
    URL = f"https://docs.google.com/uc?export=download&id={file_id}"
    gdown.download(URL, str(output_path), quiet=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir", type=str, default="./data")
    parser.add_argument("--data-key", type=str, default="rel3d")
    args = parser.parse_args()

    target_dir = Path(args.target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path("./tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if args.data_key == "rel3d":
        # Ackownledgement: Goyal, A., Yang, K., Yang, D., & Deng, J. (2020). Rel3d: A minimally contrastive benchmark for grounding spatial relations in 3d. Advances in Neural Information Processing Systems, 33, 10514-10525.
        file_id = "1sebXU7pZ0FI7lG28OkH5qSnWHpADQRFi"
        wgetgdrive(file_id, tmp_dir / "data_min.zip")
        os.system(f"unzip -o {tmp_dir / 'data_min.zip'}")
        shutil.move("data_min", target_dir / "test_rel3d")
        os.remove(tmp_dir / "data_min.zip")

    elif args.data_key == "spatialsense+":
        # Acknowledgement: Yang, Kaiyu, Olga Russakovsky, and Jia Deng. "Spatialsense: An adversarially crowdsourced benchmark for spatial relation recognition." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019.
        # first download the original SpatialSense dataset
        # os.system(f"wget https://zenodo.org/api/records/8104370/files-archive -O {tmp_dir / 'spatialsense.zip'} ")
        spatialsense_dir = target_dir / "spatialsense"
        spatialsense_dir.mkdir(parents=True, exist_ok=True)
        spatialsense_image_dir = spatialsense_dir / "images"
        spatialsense_image_dir.mkdir(parents=True, exist_ok=True)
        os.system(f"unzip -o {tmp_dir / 'spatialsense.zip'} -d {spatialsense_dir}")
        os.system(f"tar -zxvf {spatialsense_dir / 'images.tar.gz'} -C {spatialsense_image_dir}")
        # then download the annotations for SpatialSense+
        file_id = "1vIOozqk3OlxkxZgL356pD1EAGt06ZwM4"
        wgetgdrive(file_id, spatialsense_dir / "annots_spatialsenseplus.json")
        os.remove(tmp_dir / "spatialsense.zip")

    elif args.data_key == "ibot":
        # Acknowledgement: Zhou, J., Wei, C., Wang, H., Shen, W., Xie, C., Yuille, A., & Kong, T. ibot: Image BERT Pre-training with Online Tokenizer. In International Conference on Learning Representations.
        file_id = "1nO06i4xc8RAp2W8xm06cO0oTHwIZ2VAX"
        pretrained_ckp_dir = target_dir / "pretrained_checkpoints"
        pretrained_ckp_dir.mkdir(parents=True, exist_ok=True)
        wgetgdrive(file_id, pretrained_ckp_dir / "ibot_vit_base_patch16.pth")

    else:
        raise ValueError(f"Unknown data key: {args.data_key}")

    tmp_dir.rmdir()


if __name__ == "__main__":
    main()
