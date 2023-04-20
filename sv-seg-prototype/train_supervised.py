import argparse
import os
from glob import glob

import h5py
import numpy as np
import torch_em
from torch_em.model import AnisotropicUNet
from tqdm import tqdm


def _get_supervised_rois(label_paths, label_key):
    rois = []

    for path in tqdm(label_paths, desc="Compute ROIs with labels"):
        with h5py.File(path, "r") as f:
            labels = f[label_key][:]
        mask = np.where(labels != 0)
        roi_begin = [int(np.min(ma)) for ma in mask]
        roi_end = [int(np.max(ma) + 1) for ma in mask]
        rois.append(
            tuple(slice(beg, end) for beg, end in zip(roi_begin, roi_end))
        )

    return rois


def _get_paths(args, split):
    raw_paths = glob(os.path.join(args.input_folder, "*.mrc"))
    raw_paths.sort()

    label_paths = glob(os.path.join(args.label_folder, "*.h5"))
    label_paths.sort()
    assert len(raw_paths) == len(label_paths)

    val_fraction = 0.2
    val_len = int(val_fraction * len(raw_paths))

    if split is None:
        pass
    elif split == "train":
        raw_paths, label_paths = raw_paths[:-val_len], label_paths[:-val_len]
    elif split == "val":
        raw_paths, label_paths = raw_paths[-val_len:], label_paths[-val_len:]
    else:
        raise ValueError(f"Invalid split: {split}")

    return raw_paths, label_paths


def get_loader(args, split):
    patch_shape = (24, 256, 256)

    raw_paths, label_paths = _get_paths(args, split)
    raw_key = "data"
    label_key = "labels"

    label_trafo = torch_em.transform.BoundaryTransform(add_binary_target=True)
    rois = _get_supervised_rois(label_paths, label_key)

    loader = torch_em.default_segmentation_loader(
        raw_paths, raw_key, label_paths, label_key,
        batch_size=args.batch_size, patch_shape=patch_shape,
        label_transform=label_trafo, rois=rois, ndim=3, is_seg_dataset=True,
    )
    return loader


def run_training(args):
    model = AnisotropicUNet(in_channels=1, out_channels=2, final_activation="Sigmoid",
                            scale_factors=[[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]])
    train_loader = get_loader(args, "train")
    val_loader = get_loader(args, "val")
    name = "unet_source"
    trainer = torch_em.default_segmentation_trainer(
        name=name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=1e-4,
        mixed_precision=True,
        log_image_interval=100,
        save_root=args.output,
    )
    trainer.fit(iterations=args.n_iterations)


def main():
    parser = argparse.ArgumentParser(description="Train a 3D U-Net for vesicle segmentation.")
    parser.add_argument("-i", "--input_folder", required=True, help="Path to the folder with tomograms")
    parser.add_argument("-l", "--label_folder", required=True, help="Path to the folder with label data")
    parser.add_argument("-o", "--output", help="Folder where the model checkpoint and logs will be saved.")
    parser.add_argument("-b", "--batch_size", default=1, type=int)
    parser.add_argument("-n", "--n_iterations", default=50000, type=int)
    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
