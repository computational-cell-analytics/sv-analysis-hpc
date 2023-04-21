import argparse
import os
from glob import glob

import numpy as np
import torch_em
from elf.io import open_file
from torch_em.model import AnisotropicUNet
from torch_em.util.modelzoo import export_bioimageio_model
from tqdm import tqdm


def _get_supervised_rois(label_paths, label_key):
    rois = []

    for path in tqdm(label_paths, desc="Compute ROIs with labels"):
        with open_file(path, "r") as f:
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
        num_workers=4, shuffle=True,
    )
    return loader


def export_model(args):
    output_folder = "." if args.output is None else args.output
    checkpoint = os.path.join(output_folder, "checkpoints", "unet_source")
    assert os.path.exists(checkpoint), checkpoint

    export_folder = os.path.join(output_folder, "exported_model")

    input_path = glob(os.path.join(args.input_folder, "*.mrc"))[0]
    halo = [8, 128, 128]
    with open_file(input_path, "r") as f:
        shape = f["data"].shape
        center = [int(sh // 2) for sh in shape]
        bb = tuple(slice(ce - ha, ce + ha) for ce, ha in zip(center, halo))
        input_data = f["data"][bb]

    # TODO derive more metadata
    export_bioimageio_model(
        checkpoint, export_folder, input_data, name="UNet3DSVSeg",
    )


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

    # run training
    trainer.fit(iterations=args.n_iterations)

    # export the model
    export_model(args)


def main():
    parser = argparse.ArgumentParser(description="Train a 3D U-Net for vesicle segmentation.")
    parser.add_argument("-i", "--input_folder", required=True,
                        help="Path to the folder with tomograms (in mrc format).")
    parser.add_argument("-l", "--label_folder", required=True,
                        help="Path to the folder with label data (in hdf5 format).")
    parser.add_argument("-o", "--output",
                        help="Folder where the model checkpoint and logs will be saved. After training the model ")
    parser.add_argument("-b", "--batch_size", default=1, type=int)
    parser.add_argument("-n", "--n_iterations", default=50000, type=int)
    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
