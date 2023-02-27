import bioimageio.core
import h5py
import mrcfile
import numpy as np

from scipy.ndimage import distance_transform_edt
from skimage.measure import label
from skimage.segmentation import watershed
from xarray import DataArray


def run_prediction(raw, model_path, bounding_box):
    """Run prediction with the model. The model predicts foreground and boundary
    probabilities for the vesicles. It is saved in the bioimageio.io model format,
    and we use the bioimagio.core library for running tiled prediction.
    See https://bioimage.io/#/ for details."""

    model = bioimageio.core.load_resource_description(model_path)

    # The tile size and halo (cut from the tile to decrease tiling artifacts)
    tiling = {
        "tile": {"x": 256, "y": 256, "z": 32},
        "halo": {"x": 32, "y": 32, "z": 8},
    }

    # This will automatically use a GPU if available
    with bioimageio.core.create_prediction_pipeline(model) as pp:
        # The network expects 5D input (including batch and channel dimension).
        # With dimension annotations (provided by xarray).
        input_ = DataArray(raw[None, None], dims=tuple("bczyx"))

        # Run tiled prediction.
        prediction = bioimageio.core.predict_with_tiling(pp, input_, tiling=tiling, verbose=True)
        prediction = prediction[0].values[0]

    return prediction


def run_segmentation(prediction, result_path, min_size=500):
    """Run vesicle instance segmentation based on the model predictions."""

    # Get the foreground and boundary prediction (channel 0 and 1 of the predictions).
    foreground, boundaries = prediction

    # Run instance segmentation via connected components on boundaries subtracted
    # from foreground probabilities. Then expand the instances to cover the
    # full vesicles via a seeded watershed on a distance transform,
    # using the thresholded sum of foreground predictions and boundaries as mask.
    seeds = label((foreground - boundaries) > 0.5)
    dist = distance_transform_edt(seeds == 0)
    mask = (foreground + boundaries) > 0.5
    seg = watershed(dist, seeds, mask=mask)

    # Apply a size filter to get rid of small instances, that are most likely
    # false positives.
    ids, sizes = np.unique(seg, return_counts=True)
    filter_ids = ids[sizes < min_size]
    seg[np.isin(seg, filter_ids)] = 0

    # Save the segmentation result to HDF5. (This will over-ride previous contents of the file.)
    with h5py.File(result_path, "w") as f:
        f.create_dataset("segmentation", data=seg, compression="gzip")

    return seg


def main():
    # Paths to the required input files, adapt this to the paths on HPC.
    # - raw_path: Path to the input tomogram
    # - model_path: Path to the model, which is a pytorch 3D U-Net.
    #               It is saved in the bioimage.io format, which contains the pytorch
    #               weights and some additional metadata.
    raw_path = "/home/pape/Work/data/cooper/20230126_TOMO_4CP/00_AUTO_SV_DETECTION/57K_TEM/57K_tomogram.mrc"
    model_path = "./3D_UNet_for_Vesicle_Segmentation.zip"

    # Path where the segmentation result wll be saved.
    result_path = "./segmentation.h5"

    # Bounding box for running segmentation on a smaller sub-volume
    # (to decrease runtime, e.g. in case we don't have a GPU or for testing purposes)
    # bounding_box = np.s_[:32, :256, :256]  # smaller sub-volume
    bounding_box = np.s_[:]  # full volume

    # Load the raw data.
    with mrcfile.open(raw_path, "r") as f:
        raw = f.data[bounding_box]

    print("Run prediction ...")
    prediction = run_prediction(raw, model_path, bounding_box)

    print("Run segmentation ...")
    seg = run_segmentation(prediction, result_path)

    # Visual inspection of the result, will be skipped if napari is not available.
    try:
        import napari
        v = napari.Viewer()
        v.add_image(raw)
        v.add_image(prediction)
        v.add_labels(seg)
        napari.run()
    except ImportError:
        print("Napari is not available, skipping the visual inspection.")


if __name__ == "__main__":
    main()
