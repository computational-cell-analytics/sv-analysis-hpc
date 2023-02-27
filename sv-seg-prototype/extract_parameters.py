import h5py
import numpy as np
import pandas as pd

from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max


def extract_coordinates_and_radii(seg):
    distances = distance_transform_edt(seg != 0)

    maxima = peak_local_max(distances)
    coordinates = tuple(maxima[:, i] for i in range(maxima.shape[1]))

    seg_ids = seg[coordinates]
    radii = distances[coordinates]

    unique_seg_ids = np.unique(seg_ids)

    coords, rads = [], []
    for seg_id in unique_seg_ids:
        seg_mask = seg_ids == seg_id
        seg_radii = radii[seg_mask]
        seg_max = np.argmax(seg_radii)

        radius = seg_radii[seg_max]
        coord = maxima[seg_mask][seg_max]

        coords.append(coord)
        rads.append(radius)

    return np.array(coords), np.array(rads)


def main():
    # Path where the segmentation was saved.
    result_path = "./segmentation.h5"

    # Path where to save the extracted parameters.
    output_path = "./parameters.csv"

    # Load the segmentation.
    with h5py.File(result_path, "r") as f:
        seg = f["segmentation"][:]

    # Extract the coordinates and radii from the segmentation.
    print("Extract parameters ...")
    coordinates, radii = extract_coordinates_and_radii(seg)

    # Save the extracted parameters as csv.
    table = pd.DataFrame.from_dict(
        {"x": coordinates[:, 2], "y": coordinates[:, 1], "z": coordinates[:, 0], "radius": radii}
    )
    table.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
