import argparse
import os
from concurrent import futures

import h5py
import numpy as np
import pandas as pd

from elf.mesh.io import write_obj
from scipy.ndimage import distance_transform_edt
from skimage.measure import marching_cubes, regionprops
from skimage.transform import resize
from tqdm import tqdm


def extract_coordinates_and_radii(seg):
    props = regionprops(seg)

    def extract(prop):
        seg_id = prop.label
        bb = tuple(slice(bmin, bmax) for bmin, bmax in zip(prop.bbox[:3], prop.bbox[3:]))
        mask = seg[bb] == seg_id

        distances = distance_transform_edt(~mask)
        center_idx = np.argmax(distances)
        radius = distances.ravel()[center_idx]

        offset = prop.bbox[:3]
        center = np.unravel_index(center_idx, distances.shape)
        center = [off + ce for off, ce in zip(offset, center)]

        return seg_id, center, radius

    with futures.ThreadPoolExecutor(8) as tp:
        res = list(tqdm(
            tp.map(extract, props), total=len(props), desc="Extracting vesicle coordinates"
        ))

    coordinates = {re[0]: re[1] for re in res}
    radii = {re[0]: re[2] for re in res}
    return coordinates, radii


def extract_meshes_and_vesicle_coodinates(compartments, vesicles, output_folder, mitos=None):
    os.makedirs(output_folder, exist_ok=True)

    coordinates, radii = extract_coordinates_and_radii(vesicles)

    seg_ids = np.unique(compartments)
    if seg_ids[0] == 0:
        seg_ids = seg_ids[1:]

    print("Resizing compartments ...")
    compartments_resized = resize(
        compartments, vesicles.shape, order=0, anti_aliasing=False, preserve_range=True
    ).astype(compartments.dtype)

    if mitos is not None:
        print("Excluding mitos from compartment volumes ...")
        compartments[mitos != 0] = 0

    scale_factor = np.array([vsh / float(csh) for vsh, csh in zip(vesicles.shape, compartments.shape)])
    for seg_id in tqdm(seg_ids, desc="Extracting meshes"):
        mask = compartments == seg_id
        verts, faces, normals, _ = marching_cubes(mask)
        verts *= scale_factor
        mesh_path = os.path.join(output_folder, f"compartment-{seg_id}.obj")
        write_obj(mesh_path, verts, faces, normals)

        mask = compartments_resized == seg_id
        this_vesicle_ids = np.unique(vesicles[mask])[1:]
        this_coordinates = np.array([coordinates[ves_id] for ves_id in this_vesicle_ids])
        this_radii = np.array([radii[ves_id] for ves_id in this_vesicle_ids])
        table = pd.DataFrame.from_dict(
            {"x": this_coordinates[:, 2], "y": this_coordinates[:, 1],
             "z": this_coordinates[:, 0], "radius": this_radii}
        )
        vesicle_path = os.path.join(output_folder, f"vesicles-{seg_id}.csv")
        table.to_csv(vesicle_path, index=False)


# for visually checking the extracted vesicle coordinates with napari
def check_extracted(vesicles, output_folder):
    from glob import glob
    import napari

    coords = []
    coord_files = glob(os.path.join(output_folder, "*.csv"))
    for ff in coord_files:
        this_coords = pd.read_csv(ff)[["z", "y", "x"]].values
        coords.extend([coord for coord in this_coords])

    v = napari.Viewer()
    v.add_labels(vesicles)
    v.add_points(coords)
    napari.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", required=True,
        help="The path to the segmentaiton result, stored as h5 file with keys 'segmentation/vesicles',"
        "'segmentation/compartments' and 'segmentation/mitochondria' (optional)."
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="The output folder where the meshes for the compartments (one .obj file per compartment),"
        "and the vesicle coordiantes per comportment (one .csv file per compartment) will be saved."
    )
    args = parser.parse_args()

    with h5py.File(args.input, "r") as f:
        vesicles = f["segmentation/vesicles"][:]
        compartments = f["segmentation/compartments"][:]
        if "segmentation/mitochondria" in f:
            mitos = f["segmentation/mitochondria"][:]
        else:
            mitos = None

    extract_meshes_and_vesicle_coodinates(
        compartments, vesicles, mitos=mitos,
        output_folder=args.output
    )
    # check_extracted(vesicles, output_folder)


if __name__ == "__main__":
    main()
