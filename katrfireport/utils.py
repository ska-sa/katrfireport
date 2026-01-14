import io
import os
import zipfile

import numpy as np


def bl_freq_yticks(katds):
    """
    Generate y-axis tick labels for baseline-frequency plots.

    This function computes baseline lengths (in meters) for all correlation products
    in the given katdal dataset, sorts them by length, and generates a mapping of
    baseline indices to tick labels for y-axis labeling in baseline-frequency plots.

    Parameters
    ----------
    katds : katdal.DataSet
        The katdal dataset.

    Returns
    -------
    bl_idx : numpy.ndarray
        Array of indices that sort the baselines by increasing length.

    baseline_lengths_sorted : numpy.ndarray
        Sorted array of baseline lengths (in meters).

    baseline_names_sorted : numpy.ndarray
        Sorted array of baseline antenna pair names (e.g., 'm010m012').
    """

    antenna_positions = {ant.name: ant for ant in katds.ants}
    baseline_lengths = []
    baseline_names = []

    for ant1, ant2 in katds.corr_products:
        pos1 = np.array(antenna_positions[ant1[:-1]].position_enu)
        pos2 = np.array(antenna_positions[ant2[:-1]].position_enu)
        length = np.linalg.norm(pos1 - pos2)
        baseline_lengths.append(length)
        baseline_names.append(f"{ant1[:-1]}{ant2[:-1]}")

    baseline_lengths = np.array(baseline_lengths)
    bl_idx = np.argsort(baseline_lengths)
    baseline_lengths_sorted = baseline_lengths[bl_idx].astype(int)
    baseline_names_sorted = np.array(baseline_names)[bl_idx]

    return bl_idx.astype(int), baseline_lengths_sorted, baseline_names_sorted


def zip_zarr_dir(zarr_path):
    """Zip the Zarr directory into memory and return as BytesIO."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(zarr_path):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, start=zarr_path)
                zf.write(full_path, arcname=rel_path)
    buffer.seek(0)
    return buffer
