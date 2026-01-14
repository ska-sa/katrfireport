import json
import logging
from datetime import datetime

import holoviews as hv
import katdal
import numpy as np
import panel as pn
import zarr

from dask.diagnostics import ProgressBar
from katdal.flags import NAMES as FLAG_NAMES
from numcodecs import VLenUTF8

from .utils import bl_freq_yticks

hv.extension("bokeh")
pn.extension("katex")
pn.extension("mathjax")

# Inject custom CSS
pn.config.raw_css = [
    """
    .bk-tabs-header {
        background-color: #333;
        color: white;
    }
    .bk-tabs-header .bk-tab {
        padding: 0.5em 1em;
        font-weight: bold;
    }
    .bk-tabs-header .bk-tab.bk-active {
        background-color: #555;
        color: #9cf;
    }
    .bk-panel {
        background: #f4f4f4;
        padding: 10px;
        border-radius: 6px;
    }
    """
]


def extract_raw_flags(katds):
    """
    Extract raw flag data from a katdal dataset without loading visibilities.

    Parameters
    ----------
    katds : katdal.DataSet
        The dataset object.
    autocorr : bool, optional
        If True, select autocorrelation products instead of cross-correlations.

    Returns
    -------
    dask.array.Array
        Lazy-selected raw flags.
    """
    info = katds.source.data.chunk_info["flags"]
    store = katds.source.data.store
    array_name = store.join(info["prefix"], "flags")

    full_raw_flags = store.get_dask_array(
        array_name,
        info["chunks"],
        info["dtype"],
        index=katds.source.data.preselect_index,
        errors=0,
    )

    keep = (katds._time_keep, katds._freq_keep, katds._corrprod_keep)

    selected_raw_flags = katdal.lazy_indexer.DaskLazyIndexer(full_raw_flags, keep)
    return selected_raw_flags.dataset


def process_autocorr(katds, pols, zarr_path):
    """
    Process autocorrelation RFI stats (per antenna, per flag bit)
    and store them under /autocorr in the existing Zarr store.
    """
    flag_names = FLAG_NAMES
    zarr_root = zarr.open(zarr_path, mode="a")

    auto_root = zarr_root.require_group("autocorr")
    auto_root.attrs["description"] = "Autocorrelation RFI flag stats"

    # Save antenna names once
    auto_root.attrs.setdefault("antenna_names", [ant.name for ant in katds.ants])

    for pol in pols:
        logging.info(f"Processing autocorr pol: {pol}")
        katds.select(corrprods="auto", scans="track", pol=pol)
        scan_list = list(katds.scans())
        # Initialize All flags scan-level antenna means
        n_scans = len(scan_list)
        ants = katds.ants
        scan_ant_mean = {ant.name: np.zeros(n_scans, dtype="f4") for ant in ants}
        scan_ant_mean["scan_timestamps"] = np.zeros(n_scans, dtype="f8")
        scan_ant_mean["scan_targets"] = np.empty(n_scans, dtype=object)
        timestamps = katds.timestamps

        n_times, n_freqs, n_ants = katds.shape
        pol_group = auto_root.require_group(pol)

        freq_time_accum = {
            name: np.zeros((n_times, n_ants, n_freqs), dtype="f4")
            for name in flag_names
            if name != "ALL"
        }
        freq_time_all = np.zeros((n_times, n_ants, n_freqs), dtype="f4")

        start_idx = 0
        for scan_idx, (idx, state, target) in enumerate(katds.scans()):
            logging.info(f"Processing scan: {scan_idx+1} of {len(scan_list)}")
            scan_flags = extract_raw_flags(katds).compute().astype(np.uint8)
            n_scan_times = scan_flags.shape[0]
            end_idx = start_idx + n_scan_times

            for j, name in enumerate(flag_names):
                if name == "ALL":
                    continue
                mask = (scan_flags & (1 << j)) > 0
                freq_time_accum[name][start_idx:end_idx] = mask.transpose(
                    0, 2, 1
                ).astype("f4")

            freq_time_all[start_idx:end_idx] = (
                (scan_flags > 0).transpose(0, 2, 1).astype("f4")
            )

            # ---------- All flags scan-level antenna means ----------
            mask_all = scan_flags > 0
            per_ant_mean = mask_all.mean(axis=(0, 1))

            scan_times = timestamps[start_idx:end_idx]
            scan_ts = scan_times.mean()  # midpoint UNIX time

            scan_ant_mean["scan_timestamps"][scan_idx] = scan_ts
            scan_ant_mean["scan_targets"][scan_idx] = target.name

            for ant_idx, ant in enumerate(ants):
                scan_ant_mean[ant.name][scan_idx] = per_ant_mean[ant_idx]

            start_idx = end_idx

        # ---- save freq_time per antenna stats ----
        for name, data in freq_time_accum.items():
            grp = pol_group.require_group(name)
            grp.create_dataset(
                "freq_time",
                data=data,
                chunks=(1000, n_ants, n_freqs),
                dtype="f4",
                overwrite=True,
            )

        all_grp = pol_group.require_group("ALL")
        all_grp.create_dataset(
            "freq_time",
            data=freq_time_all,
            chunks=(1000, n_ants, n_freqs),
            dtype="f4",
            overwrite=True,
        )
        # Save scan-level antenna stats
        scan_grp = pol_group.require_group("scan_ant_mean")
        scan_grp.attrs["description"] = (
            "Mean autocorr flag fraction per scan per antenna"
        )
        for key, value in scan_ant_mean.items():
            if key == "scan_targets":
                scan_grp.create_dataset(
                    key,
                    data=value.astype(str),
                    object_codec=VLenUTF8(),
                    overwrite=True,
                )
            else:
                scan_grp.create_dataset(
                    key,
                    data=value,
                    overwrite=True,
                )


def process_crosscorr(katds, pols, zarr_path):
    """
    Process cross-correlation RFI statistics and store them under
    /crosscorr in the Zarr hierarchy.
    """
    zarr_root = zarr.open(zarr_path, mode="a")

    cross_root = zarr_root.require_group("crosscorr")
    cross_root.attrs["description"] = "Cross-correlation RFI flag stats"
    flag_names = list(FLAG_NAMES)
    cross_root.attrs["flag_names"] = flag_names

    for pol in pols:
        logging.info(f"Processing crosscorr pol: {pol}")
        katds.select(corrprods="cross", scans="track", pol=pol)

        pol_group = cross_root.require_group(pol)
        _process_single_pol(katds, pol_group, pol, flag_names)

    # ---- metadata saved once ----
    bl_idx, bl_lengths, bl_names = bl_freq_yticks(katds)

    cross_root.attrs.update(
        {
            "utc_times": katds.timestamps[:].tolist(),
            "frequency(MHz)": (katds.freqs / 1e6).tolist(),
            "baseline_index_sorted": bl_idx.tolist(),
            "baseline_lengths_sorted": bl_lengths.tolist(),
            "baseline_names_sorted": bl_names.tolist(),
            "capture_block_id": katds.name.split("_")[0],
            "targets": [
                katds.sensor.get("Observation/target")[i].name
                for i in range(katds.shape[0])
            ],
        }
    )


def _process_single_pol(katds, pol_group, pol_label, flag_names):
    n_times, n_freqs, n_baselines = katds.shape

    freq_time_all = np.zeros((n_times, n_freqs), dtype="f4")
    bl_freq_all = np.zeros((n_baselines, n_freqs), dtype="f4")

    freq_time_accum = {
        name: np.zeros((n_times, n_freqs), dtype="f4") for name in flag_names
    }
    bl_freq_accum = {
        name: np.zeros((n_baselines, n_freqs), dtype="f4") for name in flag_names
    }

    start_idx = 0
    scan_list = [scan for scan in katds.scans()]

    for i, (idx, state, target) in enumerate(katds.scans()):
        logging.info(f"Processing scan: {i+1} of {len(scan_list)}")
        with ProgressBar():
            scan_flags = extract_raw_flags(katds).compute().astype(np.uint8)
        n_scan_times = scan_flags.shape[0]
        end_idx = start_idx + n_scan_times

        for j, name in enumerate(flag_names):
            mask = (scan_flags & (1 << j)) > 0
            freq_time_accum[name][start_idx:end_idx, :] = mask.mean(axis=2)
            bl_freq_accum[name] += (mask.sum(axis=0) / n_times).T

        mask_all = scan_flags > 0
        freq_time_all[start_idx:end_idx, :] = mask_all.mean(axis=2)
        bl_freq_all += (mask_all.sum(axis=0) / n_times).T

        start_idx = end_idx

    # Save datasets
    for name in flag_names:
        group = pol_group.require_group(name)
        group.create_dataset(
            "freq_time", data=freq_time_accum[name], chunks=(1000, n_freqs), dtype="f4"
        )
        group.create_dataset(
            "bl_freq",
            data=bl_freq_accum[name],
            chunks=(n_baselines, n_freqs),
            dtype="f4",
        )

    all_group = pol_group.require_group("ALL")
    all_group.create_dataset(
        "freq_time", data=freq_time_all, chunks=(1000, n_freqs), dtype="f4"
    )
    all_group.create_dataset(
        "bl_freq", data=bl_freq_all, chunks=(n_baselines, n_freqs), dtype="f4"
    )


def write_metadata(katds: katdal.DataSet, filename: str) -> None:
    obs_params = katds.obs_params
    targets = [scan[2] for scan in katds.scans()]
    metadata = {
        "ProductType": {
            "ProductTypeName": "MeerKATReductionProduct",
            "ReductionName": "RFIReport",
        },
        "CaptureBlockId": katds.source.capture_block_id,
        "ScheduleBlockIdCode": obs_params.get("sb_id_code", "UNKNOWN"),
        "Description": obs_params.get("description", "UNKNOWN") + ": ",
        "ProposalId": obs_params.get("proposal_id", "UNKNOWN"),
        "Observer": obs_params.get("observer", "UNKNOWN"),
        # Solr doesn't accept +00:00, only Z, so we can't just format a timezone-aware value
        "StartTime": datetime.utcnow().isoformat() + "Z",
        "Bandwidth": katds.channel_width * katds.freqs.shape[0],
        "ChannelWidth": katds.channel_width,
        "NumFreqChannels": katds.freqs.shape[0],
        "RightAscension": [str(target.radec()[0]) for target in targets],
        "Declination": [str(target.radec()[1]) for target in targets],
        # JSON schema limits format to fixed-point with at most 10 decimal places
        "DecRa": [
            ",".join("{:.10f}".format(a) for a in np.rad2deg(target.radec())[::-1])
            for target in targets
        ],
        "Targets": [target.name for target in targets],
        "KatpointTargets": [target.description for target in targets],
    }
    with open(filename, "w") as f:
        json.dump(metadata, f, allow_nan=False, indent=2)
