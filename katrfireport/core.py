import json
import logging

import holoviews as hv
import katdal
import numpy as np
import panel as pn
import zarr
from katdal.flags import NAMES as FLAG_NAMES

hv.extension('bokeh')
pn.extension('katex')
pn.extension('mathjax')

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

    Returns:
        dask.array.Array: Lazy-selected raw flags.
    """
    info = katds.source.data.chunk_info['flags']
    store = katds.source.data.store
    array_name = store.join(info['prefix'], 'flags')
    full_raw_flags = store.get_dask_array(
        array_name,
        info['chunks'],
        info['dtype'],
        index=katds.source.data.preselect_index,
        errors=0,
    )
    keep = (katds._time_keep, katds._freq_keep, katds._corrprod_keep)
    selected_raw_flags = katdal.lazy_indexer.DaskLazyIndexer(full_raw_flags, keep)
    return selected_raw_flags.dataset


def process_dual_pol(katds, pols, zarr_path):
    """
    Process dual-polarisation RFI statistics and save to a Zarr store.

    Args:
        katds: katdal dataset object.
        pols: List of polarisation labels (e.g. ['HH', 'VV']).
        zarr_path: Path where the Zarr store will be saved.
    """
    zarr_root = zarr.open(zarr_path, mode='w')
    zarr_root.attrs['description'] = 'RFI Flags occupancy stats (dual pol loop)'
    flag_names = FLAG_NAMES
    # Save metadata from first pol
    zarr_root.create_dataset('utc_times', data=katds.timestamps[:], dtype='f8')
    zarr_root.create_dataset('frequency(MHz)', data=katds.freqs / 1e6, dtype='f8')

    all_stats = []

    for pol in pols:
        logging.info(f"🔹 Processing pol: {pol}")
        katds.select(corrprods='cross', scans='track', pol=pol)

        stats = _process_single_pol(katds, zarr_root, pol, flag_names)
        all_stats.extend(stats)

    # Save combined per-scan stats
    zarr_root.attrs['per_scan_stats'] = json.dumps(all_stats)


def _process_single_pol(katds, zarr_root, pol_label, flag_names):
    n_times, n_freqs, n_baselines = katds.shape
    pol_group = zarr_root.require_group(pol_label)

    freq_time_all = np.zeros((n_times, n_freqs), dtype='f4')
    bl_freq_all = np.zeros((n_baselines, n_freqs), dtype='f4')

    freq_time_accum = {name: np.zeros((n_times, n_freqs), dtype='f4') for name in flag_names}
    bl_freq_accum = {name: np.zeros((n_baselines, n_freqs), dtype='f4') for name in flag_names}

    scan_stats = []
    start_idx = 0
    scan_list = [scan for scan in katds.scans()]

    for i, (idx, state, target) in enumerate(katds.scans()):
        logging.info(f"Processing scan: {i+1} of {len(scan_list)}")
        scan_flags = extract_raw_flags(katds).compute().astype(np.uint8)
        n_scan_times = scan_flags.shape[0]
        total_samples = np.prod(scan_flags.shape)
        end_idx = start_idx + n_scan_times
        scan_duration = katds.timestamps[-1] - katds.timestamps[0]

        for j, name in enumerate(flag_names):
            mask = (scan_flags & (1 << j)) > 0
            freq_time_accum[name][start_idx:end_idx, :] = mask.mean(axis=2)
            bl_freq_accum[name] += (mask.sum(axis=0) / n_times).T
            flagged = mask.sum()

            frac_flagged = float(flagged / total_samples) if total_samples > 0 else 0.0

            scan_stats.append({
                'pol': pol_label,
                'scan_index': i,
                'target_name': target.name,
                'state': str(state),
                'flag_type': name,
                'start_idx': int(start_idx),
                'end_idx': int(end_idx),
                'fraction_flagged': frac_flagged,
                'scan_duration': scan_duration,
                'num_samples': int(total_samples),
            })

        mask_all = (scan_flags > 0)
        freq_time_all[start_idx:end_idx, :] = mask_all.mean(axis=2)
        bl_freq_all += (mask_all.sum(axis=0) / n_times).T

        flagged_all = mask_all.sum()
        frac_flagged_all = float(flagged_all / total_samples) if total_samples > 0 else 0.0

        scan_stats.append({
            'pol': pol_label,
            'scan_index': i,
            'target_name': target.name,
            'state': str(state),
            'flag_type': 'ALL',
            'start_idx': int(start_idx),
            'end_idx': int(end_idx),
            'fraction_flagged': frac_flagged_all,
            'scan_duration': scan_duration,
            'num_samples': int(total_samples),
        })

        start_idx = end_idx

    # Save datasets
    for name in flag_names:
        group = pol_group.require_group(name)
        group.create_dataset('freq_time', data=freq_time_accum[name], chunks=(1000, n_freqs),
                             dtype='f4')
        group.create_dataset('bl_freq', data=bl_freq_accum[name], chunks=(n_baselines, n_freqs),
                             dtype='f4')

    all_group = pol_group.require_group('ALL')
    all_group.create_dataset('freq_time', data=freq_time_all, chunks=(1000, n_freqs),
                             dtype='f4')
    all_group.create_dataset('bl_freq', data=bl_freq_all, chunks=(n_baselines, n_freqs),
                             dtype='f4')

    return scan_stats
