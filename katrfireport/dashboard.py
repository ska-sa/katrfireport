import json
import logging

import colorcet as cc
import holoviews as hv
import numpy as np
import panel as pn
import pandas as pd
import xarray as xr
import zarr
from datetime import datetime
from holoviews.operation.datashader import rasterize

from .utils import zip_zarr_dir


def make_dual_pol_view(zarr_store, flag_name):
    freqs = zarr_store.attrs.get('frequency(MHz)')
    timestamps = zarr_store.attrs.get('utc_times')
    bl_idx = zarr_store.attrs.get('baseline_index_sorted')
    baseline_lengths_sorted = zarr_store.attrs.get('baseline_lengths_sorted')
    baseline_names_sorted = zarr_store.attrs.get('baseline_names_sorted')
    targets = zarr_store.attrs.get('targets')

    # DataArrays
    da_ft_hh = xr.DataArray(
        zarr_store[f"HH/{flag_name}/freq_time"][:],
        coords=[
            (
                'time',
                np.arange(
                    zarr_store[f"HH/{flag_name}/freq_time"].shape[0]
                )
            ),
            ('frequency', freqs),
        ]
    )

    da_ft_vv = xr.DataArray(zarr_store[f"VV/{flag_name}/freq_time"][:],
                            coords=[
                                (
                                    'time',
                                    np.arange(
                                        zarr_store[f"VV/{flag_name}/freq_time"].shape[0]
                                    )
                                ),
                                ('frequency', freqs),
                                ])
    da_bf_hh = xr.DataArray(zarr_store[f"HH/{flag_name}/bl_freq"][:][bl_idx, :],
                            coords=[('baseline', np.arange(len(bl_idx))),
                                    ('frequency', freqs)])
    da_bf_vv = xr.DataArray(zarr_store[f"VV/{flag_name}/bl_freq"][:][bl_idx, :],
                            coords=[('baseline', np.arange(len(bl_idx))),
                                    ('frequency', freqs)])

    # Hover info panels
    ft_info = pn.pane.Markdown("**Freq-Time Hover info:**", width=900)
    bf_info = pn.pane.Markdown("**Baseline-Freq Hover info:**", width=900)

    def make_dynamic_plot(da, ydim, ylabel, title, info_pane, ylabels):
        range_stream = hv.streams.RangeXY()

        base_img = hv.Image(da, kdims=['frequency', ydim])
        raster_img = rasterize(base_img, dynamic=False)

        stream = hv.streams.Tap(source=raster_img)

        def callback(x, y):
            if x is None or y is None:
                info_pane.object = f"**{title} Hover info:**"
                return
            freq_idx = (np.abs(da['frequency'].values - x)).argmin()
            y_idx = (np.abs(da[ydim].values - y)).argmin()
            z_val = da.values[y_idx, freq_idx]

            if ydim == 'time':
                if 0 <= y_idx < len(timestamps):
                    utc_time = datetime.utcfromtimestamp(
                        timestamps[y_idx]).strftime('%Y-%m-%d %H:%M:%S')
                else:
                    utc_time = "Out of range"
                info_pane.object = (
                    f"**{title}**  \n"
                    f"Frequency: {x:.2f} MHz  \n"
                    f"Time (UTC): {utc_time}  \n"
                    f"RFI Fraction: {z_val:.4f}"
                )
            else:
                if 0 <= y_idx < len(ylabels):
                    bl_name = baseline_names_sorted[y_idx]
                    bl_len = baseline_lengths_sorted[y_idx]
                else:
                    bl_name = "Unknown"
                    bl_len = "N/A"
                info_pane.object = (
                    f"**{title}**  \n"
                    f"Frequency: {x:.2f} MHz  \n"
                    f"Baseline: {bl_name}  \n"
                    f"Length: {bl_len:.2f} m  \n"
                    f"RFI Fraction: {z_val:.4f}"
                )

        stream.add_subscriber(lambda **kwargs: callback(kwargs.get('x'), kwargs.get('y')))

        def dyn_plot(x_range=None, y_range=None):
            yticks = None
            if y_range is not None:
                y_vals = da[ydim].values
                mask = (y_vals >= y_range[0]) & (y_vals <= y_range[1])
                visible_idx = y_vals[mask]
                if len(visible_idx) > 0:
                    step = max(1, len(visible_idx) // 8)
                    yticks = [(v, ylabels[int(v)])
                              for v in visible_idx[::step] if int(v) < len(ylabels)]
            return raster_img.opts(
                cmap='Viridis', colorbar=True, width=950, height=500,
                xlabel='Frequency (MHz)', ylabel=ylabel,
                yticks=yticks, title=title, tools=['tap']
            )

        dmap = hv.DynamicMap(dyn_plot, streams=[range_stream])
        overlay = dmap * raster_img
        return overlay

    ft_hh = make_dynamic_plot(da_ft_hh, 'time', 'Target names', f'{flag_name} HH', ft_info,
                              [targets[i] for i in range(len(targets))])
    ft_vv = make_dynamic_plot(da_ft_vv, 'time', '', f'{flag_name} VV', ft_info,
                              [targets[i] for i in range(len(targets))])
    bf_hh = make_dynamic_plot(da_bf_hh, 'baseline', 'Baseline length (m)', f'{flag_name} HH',
                              bf_info, baseline_lengths_sorted)
    bf_vv = make_dynamic_plot(da_bf_vv, 'baseline', '', f'{flag_name} VV', bf_info,
                              baseline_lengths_sorted)

    ft_row = pn.Column(
        pn.Row(ft_hh, ft_vv),
        pn.Row(ft_info, align='center')
    )
    bf_row = pn.Column(
        pn.Row(bf_hh, bf_vv),
        pn.Row(bf_info, align='center')
    )

    return ft_row, bf_row


def create_dual_pol_dashboard(zarr_path, scan_stats=True):
    """
    Create an interactive Panel dashboard for dual-polarization RFI statistics visualization,
    including an extra tab with RFI fraction line plots per scan.

    Parameters:
    - zarr_path: str, path to Zarr data directory
    - scan_stats_df: If True add extra tab with stats per scans.
        If False, skip the extra tab.
    """

    zarr_store = zarr.open(zarr_path, mode='r')
    flag_names = list(zarr_store['HH'].group_keys())
    obs_cbid = zarr_store.attrs.get('capture_block_id')

    header = pn.pane.HTML(
        f"""
        <div style="text-align: center;">
        <h1 style="color:#333; font-weight:bold; font-size: 32px; margin-bottom: 0;">
            MeerKAT RFI REPORT FOR OBSERVATION WITH CBID {obs_cbid}
        </h1>
        </div>
        """,
        margin=(10, 10)
    )

    report_description = pn.pane.Markdown(
        """
        <div style="font-size:16px;">
        This interactive report summarizes radio-frequency interference (RFI) statistics
        detected during the observation. It provides both frequency-time and baseline-frequency
        visualizations for each flag type across HH and VV polarizations.

        **Functionality of this report:**
        - 📊 Interactive frequency-time and baseline-frequency plots for each flag type.
        - 🖱️ Click on (or tap) a point in any plot to display detailed information below the plot:
          frequency, time (UTC), baseline name, baseline length, and RFI fraction.
        - 🔍 Zoom into a section of a plot — all other linked plots will zoom to the same region.
        - 🗂️ Explore different flag types using tabs (each tab corresponds to a different\
        flag type).
        - 📥 Download a ZIP file of the underlying Zarr data for further analysis.
        </div>
        """,
        margin=(10, 5)
    )

    flag_description = {
        "ALL": "All flag types combined.",
        "cal_rfi": "Flags from the calibration pipeline (SDP AOFlagger).",
        "cam": "Flags from the control and monitoring system (e.g. antenna down).",
        "data_lost": "Data missing from the correlator.",
        "ingest_rfi": "High-time-resolution RFI flags from ingest.",
        "postproc": "Flags from failed or invalid calibration solutions.",
        "predicted_rfi": "Satellite RFI predictions (not in use currently).",
        "reserved": "Reserved for future use.",
        "static": " Known persistent RFI channels.",
    }
    scan_stats_description = pn.pane.Markdown(
        """
        <div style="font-size:16px;">
        ### RFI Scan Statistics Summary
        This tab presents interactive plots of raw and weighted RFI fraction flagged per scan,\
        for each flag type.

        - **Raw Fraction Flagged**: The fraction of data flagged within each scan.
        - **Weighted Fraction Flagged**: Average fraction flagged across scans, weighted by\
        scan duration:

        $$
        \\text{Weighted Fraction} = \\frac{\\sum (\\text{fraction flagged} \\times
        \\text{scan duration})}{\\sum \\text{scan duration}}
        $$

        - **Flag Type Definitions**:
        - `ALL`: All flag types combined.
        - `cal_rfi`: Flags from the calibration pipeline (SDP AOFlagger).
        - `cam`: Flags from the control and monitoring system (e.g. antenna down).
        - `data_lost`: Data missing from the correlator.
        - `ingest_rfi`: High-time-resolution RFI flags from ingest.
        </div>
        """,
        margin=(10, 5)
    )

    freq_time_tabset = pn.Tabs(
        styles={'background': '#fafafa'}, margin=(10, 10), sizing_mode='stretch_both'
    )
    bl_freq_tabset = pn.Tabs(
        styles={'background': '#fafafa'}, margin=(10, 10), sizing_mode='stretch_both'
    )

    for flag in flag_names:
        ft_row, bf_row = make_dual_pol_view(zarr_store, flag)

        description = flag_description.get(flag, "No description available for this flag type.")
        info_bar = pn.pane.Markdown(
            f"ℹ️ **{flag}**: {description}",
            styles={'color': '#555', 'background': '#f9f9f9', 'padding': '5px',
                    'border': '1px solid #ccc'},
            margin=(5, 5)
        )

        freq_time_tabset.append(
            (
                f"{flag}",
                pn.Column(
                    info_bar,
                    ft_row,
                    margin=5,
                    styles={'border': '1px solid #ddd', 'border-radius': '5px'},
                ),
            )
        )

        bl_freq_tabset.append(
            (
                f"{flag}",
                pn.Column(
                    info_bar,
                    bf_row,
                    margin=5,
                    styles={'border': '1px solid #ddd', 'border-radius': '5px'},
                ),
            )
        )

    # Add download button
    download_button = pn.widgets.FileDownload(
        callback=lambda: zip_zarr_dir(zarr_path),
        filename='zarr_data.zip',
        label='📥 Download RFI Statistics Zarr File',
    )

    # Group freq + bl freq into their own Panel column with the description
    freq_bl_panel = pn.Column(
        report_description,
        pn.Tabs(
            ("📊 Frequency-Time", freq_time_tabset),
            ("📈 Baseline-Frequency", bl_freq_tabset),
            tabs_location='above',  # or 'left' if you prefer
            styles={'background': '#fafafa'},
            sizing_mode='stretch_both'
        )
    )

    # Create main tabs container
    main_tabs_items = [
        ("📊 RFI Waterfall Plots", freq_bl_panel),]

    # Add extra Scan Stats tab if scan_stats is True
    if scan_stats:
        timestamps = zarr_store.attrs.get('utc_times')
        scan_stats_json = zarr_store.attrs.get('per_scan_stats', None)
        if scan_stats_json is None:
            logging.warning("Warning: No per_scan_stats attribute found in zarr store;\
            skipping Scan Stats tab.")
            pass
        else:
            scan_stats_list = json.loads(scan_stats_json)
            df_scan_stats = pd.DataFrame(scan_stats_list)
            df_scan_stats['utc_time'] = df_scan_stats['start_idx'].apply(
                lambda i: datetime.utcfromtimestamp(
                    timestamps[i]).strftime('%Y-%m-%d %H:%M:%S'))
            # Exclude flag types that are always zeros
            df_clean_scan = df_scan_stats[~df_scan_stats['flag_type'].isin([
                'reserved0', 'postproc', 'static', 'predicted_rfi'])]
            line_hh = make_line_plot(df_clean_scan, 'HH')
            line_vv = make_line_plot(df_clean_scan, 'VV')
            scan_stats_pane = pn.Column(
                    scan_stats_description,
                    pn.Row(
                        pn.pane.HoloViews(line_hh),
                        pn.pane.HoloViews(line_vv),
                        margin=10
                    )
                )

            main_tabs_items.append(
                ("📉RFI Scan Summary", scan_stats_pane)
            )

        main_tabs = pn.Tabs(
            *main_tabs_items,
            tabs_location='left',
            styles={'background': '#eee'},
            sizing_mode='stretch_both',
        )
    return pn.Column(header, download_button, main_tabs)


def make_line_plot(df, pol):
    df_pol = df[df['pol'] == pol].copy()
    overlays = []
    color_list = cc.glasbey_dark  # Distinct colors
    flag_types = df_pol['flag_type'].unique()

    for i, flag_type in enumerate(flag_types):
        df_flag = df_pol[df_pol['flag_type'] == flag_type].copy()
        # Compute weighted average
        weighted_frac = (
            (df_flag['fraction_flagged'] * df_flag['scan_duration']).sum() /
            df_flag['scan_duration'].sum()
        )
        # Add weighted_frac so it appears in hover
        df_flag['weighted_fraction_flagged'] = weighted_frac
        # Common color
        color = color_list[i % len(color_list)]
        # Update label to include weighted avg
        label = f"{flag_type} (weighted avg: {weighted_frac:.3f})"
        # Raw fraction flagged curve
        curve = hv.Curve(
            df_flag,
            kdims='scan_index',
            vdims=[
                ('fraction_flagged', 'Raw Fraction Flagged'),
                ('weighted_fraction_flagged', 'Weighted Fraction Flagged'),
                ('target_name', 'Target'),
                ('utc_time', 'UTC Time')
            ],
            label=label
        ).opts(
            tools=['hover'],
            color=color,
            width=1000,
            height=600,
            padding=0.1,
            show_legend=True
        )
        # Markers at scan points
        points = hv.Scatter(
            df_flag,
            kdims='scan_index',
            vdims=[
                ('fraction_flagged', 'Raw Fraction Flagged'),
                ('weighted_fraction_flagged', 'Weighted Fraction Flagged'),
                ('target_name', 'Target'),
                ('utc_time', 'UTC Time')
            ]
        ).opts(
            color=color,
            marker='circle',
            size=6,
            tools=['hover'],
            show_legend=False
        )
        overlays.append(curve * points)

    return hv.Overlay(overlays).opts(
        title=f"RFI Fraction Flagged with Weighted Avg - {pol}",
        legend_position='right'
    )
