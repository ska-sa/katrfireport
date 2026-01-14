from katdal.flags import NAMES as FLAG_NAMES

import holoviews as hv
import numpy as np
import panel as pn
import xarray as xr
import zarr
from holoviews.operation.datashader import rasterize


class RFIReportBase:
    def __init__(self, zarr_store):
        self.zarr = zarr_store
        self.cross_root = zarr_store["crosscorr"]

        # ---- metadata ----
        self.freqs = np.asarray(self.cross_root.attrs["frequency(MHz)"])
        self.timestamps = np.asarray(self.cross_root.attrs["utc_times"])
        self.targets = self.cross_root.attrs["targets"]

        self.baseline_lengths = self.cross_root.attrs.get(
            "baseline_lengths_sorted", None
        )
        self.baseline_names = self.cross_root.attrs.get("baseline_names_sorted", None)

    def make_dynamic_plot(self, da, ydim, ylabel, title, ylabels, static=False):
        range_stream = hv.streams.RangeXY()
        base_img = hv.Image(da, kdims=["frequency", ydim])
        raster_img = rasterize(base_img, dynamic=False)

        def dyn_plot(x_range=None, y_range=None):
            yticks = None
            if y_range is not None:
                y_vals = da[ydim].values
                mask = (y_vals >= y_range[0]) & (y_vals <= y_range[1])
                visible = y_vals[mask]
                if len(visible) > 0:
                    step = max(1, len(visible) // 8)
                    yticks = [
                        (v, ylabels[int(v)])
                        for v in visible[::step]
                        if int(v) < len(ylabels)
                    ]

            return raster_img.opts(
                cmap="Viridis",
                colorbar=True,
                width=1200,
                height=500,
                xlabel="Frequency (MHz)",
                ylabel=ylabel,
                yticks=yticks,
                title=title,
                tools=["tap"],
            )

        if static:
            return dyn_plot(y_range=(da[ydim].values.min(), da[ydim].values.max()))

        dmap = hv.DynamicMap(dyn_plot, streams=[range_stream])
        return dmap * raster_img


class CrossCorrView(RFIReportBase):
    """
    Cross-correlation RFI visualisation.

    Produces:
    - Frequency–Time plots (HH & VV)
    - Baseline–Frequency plots (HH & VV)
    """

    def __init__(self, zarr_store):
        super().__init__(zarr_store)

    @property
    def description(self):
        return pn.pane.HTML(
            """
            <div style="font-size:16px; padding:10px;">
              <h3>Cross-Correlation RFI Statistics</h3>
              <p>
                This tab summarises RFI statistics derived from
                <b>cross-correlated visibilities</b>.
              </p>
            </div>
            """,
            sizing_mode="stretch_width",
            margin=(10, 10),
        )

    # --------------------------------
    # Plot construction for one flag
    # -------------------------------
    def dual_pol_view(self, flag_name):
        root = self.cross_root
        bl_idx = np.asarray(root.attrs["baseline_index_sorted"])

        # -------- freq–time DataArray --------
        def ft_da(pol):
            data = root[f"{pol}/{flag_name}/freq_time"][:]
            return xr.DataArray(
                data,
                dims=("time", "frequency"),
                coords={
                    "time": np.arange(data.shape[0]),
                    "frequency": self.freqs,
                },
                name="freq_time",
            )

        # -------- baseline–frequency DataArray --------
        def bf_da(pol):
            data = root[f"{pol}/{flag_name}/bl_freq"][:][bl_idx, :]
            return xr.DataArray(
                data,
                dims=("baseline", "frequency"),
                coords={
                    "baseline": np.arange(len(bl_idx)),
                    "frequency": self.freqs,
                },
                name="bl_freq",
            )

        # -------- plots --------
        ft_hh = self.make_dynamic_plot(
            ft_da("HH"),
            ydim="time",
            ylabel="Target",
            title=f"{flag_name} HH",
            ylabels=self.targets,
        )

        ft_vv = self.make_dynamic_plot(
            ft_da("VV"),
            ydim="time",
            ylabel="",
            title=f"{flag_name} VV",
            ylabels=self.targets,
        )

        bf_hh = self.make_dynamic_plot(
            bf_da("HH"),
            ydim="baseline",
            ylabel="Baseline length (m)",
            title=f"{flag_name} HH",
            ylabels=self.baseline_lengths,
        )

        bf_vv = self.make_dynamic_plot(
            bf_da("VV"),
            ydim="baseline",
            ylabel="",
            title=f"{flag_name} VV",
            ylabels=self.baseline_lengths,
        )

        ft_row = pn.Column(ft_hh, ft_vv, sizing_mode="stretch_width")
        bf_row = pn.Column(bf_hh, bf_vv, sizing_mode="stretch_width")

        return ft_row, bf_row

    # -----------------------------
    # Build the CrossCorr tab
    # -----------------------------
    def build_tab(self):
        flag_names = list(self.cross_root.attrs["flag_names"])
        if "ALL" not in flag_names:
            flag_names.append("ALL")

        freq_time_tabs = pn.Tabs(
            sizing_mode="stretch_width",
            styles={"background": "#fafafa"},
        )

        bl_freq_tabs = pn.Tabs(
            sizing_mode="stretch_width",
            styles={"background": "#fafafa"},
        )

        for flag in flag_names:
            ft_row, bf_row = self.dual_pol_view(flag)

            freq_time_tabs.append(
                (
                    flag,
                    pn.Column(
                        ft_row,
                        sizing_mode="stretch_width",
                        styles={
                            "background": "#fafafa",
                            "padding": "10px",
                            "border": "1px solid #ddd",
                            "border-radius": "5px",
                        },
                    ),
                )
            )

            bl_freq_tabs.append(
                (
                    flag,
                    pn.Column(
                        bf_row,
                        sizing_mode="stretch_width",
                        styles={
                            "background": "#fafafa",
                            "padding": "10px",
                            "border": "1px solid #ddd",
                            "border-radius": "5px",
                        },
                    ),
                )
            )

        return pn.Column(
            self.description,
            pn.Tabs(
                ("📊 Frequency-Time", freq_time_tabs),
                ("📈 Baseline-Frequency", bl_freq_tabs),
                tabs_location="above",
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_width",
        )


class AutoCorrView(RFIReportBase):
    def __init__(self, zarr_store, pol="HH"):
        super().__init__(zarr_store)
        self.auto_root = zarr_store["autocorr"]
        self.pol = pol
        self.antennas = self.auto_root.attrs["antenna_names"]

    @property
    def description(self):
        return pn.pane.HTML(
            """
            <div style="font-size:16px; padding:10px;">
              <h3>Auto-Correlation RFI Statistics</h3>

              <p>
                This tab summarises RFI statistics derived from
                <b>auto-correlated visibilities</b>.
              </p>

              <p>
                You can select an antenna of interest and see the
                contributions of different flag bits.
              </p>
            </div>
            """,
            sizing_mode="stretch_width",
            margin=(10, 10),
        )

    def antenna_freq_panel(self):
        flag_types = list(FLAG_NAMES)
        if "ALL" not in flag_types:
            flag_types.append("ALL")
        da_flags = {
            flag: xr.DataArray(
                self.auto_root[f"{self.pol}/{flag}/freq_time"][:],
                dims=("time", "antenna", "frequency"),
                coords={
                    "time": np.arange(
                        self.auto_root[f"{self.pol}/{flag}/freq_time"].shape[0]
                    ),
                    "antenna": np.arange(
                        self.auto_root[f"{self.pol}/{flag}/freq_time"].shape[1]
                    ),
                    "frequency": self.freqs,
                },
            )
            for flag in flag_types
        }

        antenna_selector = pn.widgets.Select(
            name="Antenna",
            options={name: i for i, name in enumerate(self.antennas)},
            value=0,
            width=250,
        )

        def make_plot(flag, antenna_idx):
            da = da_flags[flag].isel(antenna=antenna_idx)
            return self.make_dynamic_plot(
                da,
                ydim="time",
                ylabel="Target",
                title=f"{flag} – Antenna {self.antennas[antenna_idx]}",
                ylabels=self.targets,
            )

        # Create reactive bound plots
        plots = [
            pn.bind(make_plot, flag, antenna_selector.param.value)
            for flag in flag_types
        ]

        reactive_plots = [pn.panel(p, linked_axes=True) for p in plots]

        plots_card = pn.Column(
            pn.Row(pn.Spacer(width=10), antenna_selector),
            *reactive_plots,
            sizing_mode="stretch_width",
        )

        return pn.Column(
            plots_card,
            sizing_mode="stretch_width",
        )

    def dashboard_tab(self):
        return pn.Column(
            self.description,
            pn.Tabs(
                ("Frequency-Time", self.antenna_freq_panel()),
                tabs_location="above",
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_width",
        )


class RFIDashboard:
    def __init__(self, zarr_path):
        self.zarr = zarr.open(zarr_path, mode="r")
        self.cross_view = CrossCorrView(self.zarr)
        self.auto_view = AutoCorrView(self.zarr)

        self.obs_cbid = self.zarr["crosscorr"].attrs["capture_block_id"]
        self.flag_names = self.zarr["crosscorr"].attrs["flag_names"]

    def build(self):
        header = pn.pane.HTML(
            f"<h1>MeerKAT RFI Report – {self.obs_cbid}</h1>",
            sizing_mode="stretch_width",
        )

        cross_tab = self.cross_view.build_tab()
        auto_tab = self.auto_view.dashboard_tab()

        return pn.Column(
            header,
            pn.Tabs(
                ("Cross Correlation", cross_tab),
                ("Auto Correlation", auto_tab),
                tabs_location="left",
            ),
            sizing_mode="stretch_both",
        )
