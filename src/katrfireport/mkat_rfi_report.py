import logging
from typing import Dict, List
import json
import os

import katdal
import numpy as np
import pandas as pd

import bokeh.embed
import bokeh.palettes
import bokeh.plotting
import bokeh.model
import bokeh.models
import bokeh.resources
from bokeh.io import show, output_file
from bokeh.layouts import column
from bokeh.models import LinearColorMapper, ColorBar, BasicTicker
from bokeh.models.layouts import TabPanel as Panel
from bokeh.models.layouts import Tabs
from bokeh.models import Range1d
import warnings
warnings.filterwarnings('ignore')


class BasePlot:
    """Base class for plotting frequency-based RFI statistics."""

    def __init__(self, dataset, **kwargs) -> None:
        self.dataset = dataset
        self.frequency = dataset.freqs / 1e6  # Convert frequency to MHz
        self.unixtime = dataset.timestamps  # Unix timestamps
        self.x_range = Range1d(self.frequency.min(), self.frequency.max())

    @staticmethod
    def _waterfall_plot(fig: bokeh.plotting.figure, *args, **kwargs) -> None:
        """Helper function to add a waterfall image plot with a colorbar."""
        fig.image(*args, **kwargs)
        color_mapper = LinearColorMapper(palette="Viridis256", low=0, high=1)
        color_bar = ColorBar(color_mapper=color_mapper, location=(5, 6), ticker=BasicTicker())
        fig.add_layout(color_bar, 'right')

    def make_rfi_stats_data_source(self, two_d_array) -> bokeh.models.ColumnDataSource:
        """Create a Bokeh data source from a 2D RFI statistics array."""
        return bokeh.models.ColumnDataSource({'image': two_d_array})

    def format_fig(self, title: str, pol: str):
        """Format a Bokeh figure for plotting."""
        return bokeh.plotting.figure(
            x_axis_label='Frequency [MHz]',
            y_axis_label=f'Pol {pol} Measurements',
            sizing_mode='stretch_width',
            title=title,
            width=1000, height=500, toolbar_location='above'
        )

    def collect_plots(self) -> Dict[str, Dict[str, bokeh.model.Model]]:
        """Generate frequency-based RFI plots for both HH and VV polarizations."""
        pols = ['HH', 'VV']
        plots_per_pol = {pol: self.make_plots(pol) for pol in pols}
        return plots_per_pol

    def make_plots(self, pol: str) -> Dict[str, bokeh.model.Model]:
        """Generate RFI plots for different flags."""
        flags = ['data_lost', 'cam', 'ingest_rfi', 'cal_rfi', 'combined_flags']
        plots = {}

        for flag in flags:
            logging.info(f'Processing {flag} for polarization {pol}')
            if flag != "combined_flags":
                self.dataset.select(scans='track', corrprods='cross', flags=flag, pol=pol)
            else:
                self.dataset.select(scans='track', corrprods='cross', pol=pol)

            two_d_array = self.extract_rfi_data(self.dataset)
            source = self.make_rfi_stats_data_source(two_d_array)
            fig = self.twoD_fig(flag, pol, source)
            plots[flag] = fig

        return plots

    def extract_rfi_data(self, dataset):
        """Abstract method to extract RFI data."""
        raise NotImplementedError("This method must be implemented in subclasses.")

    def write_metadata(self, dataset: katdal.DataSet, filename: str) -> None:
        """Write metadata to a JSON file."""

        metadata = {
            "ProductType": {
                "ProductTypeName": "MeerKATReductionProduct",
                "ReductionName": "RFIReport"
            },
            "Description": dataset.obs_params.get("description", "N/A"),
            "ScheduleBlockIdCode": dataset.obs_params.get("sb_id_code", "N/A"),
            "ProposalId": dataset.obs_params.get("proposal_id", "N/A"),
            "CaptureBlockId": dataset.obs_params.get("capture_block_id", "N/A"),
        }

        try:
            with open(filename, "w") as f:
                json.dump(metadata, f, allow_nan=False, indent=2)
            logging.info(f"Metadata successfully written to {filename}")
        except IOError as e:
            logging.info(f"Error writing metadata to {filename}: {e}")


class PlotFreqTimeStats(BasePlot):
    """Plot RFI statistics for Frequency-Time visualization."""

    def __init__(self, dataset, **kwargs) -> None:
        super().__init__(dataset, **kwargs)
        self.y_range = Range1d(0, len(self.unixtime))

    def extract_rfi_data(self, dataset):
        """Extract and process RFI data for frequency-time visualization."""
        return np.mean(dataset.flags, axis=2)

    def twoD_fig(self, title, pol,
                 source: bokeh.models.ColumnDataSource) -> bokeh.model.Model:
        """Generate a frequency-time plot."""
        fig = self.format_fig(title, pol)
        fig.x_range = self.x_range
        fig.y_range = self.y_range
        y_ticks_dic = self.make_ticks(self.dataset)
        fig.yaxis.ticker = np.array(list(y_ticks_dic.keys()), dtype=np.float64)
        fig.yaxis.major_label_overrides = y_ticks_dic
        self._waterfall_plot(
            fig, image=[source.data['image']], x=self.frequency.min(),
            y=0, dw=self.frequency.max() - self.frequency.min(),
            dh=len(self.unixtime), palette="Viridis256"
        )
        return fig

    def make_ticks(self, dataset: katdal.DataSet):
        """Generate y-axis tick labels for Bokeh plots."""

        # Extract scan names and observation targets
        scans = [scan[2].name for scan in dataset.scans()]
        targets = [
            dataset.sensor.get('Observation/target').__getitem__(i).name
            for i in range(dataset.shape[0])
        ]

        # Create y-axis values
        num_timestamps = len(targets)
        y_values = np.arange(num_timestamps)

        # Determine step size for tick selection
        step = max(1, num_timestamps // (len(scans) + 1))

        # Select ticks and corresponding labels
        selected_ticks = y_values[::step]
        selected_labels = {int(y): targets[i] for i, y in enumerate(selected_ticks)}
        return selected_labels


class PlotFreqBaseline(BasePlot):
    """Plot RFI statistics for Frequency-Baseline visualization."""

    def __init__(self, dataset, path_bl_csv: str, **kwargs) -> None:
        super().__init__(dataset, **kwargs)
        self.path_bl_csv = path_bl_csv
        self.ordered_bl = self.get_bl_idx(dataset)[1]
        self.y_range = Range1d(self.ordered_bl.min(), self.ordered_bl.max())

    def get_bl_idx(self, dataset):
        """Get ordered baseline indices."""
        bl_lens = pd.read_csv(self.path_bl_csv)
        corrprods = self.get_corrprods(dataset)
        bl_idx = np.argsort([bl_lens[corr].values[0] for corr in corrprods])
        ordered_bl = np.array([bl_lens[corr].values[0] for corr in corrprods])[bl_idx]
        return bl_idx, ordered_bl

    def get_corrprods(self, dataset):
        """Get correlation products."""
        return np.array([bl[0][:-1] + bl[1][:-1] for bl in dataset.corr_products])

    def extract_rfi_data(self, dataset):
        """Extract and process RFI data for frequency-baseline visualization."""
        two_d_array = np.mean(dataset.flags, axis=0)
        bl_idx = self.get_bl_idx(dataset)[0]
        return two_d_array[:, bl_idx].T

    def twoD_fig(self, title, pol, source: bokeh.models.ColumnDataSource) -> bokeh.model.Model:
        """Generate a frequency-baseline plot."""
        fig = self.format_fig(title, pol)
        fig.x_range = self.x_range
        fig.y_range = self.y_range

        self._waterfall_plot(
            fig, image=[source.data['image']], x=self.frequency.min(),
            y=self.ordered_bl.min(),
            dw=self.frequency.max() - self.frequency.min(),
            dh=self.ordered_bl.max() - self.ordered_bl.min(),
            palette="Viridis256"
        )
        return fig


class RfiReportLayout:
    """Create an RFI Report layout and generate an HTML report."""

    def __init__(self, bokeh_models: Dict[str, Dict[str, object]], filename: str) -> None:
        """
        Initialize the RFI report layout.

        Parameters:
        -----------
        bokeh_models : dict
            Dictionary containing Bokeh plots categorized by polarization and flag type.
        filename : str
            Output filename for the HTML report.
        """
        self.plots = bokeh_models
        self.filename = filename

    def create_layout(self) -> None:
        """
        Generate a Bokeh layout with categorized RFI flag plots and save it as an HTML file.
        """
        # Extract plots for HH and VV polarizations
        hh_plots = self.plots['HH']
        vv_plots = self.plots['VV']

        # Define flag categories
        flag_categories = {
            "All flags": "combined_flags",
            "Ingest flags": "ingest_rfi",
            "Cal flags": "cal_rfi",
            "Data lost flags": "data_lost",
            "Cam flags": "cam"
            }

        # Create Bokeh panels for each flag category
        tabs = [
            Panel(
                child=column(hh_plots[flag], vv_plots[flag], sizing_mode="stretch_width"),
                title=title
            )
            for title, flag in flag_categories.items()
        ]

        # Assemble and display the layout
        layout = Tabs(tabs=tabs)
        logging.info(f"Saving report to: {self.filename}")
        output_file(self.filename, mode='cdn')
        show(layout)

    def create_main_html(self, main_filename: str, other_html_files: List[str],
                         output_dir: str) -> None:
        """
        Create a main HTML file that links multiple RFI report HTML files together.

        Parameters:
        -----------
        main_filename : str
            Path to the main HTML output file.
        other_html_files : list
            List of HTML report filenames to be embedded.
        output_dir : str
            Directory where the HTML reports are stored.
        """
        plot_types = [
            "Frequency-Time RFI Statistics",
            "Frequency-Baseline RFI Statistics"
        ]

        # Start HTML content
        main_html_content = """<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>MeerKAT RFI Report</title>
        </head>
        <body>
            <h1>MeerKAT RFI Report</h1>
        """

        # Embed each additional HTML report
        for i, html_file in enumerate(other_html_files):
            html_path = os.path.join(output_dir, html_file)
            with open(html_path, "r") as file:
                main_html_content += f"<h2>{plot_types[i]}</h2>\n{file.read()}"

        # Close HTML structure
        main_html_content += "\n</body>\n</html>"

        # Write to the main HTML file
        with open(main_filename, "w") as file:
            file.write(main_html_content)

        logging.info(f"Main HTML report saved: {main_filename}")
