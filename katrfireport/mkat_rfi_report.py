import logging
from typing import Dict
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
from bokeh.io import output_notebook
from bokeh.layouts import gridplot, column
from bokeh.models import LinearColorMapper, ColorBar, BasicTicker
from bokeh.models.widgets import Panel
from bokeh.models.widgets import Tabs
from bokeh.models import Range1d
import warnings
warnings.filterwarnings('ignore')


class PlotFreqTimeStats:
    """Collect RFI statistics for frequency-time plot."""
    def __init__(self, dataset: katdal.DataSet, **kwargs) -> None:
        self.unixtime = dataset.timestamps  # Unix timestamps
        self.frequency = dataset.freqs/1e6  # Frequency in MHz
        self.x_range = Range1d(self.frequency.min(), self.frequency.max())
        self.y_range = Range1d(0, len(self.unixtime))

    @staticmethod
    def _waterfall_plot(fig: bokeh.plotting.Figure, *args, **kwargs) -> None:
        fig.image(*args, **kwargs)
        # Creating a colorbar object
        color = LinearColorMapper(palette="Viridis256", low=0, high=1)
        cb = ColorBar(color_mapper=color, location=(5, 6), ticker=BasicTicker())
        fig.add_layout(cb, 'right')

    def make_rfi_stats_data_source(self, two_d_array) -> bokeh.models.ColumnDataSource:
        data = {'image': two_d_array}
        return bokeh.models.ColumnDataSource(data)

    def format_fig(self, title, pol, dataset: katdal.DataSet):
        fig = bokeh.plotting.figure(
        x_axis_label='Frequency MHz',
        y_axis_label=('Pol {} Scans').format(pol),
        sizing_mode='stretch_width',
        title=title,
        width=1000, height=500, toolbar_location='above')
        return fig

    def freq_time_fig(self, dataset: katdal.DataSet, title, pol,
                      source: bokeh.models.ColumnDataSource) -> bokeh.model.Model:
        fig = self.format_fig(title, pol)
        freqs = self.frequency
        fig.x_range = self.x_range
        fig.y_range = self.y_range
        y_ticks_dic = self.make_ticks(dataset)
        fig.yaxis.ticker = np.array([*y_ticks_dic], dtype=np.float)
        fig.yaxis.major_label_overrides = y_ticks_dic
        self._waterfall_plot(
            fig, image=[source.data['image']], x=freqs.min(),
            y=0, dw=freqs.max()-freqs.min(),
            dh=len(self.unixtime), palette="Viridis256")
        return fig

    def make_ticks(self, dataset: katdal.DataSet):
        """Make axis ticks"""
        # Make yticks dictionary required by bokeh
        targets = [scan[2].name for scan in dataset.scans()]
        nscans = dataset.shape[0]
        step = nscans//len(targets)+1
        indices = np.arange(0, nscans)[::step]+10
        y_ticks_dic = {}
        for i in range(len(targets)):
            y_ticks_dic[str(indices[i])] = targets[i]
        return y_ticks_dic

    def make_plots(self, pol, dataset: katdal.DataSet) -> Dict[str, bokeh.model.Model]:
        """Generate Bokeh figures for the plots."""

        flags = ['data_lost', 'cam', 'ingest_rfi', 'cal_rfi', 'combined_flags']
        plots_source = {}
        for i in range(len(flags)):
            logging.info(' {} flags'.format(flags[i]))
            if flags[i] != 'combined_flags':
                dataset.select(scans='track', corrprods='cross', flags=flags[i], pol=pol)
            else:
                dataset.select(scans='track', corrprods='cross', pol=pol)
            two_d_array = np.mean(dataset.flags[:, :, :], axis=2)
            source = self.make_rfi_stats_data_source(two_d_array)
            fig = self.freq_time_fig(dataset, flags[i], pol, source)
            plots_source[flags[i]] = fig
        return plots_source

    def collect_plots(self, dataset: katdal.DataSet):
        """Collect HH and VV frequency time plots."""

        pols = ['HH', 'VV']
        plots_per_pol = {}
        for i in range(len(pols)):
            logging.info('Processing {} polarization'.format(pols[i]))
            plots_per_pol[pols[i]] = self.make_plots(pols[i], dataset)
        return plots_per_pol

    def write_metadata(self, dataset: katdal.DataSet,
                       filename: str) -> None:
        metadata = {
            'ProductType': {
                'ProductTypeName': 'MeerKATReductionProduct',
                'ReductionName': 'RFIReport'
            },
            'Description': dataset.obs_params['description'],
            'ScheduleBlockIdCode': dataset.obs_params['sb_id_code'],
            'ProposalId': dataset.obs_params['proposal_id'],
            'CaptureBlockId': dataset.obs_params['capture_block_id']
        }
        with open(filename, 'w') as f:
            json.dump(metadata, f, allow_nan=False, indent=2)

    def create_main_html(self, main_filename, other_html_files, output_dir):
        # Content of main HTML file
        main_html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>MeerKAT RFI Report</title>
        </head>
        <body>
            <h1>MeerKAT RFI Report</h1>
        """
        plot_types = ['Frequency-Time RFI Statistics', 'Frequency-Baseline RFI Statistics']
        # Read and embed the content of other HTML files
        for i, html_file in enumerate(other_html_files):
            html_f = os.path.join(output_dir, html_file)
            with open(html_f, "r") as file:
                html_content = file.read()
                main_html_content += f"<h2>{plot_types[i]}</h2>"
                main_html_content += html_content

        main_html_content += """
        </body>
        </html>
        """

        # Write content to main HTML file
        with open(main_filename, "w") as file:
            file.write(main_html_content)


class PlotFreqBaseline(PlotFreqTimeStats):
    """Collect RFI for frequency-baeline statistics."""

    def __init__(self, dataset: katdal.DataSet, **kwargs) -> None:
        self.path_bl_csv = kwargs['path_bl_csv']
        self.frequency = dataset.freqs/1e6  # Frequency in MHz
        self.unixtime = dataset.timestamps  # Unix timestamps
        self.x_range = Range1d(self.frequency.min(), self.frequency.max())
        self.ordered_bl = self.get_bl_idx(dataset)[1]
        self.y_range = Range1d(self.ordered_bl.min(), self.ordered_bl.max())

    def get_bl_idx(self, dataset: katdal.DataSet):
        """
        Get the indices of the correlation products.

        Parameters:
        -----------
        vis : katdal.visdatav4.VisibilityDataV4
           katdal data object

        Returns:
        --------
        output : numpy array
           array of ordered baseline indices
        """
        bl_lens = pd.read_csv(self.path_bl_csv)
        corrprods = self.get_corrprods(dataset)
        corrprod_bl = []
        for corrprod in corrprods:
            corrprod_bl.append(bl_lens[corrprod].values[0])
        # Baseline length as per correlation products
        corrprod_bl = np.array(corrprod_bl)
        bl_idx = np.argsort(corrprod_bl)
        ordered_bl = corrprod_bl[bl_idx]
        return bl_idx, ordered_bl

    def get_corrprods(self, dataset: katdal.DataSet):
        """
        Get the correlation products

        Parameters:
        ----------
        vis : katdal.visdatav4.VisibilityDataV4
           katdal data object

        Returns:
        --------
        output : numpy array
             array of anntena combination of the correlation products
        """
        bl = dataset.corr_products
        corrprods = []
        for i in range(len(bl)):
            corrprods.append((bl[i][0][0:-1]+bl[i][1][0:-1]))
        return np.array(corrprods)

    def make_plots(self, pol, dataset: katdal.DataSet) -> Dict[str, bokeh.model.Model]:
        """Generate Bokeh figures for the plots."""
        flags = ['data_lost', 'cam', 'ingest_rfi', 'cal_rfi', 'combined_flags']
        plots_source = {}

        for i in range(len(flags)):
            logging.info(' {} flags'.format(flags[i]))
            if flags[i] != 'combined_flags':
                dataset.select(scans='track', corrprods='cross', flags=flags[i], pol=pol)
            else:
                dataset.select(scans='track', corrprods='cross', pol=pol)
            two_d_array = np.mean(dataset.flags[:, :, :], axis=0)
            bl_idx = self.get_bl_idx(dataset)[0]
            two_d_array = two_d_array[:, bl_idx]
            source = self.make_rfi_stats_data_source(two_d_array.T)
            fig = self.freq_time_fig(dataset, flags[i], pol, source)
            plots_source[flags[i]] = fig
        return plots_source

    def freq_time_fig(self, dataset: katdal.DataSet, title, pol,
                      source: bokeh.models.ColumnDataSource) -> bokeh.model.Model:
        fig = self.format_fig(title, pol)
        freqs = self.frequency
        fig.x_range = self.x_range
        fig.y_range = self.y_range
        self._waterfall_plot(
            fig, image=[source.data['image']],
            x=freqs.min(), y=self.ordered_bl.min(),
            dw=freqs.max()-freqs.min(),
            dh=self.ordered_bl.max()-self.ordered_bl.min(),
            palette="Viridis256")
        return fig

      def format_fig(self, title, pol, dataset: katdal.DataSet):    
        fig = bokeh.plotting.figure(
            x_axis_label='Frequency [MHz]',
            y_axis_label=('Pol {} Baseline length [m]').format(pol),
            sizing_mode='stretch_width',
            title=title,
            width=1000, height=500, toolbar_location='above')


class RfiReportLayout:
    """Create RFI Report layout"""
    def __init__(self, bokeh_models, **kwargs) -> None:
            self.plots = bokeh_models
            self.cbid = kwargs['cbid']
    def create_layout(self):
        plots = self.plots
        HH = plots['HH']
        VV = plots['VV']
        system_flags = column(HH['combined_flags'], VV['combined_flags'])
        ingest_flags = column( HH['ingest_rfi'], VV['ingest_rfi'])
        cal_flags = column( HH['cal_rfi'], VV['cal_rfi'])
        data_lost = column(HH['data_lost'],VV['data_lost'] )
        cam_flags = column( HH['cam'], VV['cam'])
        # Create tabs
        tab1 = Panel(child=system_flags, title='All flags')
        tab2 = Panel(child=ingest_flags, title='Ingest RFI flags')
        tab3 = Panel(child=cal_flags, title='Cal RFI flags')
        tab4 = Panel(child=data_lost, title='data RFI flags')
        tab5 = Panel(child=cam_flags, title='cam RFI flags')
        
        # create a layout from tabs
        layout = Tabs(tabs=[tab1, tab2, tab3, tab4, tab5])
        # save html layout into disk
        filename='MeerKAT_RFI_Report_'+str(self.cbid)+'.html'
        output_file(filename, mode='inline')
        return show(layout)
