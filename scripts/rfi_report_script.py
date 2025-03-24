#!/usr/bin/env python3
import argparse
import os
import logging
import sys
import uuid

import katdal
from katsdpservices import setup_logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import katrfireport.mkat_rfi_report as report


def create_parser():
    parser = argparse.ArgumentParser(description='MeerKAT Interactive RFI report.')
    logging.info('MEERKAT RFI REPORT')
    parser = argparse.ArgumentParser()
    parser.add_argument('katdata', type=str, help='Katdal observation reference')
    parser.add_argument('--access-key',
                        help='S3 access key to access the data')
    parser.add_argument('--secret-key',
                        help='S3 secret key to access the data')
    parser.add_argument('--token',
                        help="JWT to access the MeerKAT archive")
    parser.add_argument('output_dir', type=str, help='Parent directory for output')
    parser.add_argument('prefix', type=str, help='Prefix for output directories and filenames')
    parser.add_argument('--log-level', type=str, metavar='LEVEL',
                        help='Logging level [INFO]')
    return parser


def main() -> None:

    setup_logging()
    parser = create_parser()
    args = parser.parse_args()
    # Read the observation
    if args.log_level is not None:
        logging.getLogger().setLevel(args.log_level.upper())

    dataset = katdal.open(args.katdata, upgrade_flags=True)
    output_dir = '{}_{}'.format(args.prefix, uuid.uuid4())
    output_dir = os.path.join(args.output_dir, output_dir)
    # Define the relative path to the baseline length file
    path_bl_csv = os.path.join(os.path.dirname(__file__), '..', 'conf', 'meerkatbaselinelength.csv')
    cbid = dataset.name.split('_')[0]
    tmp_dir = output_dir + '.writing'
    os.mkdir(tmp_dir)
    os.chdir(tmp_dir)
    attributes = ['freq_time', 'freq_baseline']
    html_files = []
    # Create JSON metdata file
    for i, val in enumerate(attributes):
        try:
            logging.info('Collecting {} RFI Statistics'.format(val))
            filename = (cbid + '_' + '{}' + '_' + args.prefix + '_' + 'report.html').format(val)
            html_files.append(filename)
            if val == 'freq_time':
                rfi_stats = report.PlotFreqTimeStats(dataset)
            if val == 'freq_baseline':
                rfi_stats = report.PlotFreqBaseline(dataset, path_bl_csv=path_bl_csv)
            logging.info('Creating bokeh report for {}'.format(val))
            plots = rfi_stats.collect_plots()
            logging.info('Creating bokeh report for {}'.format(val))
            layout = report.RfiReportLayout(plots, filename)
            layout.create_layout()
        except Exception:
            raise
    layout.create_main_html('index.html', html_files, tmp_dir)
    rfi_stats.write_metadata(dataset, 'metadata.json')
    os.rename(tmp_dir, output_dir)


if __name__ == '__main__':
    main()
