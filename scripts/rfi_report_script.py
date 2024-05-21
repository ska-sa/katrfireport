#!/usr/bin/env python3
import argparse
import os
import logging
import shutil
import uuid

import katdal

import katrfireport.mkat_rfi_report as report


def initialize_logs():
    """
    Initialize the log settings
    """
    logging.basicConfig(format='%(message)s', level=logging.INFO)


def main() -> None:
    # Initializing the log settings
    initialize_logs()
    logging.info('MEERKAT RFI REPORT')
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='Input dataset')
    parser.add_argument('baseline_length', type=str, help='path to the csv file with MeerKAT '
                        'anntena pair lengths')
    parser.add_argument('output_dir', type=str, help='Parent directory for output')
    parser.add_argument('prefix', type=str, help='Prefix for output directories and filenames')
    parser.add_argument('--log-level', type=str, metavar='LEVEL',
                        help='Logging level [INFO]')
    args = parser.parse_args()
    if args.log_level is not None:
        logging.getLogger().setLevel(args.log_level.upper())

    dataset = katdal.open(args.dataset, upgrade_flags=True)
    output_dir = '{}_{}'.format(args.prefix, uuid.uuid4())
    output_dir = os.path.join(args.output_dir, output_dir)
    cbid = dataset.name.split('_')[0]
    path_bl_csv = args.baseline_length
    tmp_dir = output_dir + '.writing'
    os.mkdir(tmp_dir)
    attributes = ['freq_time', 'freq_baseline']
    html_files = []
    for i, val in enumerate(attributes):
        try:
            logging.info('Collecting {} RFI Statistics'.format(val))
            filename = (cbid + '_' + '{}' + '_' + args.prefix + '_' + 'report.html').format(val)
            html_files.append(filename)
            if val == 'freq_time':
                rfi_stats = report.PlotFreqTimeStats(dataset)
            if val == 'freq_baseline':
                rfi_stats = report.PlotFreqBaseline(dataset, path_bl_csv=path_bl_csv)

            plots = rfi_stats.collect_plots(dataset)
            logging.info('Creating bokeh report for {}'.format(val))
            layout = report.RfiReportLayout(plots)
            layout.create_layout(filename)
            if i == 0:
                os.rename(tmp_dir, output_dir)
            shutil.move(filename, output_dir)
        except Exception:
            # Make a best effort to clean up
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
    # Create JSON metdata file
    rfi_stats.write_metadata(dataset, 'metadata.json')
    shutil.move('metadata.json', output_dir)
    # Create main html file
    rfi_stats.create_main_html('index.html', html_files, output_dir)
    shutil.move('index.html', output_dir)


if __name__ == '__main__':
    main()
