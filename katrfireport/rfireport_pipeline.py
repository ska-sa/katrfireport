import argparse
import logging
import os

import katdal
from dask.diagnostics import ProgressBar
from katsdpservices import setup_logging
import panel as pn

from .core import process_crosscorr, process_autocorr, write_metadata
from .dashboard import RFIDashboard

import warnings

# Suppress only that specific warning from Holoviews / pandas
warnings.filterwarnings(
    "ignore",
    message="Discarding nonzero nanoseconds in conversion.",
    category=UserWarning,
    module=r"holoviews\.core\.data\.pandas",
)


def export_static(zarr_path, output_dir):
    """Export the RFI dashboard as a standalone static HTML file in the same folder as Zarr."""
    pn.extension()

    dashboard = RFIDashboard(zarr_path)

    panel_layout = dashboard.build()

    cbid = os.path.basename(zarr_path).split("_")[0]
    output_html = os.path.join(output_dir, f"{cbid}_MeerKAT_Static_RFI_report.html")

    panel_layout.save(
        output_html,
        embed=True,
        resources="inline",
        title="MeerKAT RFI Report",
    )

    logging.info(f"Static HTML dashboard saved to {output_html}")


def compute(katdata, output_path):
    """
    Compute RFI stats for both cross-correlation and autocorrelation,
    save them to a single Zarr store, and generate an HTML report in the same folder.

    Parameters
    ----------
    katdata : str
        Path to the observation dataset (e.g., MeerKAT HDF5).
    output_path : str
        Folder where Zarr and HTML report will be saved.
    """

    # Reset logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    setup_logging()

    # Open the dataset
    katds = katdal.open(katdata, upgrade_flags=True)
    cbid = katds.name.split("_")[0]
    filename = f"{cbid}_flag_rfi_stats"
    output_dir = os.path.join(output_path, filename)

    # Compute and save RFI stats if Zarr store does not exist
    if not os.path.exists(output_dir):
        tmp_dir = output_dir + ".writing"
        zarr_path = os.path.join(tmp_dir, f"{cbid}_flag_stats.zarr")
        os.makedirs(tmp_dir, exist_ok=True)
        os.chdir(tmp_dir)
        logging.info("Writing metadata to JSON file.")
        write_metadata(katds, "metadata.json")

        logging.info("Computing cross-correlation RFI statistics...")
        with ProgressBar():
            process_crosscorr(katds, ["HH", "VV"], zarr_path)

        logging.info("Computing autocorrelation RFI statistics...")
        with ProgressBar():
            process_autocorr(katds, ["HH", "VV"], zarr_path)
        logging.info("Creating static HTML dashboard.")
        # Generate static HTML dashboard from Zarr
        export_static(zarr_path, tmp_dir)
        # Move final folder into place
        os.chdir(output_path)
        os.rename(tmp_dir, output_dir)
    else:
        zarr_path = os.path.join(output_dir, f"{cbid}_flag_stats.zarr")
        logging.info(f"Zarr store already exists: {zarr_path}")
        logging.info("Creating static HTML dashboard.")
        # Generate static HTML dashboard from Zarr
        export_static(zarr_path, output_dir)
    logging.info(f"Report directory finalized: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate, serve or export MeerKAT RFI reports"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run command
    compute_parser = subparsers.add_parser(
        "run", help="Compute RFI statistics save it to Zarr file and create RFI dashboard"
    )
    compute_parser.add_argument("katdata", help="Path to katdal dataset")
    compute_parser.add_argument("output_path", help="Directory to save Zarr + stats")

    args = parser.parse_args()

    if args.command == "run":
        compute(args.katdata, args.output_path)


if __name__ == "__main__":
    main()
