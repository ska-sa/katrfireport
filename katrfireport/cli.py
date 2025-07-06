import argparse
import logging
import os

import katdal
from dask.diagnostics import ProgressBar
from katsdpservices import setup_logging
import panel as pn

from .core import process_dual_pol
from .dashboard import create_dual_pol_dashboard


def compute(katdata, output_dir):
    """Compute RFI stats and save Zarr data"""
    # Setup logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    setup_logging()

    katds = katdal.open(katdata, upgrade_flags=True)
    cbid = katds.name.split('_')[0]
    zarr_path = os.path.join(output_dir, f'flag_stats_{cbid}.zarr')
    tmp_dir = zarr_path + '.writing'

    if not os.path.exists(zarr_path):
        os.makedirs(tmp_dir, exist_ok=True)
        logging.info("🔹 Computing RFI statistics.")
        with ProgressBar():
            process_dual_pol(katds, ['HH', 'VV'], tmp_dir)
        os.rename(tmp_dir, zarr_path)
        logging.info(f"✅ Created Zarr: {zarr_path}")
    else:
        logging.info(f"✅ Zarr already exists: {zarr_path}")


def serve(zarr_path, katdata, port=5006, allow_origin=None):
    """Serve the RFI dashboard"""
    katds = katdal.open(katdata, upgrade_flags=True)
    katds.select(corrprods='cross', scans='track', pol='HH')
    dashboard = create_dual_pol_dashboard(zarr_path, katds)

    kwargs = {
        'address': '0.0.0.0',
        'port': port,
        'show': False,
    }
    if allow_origin:
        kwargs['allow_websocket_origin'] = [allow_origin]

    pn.serve(dashboard, **kwargs)


def main():
    parser = argparse.ArgumentParser(description="Generate and serve MeerKAT RFI reports")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Compute command
    compute_parser = subparsers.add_parser("compute", help="Compute RFI statistics Zarr file")
    compute_parser.add_argument("katdata", help="Path to katdal dataset")
    compute_parser.add_argument("output_dir", help="Directory to save Zarr + stats")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Serve the RFI dashboard")
    serve_parser.add_argument("zarr_path", help="Path to Zarr directory")
    serve_parser.add_argument("katdata", help="Path to katdal dataset")
    serve_parser.add_argument("--port", type=int, default=5006, help="Port for Panel server")
    serve_parser.add_argument("--allow-origin", help="Allowed websocket origin (e.g. host:port)")

    args = parser.parse_args()

    if args.command == "compute":
        compute(args.katdata, args.output_dir)
    elif args.command == "serve":
        serve(args.zarr_path, args.katdata, port=args.port, allow_origin=args.allow_origin)


if __name__ == "__main__":
    main()
