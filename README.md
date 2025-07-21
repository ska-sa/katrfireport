# katrfireport

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

---

**katrfireport** is a Python package for computing, storing, and interactively visualising radio-frequency interference (RFI) statistics for MeerKAT telescope observations.  
It processes `katdal` datasets to generate Zarr-format RFI flag statistics and presents interactive dashboards using [Panel](https://panel.holoviz.org/) and [HoloViews](https://holoviews.org/).

---

## Features

- Compute per-polarisation RFI flagging statistics from MeerKAT `katdal` datasets  
- Save detailed RFI flag statistics in efficient Zarr format  
- Interactive dashboards with linked frequency-time and baseline-frequency plots  
- Hover and click interactions reveal detailed RFI fraction info by frequency, time, and baseline  
- Download the underlying Zarr data for offline analysis  
- Scan-based RFI fraction summaries with weighted averages and detailed plots

---

## Installation

```
pip install katrfireport
```
Alternatively, install from source

```bash
git clone https://github.com/yourusername/katrfireport.git
cd katrfireport
pip install -e .
```

---

## Usage

The tool provides two main commands via the CLI:
1. Compute RFI statistics
```
python3 -m katrfireport.cli compute <katdata> <output_dir>
```
- ```<katdata>```: Path to your MeerKAT dataset (e.g., 1746243356_sdp_l0.full.rdb)
- ```<output_dir>```: Directory to save the Zarr output

Example,
```
python3 -m katrfireport.cli compute 1746243356_sdp_l0.full.rdb /scratch/isihlangu
```

This creates:
```
/scratch/isihlangu/flag_rfi_stats_<CBID>
```
Inside there is a zarr store file and the metadata JSON file.
```
flag_stats_<CBID>.zarr
metadata_<CBID>.json
```

2. Serve the interactive dashboard
Once the Zarr data exists, serve the dashboard:
```
python3 -m katrfireport.cli serve <zarr_path> <katdata> [--port PORT] [--allow-origin ORIGIN]
```

- ```<zarr_path>```: Path to your Zarr directory (output of compute step)
- ```--port```: (Optional) Port for Panel server (default: 5006)
- ```--allow-origin```: (Optional) Allowed websocket origin for remote access

Example
```
python3 -m katrfireport.cli serve /scratch/isihlangu/flag_rfi_stats_1746243356/flag_stats_1746243356.zarr/
   --port 8887 \
  --allow-origin qgpu01.sdpdyn.kat.ac.za:8887
```

Open in browser:
```
http://<your-host>:<port>
```







