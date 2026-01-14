# katrfireport

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

---

**katrfireport** is a Python package for computing, storing, and interactively visualising radio-frequency interference (RFI) statistics for MeerKAT telescope observations.  
It processes `katdal` datasets to generate Zarr-format RFI flag statistics and presents interactive dashboards using [Panel](https://panel.holoviz.org/) and [HoloViews](https://holoviews.org/).

---

## Features

- Compute per-polarisation RFI flagging statistics from MeerKAT `katdal` datasets  
- Save detailed RFI flag statistics in efficient Zarr format  
- Interactive dashboards with linked frequency-time and baseline-frequency plots for cross correlation products.
- Interactive dashboards with linked frequency-time plots for auto correlation products. 
- Hover and click interactions reveal detailed RFI fraction info by frequency, time, baseline and antenna.
- Download the underlying Zarr data for offline analysis  

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

The tool provides two modes, the online and offline version.
1. Online mode: Compute RFI statistics, save it to a zarr store and create HTML RFI report.
```
python3 -m katrfireport.rfireport_pipeline run <katdata> <output_dir>
```
- ```<katdata>```: Path to your MeerKAT dataset (e.g., 1746243356_sdp_l0.full.rdb)
- ```<output_dir>```: Directory to save the Zarr output

Example,
```
python3 -m katrfireport.rfireport_pipeline run 1746243356_sdp_l0.full.rdb /scratch/isihlangu
```

This creates:
```
/scratch/isihlangu/flag_rfi_stats_<CBID>
```
Inside there is a zarr store file, the metadata JSON file and the HTML RFI report.
```
flag_stats_<CBID>.zarr
metadata_<CBID>.json
<CBID>_MeerKAT_Static_RFI_report.html
```

1. Offline mode: Compute RFI statistics, save it to a zarr store and serve.







