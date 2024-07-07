# Copyright (c) 2015-2024 Satpy developers
#
# This file is part of satpy.
#
# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.
"""Base class for performance test."""
import csv
import glob
import os
import platform
import time
from datetime import datetime, UTC
from io import BytesIO
from itertools import zip_longest
from threading import Thread

import cpuinfo
import numpy as np
import pandas as pd
import psutil
import matplotlib.pyplot as plt


class SatpyPerformanceTest:
    """Test satpy performance by looping through conditions involving ``dask_array_chunk_size``and ``dask_num_workers``.
    There are two tests: ``simple_test`` and ``resampler_test``

    """
    def __init__(self, work_dir, folder_pattern, reader_name, composite, chunk_size_opts, worker_opts,
                 reader_kwargs=None):
        """Initialize SatpyPerformanceTest with some basic arguments.

        Args:
            work_dir (str): Absolute path to the directory that contains all your dataset folders.
            folder_pattern (str): Naming scheme of the dataset folders, e.g. `G16_s*_e*_FLDK`.
            reader_name (str): Reader you want to test.
            composite (str): Composite for test. Usually this could be ``true_color`` which involves a
                             lot of computation like atmospheric correction.
            chunk_size_opts (list): All the ``dask_array_chunk_size`` values you wish for, in `MiB`.
            worker_opts (list): All the ``dask_num_workers`` values you wish for.
            reader_kwargs (dict): Additional reader arguments for ``Scene``,
                                  e.g. `{'mask_saturated': False}` in modis_l1b.

        """
        super().__init__()
        self.work_dir = work_dir
        self.folder_pattern = folder_pattern
        self.reader_name = reader_name
        self.reader_kwargs = reader_kwargs
        self.composite = composite

        self.folders = glob.glob(f"{self.work_dir}/{self.folder_pattern}")

        self.chunk_size_opts = chunk_size_opts
        self.worker_opts = worker_opts
        self.total_rounds = len(self.chunk_size_opts) * len(self.worker_opts)

        self.result = {}
        self.running = True

    def monitor_system_usage(self, interval=0.5):
        """Use psutil to record CPU and memory usage. Default sample rate is 0.5s."""
        os_type = platform.system()

        process = psutil.Process()
        cpu_usage = []
        memory_usage = []
        timestamps = []

        start_time = time.time()
        while self.running:
            cpu_usage.append(process.cpu_percent())
            if os_type == "Windows":
                # In Windows, "vms" means "pagefile"
                memory_usage.append((process.memory_full_info().rss + process.memory_full_info().vms))
            elif os_type == "Linux":
                memory_usage.append((process.memory_full_info().rss + process.memory_full_info().swap))
            else:
                memory_usage.append(process.memory_full_info().rss)
            timestamps.append(time.time() - start_time)
            time.sleep(interval)

        self.result["cpu_usage"] = cpu_usage
        self.result["memory_usage"] = memory_usage
        self.result["timestamps"] = timestamps

    def write_to_csv(self, file_name):
        """Write the result of each round to a csv file."""
        with open(file_name, "w", newline="", encoding="utf-8") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["Timestamp (s)", "CPU Usage (%)", "Memory Usage (Byte)", "Process Time", "Scenes",
                                "Errors"])
            for ts, cpu, mem, pt, scn, er in zip_longest(self.result["timestamps"], self.result["cpu_usage"],
                                                         self.result["memory_usage"], self.result["process_time"],
                                                         self.result["scenes"], self.result["errors"], fillvalue="N/A"):
                csvwriter.writerow([ts, cpu, mem, pt, scn, er])

    def satpy_test(self, resampler, diff_res=False, area_def=None, resampler_kwargs=None):
        """Call satpy to do the test."""
        from satpy import find_files_and_readers, Scene

        reader_kwargs = {} if self.reader_kwargs is None else self.reader_kwargs
        resampler_kwargs = {} if resampler_kwargs is None else resampler_kwargs
        for folder in self.folders:
            files = find_files_and_readers(base_dir=folder, reader=self.reader_name)
            scn = Scene(filenames=files, reader_kwargs=reader_kwargs)
            scn.load([self.composite], generate=False if diff_res else True)

            if resampler == "none":
                scn2 = scn
            else:
                scn2 = scn.resample(area_def, resampler=resampler, **resampler_kwargs)

            scn2.save_dataset(self.composite, writer="geotiff", filename="test.tif", base_dir=self.work_dir,
                              fill_value=0, compress=None)

    def single_loop(self, conditions, area, diff_res=False, area_def=None, resampler_kwargs=None):
        """Single round of the test. """
        import dask.config
        self.running = True

        chunk_size, num_worker, resampler = conditions

        dask.config.set({"array.chunk-size": f"{chunk_size}MiB"})
        dask.config.set(num_workers=num_worker)

        try:
            num_thread = os.environ["OMP_NUM_THREADS"]
        except KeyError:
            num_thread = psutil.cpu_count(logical=True)

        # Start recording cpu/mem usage
        monitor_thread = Thread(target=self.monitor_system_usage, args=(0.5,))
        monitor_thread.start()

        errors = []
        start = time.perf_counter()
        try:
            self.satpy_test(resampler, diff_res, area_def, resampler_kwargs)
        except Exception as e:
            errors.append(e)

        end = time.perf_counter()
        # All of these must be list object
        self.result["process_time"] = [end - start]
        self.result["scenes"] = [len(self.folders)]
        self.result["errors"] = errors

        # Stop recording
        self.running = False
        monitor_thread.join()

        csv_file = (f"{self.work_dir}/{self.reader_name.replace("_", "")}_"
                    f"chunk{chunk_size}_worker{num_worker}_thread{num_thread}_{area}_{resampler}.csv")
        self.write_to_csv(csv_file)

    def simple_test(self, diff_res=False):
        """Test readers in dataset's original projection. No resampling involved or the simplest ``native`` resampling.

        Args:
            diff_res (bool): If the composite requires bands in different resolutions, this should be set to True
                             so the native resampler will match them to the ``scn.finest_area()``.
                             For example, ``true_color`` of ABI needs 500m C01 and 1000m C02 bands, so it's `True`.
                             This is not a test option and should be set properly according to the composite,
                             otherwise the test will end up in errors.

        """
        resampler = "native" if diff_res else "none"
        area = "original"

        i = 0
        for chunk_size in self.chunk_size_opts:
            for num_worker in self.worker_opts:
                print(f"Start testing CHUNK_SIZE={chunk_size}MiB, NUM_WORKER={num_worker}, resampler is {resampler}.") # noqa
                self.single_loop((chunk_size, num_worker, resampler), area, diff_res)
                i = i + 1

                if i == self.total_rounds:
                    print("All the tests finished. Generating HTML report.") # noqa
                    html_report(self.work_dir, self.reader_name)
                else:
                    print(f"ROUND {i} / {self.total_rounds} Completed. Now take a 1-min rest.") # noqa
                    time.sleep(60)

    def resampler_test(self, resamplers, area_def, resampler_kwargs=None):
        """Test readers with resampling. See https://satpy.readthedocs.io/en/latest/resample.html#resampling-algorithms
        for available resampler.

        Args:
            resamplers (list): List of resampling algorithms you want to test.
            area_def (AreaDefinition or DynamicAreaDefinition or str): Area definition or the name of an area stored
                                                                       in YAML.
            resampler_kwargs (dict): Additional arguments passed to the resampler, e.g.
                                     {'cache_dir': '/path/to/my/cache'} for ``bilinear`` or ``nearest``.

        """
        resampler_kwargs = {} if resampler_kwargs is None else resampler_kwargs
        area = "local" if len(area_def.area_id) == 0 else area_def.area_id.replace("_", "")

        i = 0
        for chunk_size in self.chunk_size_opts:
            for num_worker in self.worker_opts:
                for resampler in resamplers:
                    print(
                        f"Start testing CHUNK_SIZE={chunk_size}MiB, NUM_WORKER={num_worker}, resampler is {resampler}.") # noqa
                    self.single_loop((chunk_size, num_worker, resampler), area, area_def, resampler_kwargs)
                    i = i + 1

                    if i == self.total_rounds:
                        print("All the tests finished. Generating HTML report.") # noqa
                        html_report(self.work_dir, self.reader_name)
                    else:
                        print(f"ROUND {i} / {self.total_rounds} Completed. Now take a 1-min rest.") # noqa
                        time.sleep(60)


def process_csv(cvs_file):
    """Process result csv and return a dataframe."""
    # Extract information from the filename
    filename = os.path.basename(cvs_file)
    filename = filename.split(".")[0]
    filename_parts = filename.split("_")
    dask_array_chunk_size = int(filename_parts[1].replace("chunk", ""))
    dask_num_workers = int(filename_parts[2].replace("worker", ""))
    omp_num_threads = int(filename_parts[3].replace("thread", ""))
    area = filename_parts[4]
    resampling_alg = filename_parts[5]

    data = pd.read_csv(cvs_file, keep_default_na=False)
    scenes = int(data.loc[0, "Scenes"])
    cpu_thread = psutil.cpu_count(logical=True)

    # Prepare the row dictionary for the new CSV based on filename
    new_row = {
        "Dask Array Chunk Size (MB)": dask_array_chunk_size,
        "Dask Num Workers": dask_num_workers,
        "Omp Num Threads": omp_num_threads,
        "Area": area,
        "Resampling Algorithm": resampling_alg,
        "Time (s)": round(float(data.loc[0, "Process Time"]) / scenes, 2),
        "Avg Memory (GB)": round(data["Memory Usage (Byte)"].mean() / (1024 ** 3), 2),
        "Max Memory (GB)": round(data["Memory Usage (Byte)"].max() / (1024 ** 3), 2),
        "Avg CPU (%)": round(data["CPU Usage (%)"].mean() / cpu_thread, 2),
        "Scenes": scenes,
        "Errors": data.loc[0, "Errors"],
    }

    df = pd.DataFrame([new_row])

    return df


def combined_csv(work_dir, reader_name):
    """Collect all the csv files under work_dir and merge them in to one dataframe."""
    all_dataframes = []
    csvs = glob.glob(f"{work_dir}/{reader_name.replace("_", "")}_chunk*_worker*_thread*_*_*.csv")
    for file in csvs:
        df = process_csv(file)
        all_dataframes.append(df)

    if not all_dataframes:
        return

    combined_df = pd.concat(all_dataframes, ignore_index=True)

    # Sort the DataFrame
    # Make sure "original" area always comes first
    combined_df["sort_priority"] = np.where(combined_df["Area"].str.contains("original"), 0, 1)
    sorted_df = combined_df.sort_values(by=["sort_priority", "Area", "Resampling Algorithm",
                                            "Dask Array Chunk Size (MB)", "Dask Num Workers", "Omp Num Threads"])

    sorted_df.reset_index(drop=True, inplace=True)

    return sorted_df


def draw_hbar(dataframe, colors, title, key_x, key_y):
    """Plot the bar chart by matplotlib."""
    dpi = 100
    fig_width = 1080 / dpi
    num_bars = len(dataframe)
    # Dynamic height
    fig_height = max(600, 100 + 50 * num_bars) / dpi
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.1)

    dataframe.plot.barh(x=key_x, y=key_y, legend=True if isinstance(key_y, list) else False,
                        ax=ax, width=0.5, color=colors, stacked=True if isinstance(key_y, list) else False)
    plt.title(title, fontsize=16)
    plt.ylabel(key_x, fontsize=14)
    plt.xlabel("Memory (GB)" if isinstance(key_y, list) else key_y, fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    if isinstance(key_y, list):
        ax.legend(loc="upper right")
        # Mark the position of physical memory limit
        physical_memory = psutil.virtual_memory().total // (1024 ** 3)
        ax.axvline(x=physical_memory, color="#808080", linestyle="--")
        ax.text(physical_memory + 0.5, 1, "Physical\nMemory\nLimit", color="#808080")
    if "CPU" in ax.get_xlabel():
        ax.set_xlim([0, 100])

    # Data label right to the bar
    cumulative_widths = [0] * len(dataframe)
    for i, container in enumerate(ax.containers):
        for j, bar in enumerate(container):
            width = bar.get_width()
            cumulative_widths[j] = cumulative_widths[j] + width if isinstance(key_y, list) else width
            label_x_pos = cumulative_widths[j] + 0.3

            if i == 0:
                # For "Time", "CPU" and "Avg Memory"
                label_text = str(round(width, 2))
            else:
                # For "Max Memory"
                # Because in the dataframe for graph it's actually the difference between Max and Avg
                # so that we can draw the "stacked" bars correctly.
                # Now we have to restore the value to the real Max when writing the label.
                label_text = str(round(cumulative_widths[j], 2))

            ax.text(label_x_pos, bar.get_y() + bar.get_height() / 2, label_text, va='center')

    svg = BytesIO()
    plt.savefig(svg, format="svg")
    svg = svg.getvalue().decode("utf-8")
    plt.close()

    return svg


def html_report(work_dir, reader_name):
    """Analyze the summary dataframe and produce an HTML report."""
    import dask
    import xarray as xr
    import satpy
    import pyresample
    import pyspectral

    # Get system info
    cpu_core = psutil.cpu_count(logical=False)
    cpu_thread = psutil.cpu_count(logical=True)
    cpu_info = cpuinfo.get_cpu_info()
    cpu_model = cpu_info["brand_raw"]
    memory_info = psutil.virtual_memory().total // (1024 ** 3)
    os_info = platform.platform()

    # Get Python env
    python_version = platform.python_version()
    numpy_version = np.__version__
    dask_version = dask.__version__
    xarray_version = xr.__version__
    satpy_version = satpy.__version__
    pyresample_version = pyresample.__version__
    pyspectral_version = pyspectral.__version__
    psutil_version = psutil.__version__

    df = combined_csv(work_dir, reader_name)
    if df is None:
        print("Test CSV result not found! Or its filename doesn't fit [*_chunk*_worker*_thread*_*_*.csv]")
        return
    # Group the dataframe for report
    df["Group"] = "Area: " + df["Area"] + " - " + "Resampler: " + df["Resampling Algorithm"]
    groups = df["Group"].unique()

    # HTML head
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Satpy Performance Test Report for {reader_name}</title>
        <style>
        table {{
            margin-left: auto;
            margin-right: auto;
        }}
        th, td {{
           max-width: 100px;
        }}
        table, th, td {{
            border: 1px solid black;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            white-space: normal;
            word-wrap: break-word;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        tr:nth-child(even) {{background-color: #f9f9f9}}
        tr:hover {{background-color: #f1f1f1}}    
        </style>
        <style>
        centered-svg {{
            display: block;
            margin-left: auto;
            margin-right: auto;
        }}
        </style>
    </head>
    <body>
        <h1>Satpy Performance Test Report for {reader_name}</h1>
        <h3>Generation Date: UTC {datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")}</h3>
        <h2>1. System Info</h2>
        <h3>1.1 Platform</h3>
        <p>CPU: {cpu_model}, {cpu_core} cores / {cpu_thread} threads in total</p>
        <p>Physical Memory: {memory_info} GB</p>
        <p>OS: {os_info}</p>
        <h3>1.2 Python Environment</h3>
        <p>Python: {python_version}</p>
        <p>Numpy: {numpy_version}</p>
        <p>Dask: {dask_version}</p>
        <p>Xarray: {xarray_version}</p>
        <p>Satpy: {satpy_version}</p>
        <p>Pyresample: {pyresample_version}</p>
        <p>Pyspectral: {pyspectral_version}</p>
        <p>Psutil: {psutil_version}</p>
        <h2>2. Test Results</h2>
    """

    figures = {"Process Time (single scene average)": {"key_y": "Time (s)", "colors": ["#4E79A7"]},
               "Average CPU Usage": {"key_y": "Avg CPU (%)", "colors": ["#F28E2B"]},
               "Memory Usage": {"key_y": ["Avg Memory (GB)", "Max Memory (GB)"], "colors": ["#59A14F", "#EDC948"]}}

    for group in groups:
        group_df = df[df["Group"] == group]
        # Drop unnecessary column
        group_df_table = group_df.drop(["Group", "Area", "Resampling Algorithm", "sort_priority"],
                                       axis=1, inplace=False)

        group_df_graph = group_df.copy()
        # Build a new column containing value group to make it easier for plotting the chart
        group_df_graph["Chunk size - Num workers - Num Threads"] = (
                group_df_graph["Dask Array Chunk Size (MB)"].astype(str) + " - " +
                group_df_graph["Dask Num Workers"].astype(str) + " - " +
                group_df_graph["Omp Num Threads"].astype(str))
        group_df_graph = group_df_graph.sort_values(by=["Dask Array Chunk Size (MB)", "Dask Num Workers",
                                                        "Omp Num Threads"], ascending=False)
        # For stacked bar
        group_df_graph["Max Memory (GB)"] = group_df_graph["Max Memory (GB)"] - group_df_graph["Avg Memory (GB)"]
        # Replace all the error rows with 0 so the chart will be significant in these rows
        group_df_graph.loc[group_df_graph["Errors"] != "N/A", ["Time (s)", "Avg CPU (%)",
                                                               "Avg Memory (GB)", "Max Memory (GB)"]] = 0

        group_html = group_df_table.to_html(index=False)
        html_content += f"""
        <h3>2.{groups.tolist().index(group) + 1} {group}</h3>
        <h4>2.{groups.tolist().index(group) + 1}.1 Table</h4>
        {group_html}
        <h4>2.{groups.tolist().index(group) + 1}.2 Charts</h4>
        """

        # Plot three charts: time, cpu and mem
        for figure in figures.keys():
            svg_bar = draw_hbar(group_df_graph, figures[figure]["colors"], figure,
                                "Chunk size - Num workers - Num Threads", figures[figure]["key_y"])
            html_content += f"""
                <div id="{groups.tolist().index(group) + 1}_chart{list(figures.keys()).index(figure) + 1}"
                class="centered-svg">
                    {svg_bar}
                </div>
                """

    # Finish HTML report
    html_content += """
    </body>
    </html>
    """

    # Save it to disk
    with open(f"{work_dir}/satpy_performance_report.html", "w", encoding="utf-8") as f:
        f.write(html_content)