import pathlib
import pandas as pd
import iso8601.iso8601
import datetime
import seaborn
import matplotlib.pyplot as plt
import collections
import tqdm
import itertools
import warnings
import numpy as np

warnings.simplefilter(action='ignore', category=FutureWarning)


def round(t):
    t = datetime.datetime(t.year, t.month, t.day, t.hour, t.minute, t.second)
    return t


def makehash():
    return collections.defaultdict(makehash)


data_dir = pathlib.Path("data")
hpa_dir = data_dir.joinpath("hpa")
vpa_dir = data_dir.joinpath("vpa")
nep_dir = data_dir.joinpath("neptune")

run_dirs = [nep_dir, hpa_dir, vpa_dir]

runs = list(range(5))

functions = [
    "compression",
    "dynamic-html",
    "graph-bfs",
    "graph-mst",
    "pagerank",
    "thumbnailer",
    "video-processing"
]

approaches = [
    "hpa",
    "ex_nep",
    "nep"
]

run_dir = hpa_dir
app = "2"
run = 0
desired = False
approach = "hpa"

for approach in ["hpa", "nep", "ex_nep"]:
    for app in ["1", "2"]:
        for desired in [True, False]:

            if approach == "hpa":
                res_df = pd.read_csv(data_dir.joinpath(f"multi/60min/{approach}/multi{app}/multi_app{app}_{run}_pod_log.csv"))
                res_df['timestamp'] = res_df['timestamp'].map(lambda x: iso8601.parse_date(x))
                res_df['timestamp'] = res_df['timestamp'].map(round)
                res_df = res_df.sort_values('timestamp')
                res_df['container_name'] = res_df['container_name'].map(lambda x: x.replace("sebs-", ""))
                if not desired:
                    res_df = res_df[~res_df['pod_address'].isna()]
                res_df['cpu'] = res_df['cpu'] + 0.1
                res_df = res_df.groupby(["timestamp", "container_name"]).sum().reset_index()
            else:
                if desired:
                    res_df = pd.read_csv(data_dir.joinpath(f"multi/60min/{approach}/multi{app}/multi_app{app}_{run}_podscale_log.csv"))
                    res_df['timestamp'] = res_df['timestamp'].map(lambda x: iso8601.parse_date(x))
                    res_df['timestamp'] = res_df['timestamp'].map(round)
                    res_df = res_df.sort_values('timestamp')
                    res_df['cpu'] = res_df['cpu_capped']
                    res_df['cpu'] = res_df['cpu'].map(lambda x: max(x, 1))
                    res_df['cpu'] = res_df['cpu'] + 0.1
                    res_df['container_name'] = res_df['pod_name'].map(lambda x: "-".join(x.split("-")[1:-1]))
                else:
                    res_df = pd.read_csv(data_dir.joinpath(f"multi/60min/{approach}/multi{app}/multi_app{app}_{run}_pod_log.csv"))
                    res_df['timestamp'] = res_df['timestamp'].map(lambda x: iso8601.parse_date(x))
                    res_df['timestamp'] = res_df['timestamp'].map(round)
                    res_df = res_df.sort_values('timestamp')
                    res_df['cpu'] = res_df['cpu'] + 0.1
                    res_df['container_name'] = res_df['container_name'].map(lambda x: x.replace("sebs-", ""))
                    res_df = res_df.groupby(["timestamp", "container_name"]).sum().reset_index()
            min_time = min(res_df['timestamp'])
            res_df['timestamp'] = res_df['timestamp'] - min_time
            res_df = res_df.sort_values('timestamp')

            unique_timestamps = res_df['timestamp'].unique()
            unique_functions = res_df['container_name'].unique()

            sub_df = {}

            for f in unique_functions:
                # x['cpu'] = x['cpu'].rolling(15, win_type="triang", min_periods=15).mean()
                sub_df[f] = res_df[res_df['container_name'] == f]['cpu'].rolling(15, win_type="triang", min_periods=15).mean()

            plt.figure(figsize=(3.3,3.5))
            seaborn.set_theme(style="whitegrid")
            plt.rc('font', size=12)
            plt.rc('legend', fontsize=12)
            plt.rc('axes', titlesize=12)
            plt.rc('axes', labelsize=12)
            plt.ylabel("Allocations (cores)")
            plt.xlabel("Time (s)")
            if app == "2":
                plt.stackplot(unique_timestamps.astype(np.float64) * 1e-9, sub_df['graph-mst'], sub_df['compression'], sub_df['video-processing'], labels=['graph-mst',  'compression', 'video-processing'])
                plt.legend(loc="upper left")
            elif app == "1":
                plt.stackplot(unique_timestamps.astype(np.float64) * 1e-9, sub_df['graph-bfs'], sub_df['dynamic-html'], sub_df['thumbnailer'], sub_df['video-processing'], labels=['graph-bfs', 'dynamic-html', 'thumbnailer', 'video-processing'])
                plt.legend(loc="upper left")
            plt.ylim([0, 65])
            plt.xlim([0, 4000])
            plt.tight_layout(pad=0.2)
            plt.savefig(f"rq5-{approach}-{desired}-{app}.pdf")
            plt.show()
