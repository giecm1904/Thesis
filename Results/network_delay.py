import pathlib
import pandas as pd
import iso8601.iso8601
import datetime
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



data_dir = pathlib.Path("Multi_Region")
crh_dir = data_dir.joinpath("cr-h")
cro_dir = data_dir.joinpath("cr-o")
mcf_dir = data_dir.joinpath("mcf")
nep_dir = data_dir.joinpath("neptune")
vsvbp_dir = data_dir.joinpath("vsvbp")

run_dirs = [nep_dir, vsvbp_dir, mcf_dir, cro_dir, crh_dir]

runs = list(range(3))

functions = [
    "compression",
    "dynamic-html",
    "graph-bfs",
    "graph-mst",
    "thumbnailer",
]

run = 4

result = makehash()

for run_dir, function, run_n in list(itertools.product(run_dirs, functions, runs)):
    req_df = pd.read_csv(run_dir.joinpath(f"{function}/{function}_{run_n}_proxy_metric.csv"))
    req_df['timestamp'] = req_df['timestamp'].map(lambda x: iso8601.parse_date(x))
    req_df['timestamp'] = req_df['timestamp'].map(round)
    min_time = min(req_df['timestamp'])
    max_time = max(req_df['timestamp'])
    req_df['timestamp'] = req_df['timestamp'] - min_time
    req_df = req_df.sort_values('timestamp')

    res_df = pd.read_csv(run_dir.joinpath(f"{function}/{function}_{run_n}_pod_log.csv"))
    res_df = res_df[res_df["container_name"].map(lambda x: function in x)]
    res_df['timestamp'] = res_df['timestamp'].map(lambda x: iso8601.parse_date(x))
    res_df['timestamp'] = res_df['timestamp'].map(round)
    res_df = res_df[res_df['timestamp'] < max_time]
    res_df = res_df[res_df['timestamp'] > min_time]
    res_df['timestamp'] = res_df['timestamp'] - min_time
    res_df = res_df.sort_values('timestamp')

    if run_dir is crh_dir:
        res_df = res_df[~res_df['pod_address'].isna()]
    if run_dir is cro_dir:
        res_df['cpu'] = res_df['cpu'] + np.random.normal(0, 0.5, len(res_df['cpu']))

    cpu = res_df.groupby("timestamp").sum()['cpu'].rolling(200, win_type="triang", min_periods=20).mean()
    rt = req_df.groupby("timestamp").mean()['latency'].rolling(200, win_type="triang", min_periods=20).mean()

    ms = req_df['timestamp'].dt.total_seconds()-res_df['timestamp'].dt.total_seconds()
    ms_f = [x for x in ms if not np.isnan(x)]
    ms_f = [0 if x < 0 else x for x in ms_f]

    result[run_dir][function]['cpu'][run_n] = cpu.to_numpy()
    result[run_dir][function]['rt'][run_n] = rt.to_numpy()
    result[run_dir][function]['delay'][run_n] = ms_f

    res_df = res_df.merge(pd.DataFrame({'timestamp':req_df['timestamp'].unique()}), on='timestamp', how="right").fillna(method='bfill').fillna(method='ffill')[:]

    p_req = pd.DataFrame({'timestamp': req_df['timestamp'],
                          'ms_timestamp': req_df['latency']})

    p_res = pd.DataFrame({'timestamp': res_df['timestamp'],
                          'ms_timestamp': res_df['response_time']})

    p_req_t = p_req.groupby("timestamp").mean()['ms_timestamp'].rolling(200, win_type="triang", min_periods=20).mean()
    p_req_t = p_req_t.dropna()
    p_res_t = p_res.groupby("timestamp").mean()['ms_timestamp'].rolling(200, win_type="triang", min_periods=20).mean()
    p_res_t = p_res_t.dropna()
    tiempo_final = p_req['ms_timestamp'].subtract(p_res['ms_timestamp'])
    tiempo_final = tiempo_final.dropna()

    net_delay = (p_req_t - p_res_t).clip(0).mean()

    print(f"{run_dir}, {function}, {run_n}, net delay = {net_delay}")