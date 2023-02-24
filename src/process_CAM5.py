import glob
import os
from tqdm import tqdm
import datetime
import numpy as np
import warnings
import dill
from lib.utils import MinMaxScaler, StandardScaler, make_dir, sort_fn, sort_batch_fn
import paramiko

# Send CAM5 data to ucsd server through ssh

ucsd_dir = "/data/allen/climate_data/"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('-', username='-', password='-', allow_agent=False)
sftp = ssh.open_sftp()

def save_object(obj, filename):
    dill.dump(obj, file=open(filename, "wb"))

def process_data():
    x_minmax_scaler = MinMaxScaler()
    y_minmax_scaler = MinMaxScaler()

    x_paths = sorted(glob.glob(os.path.join(f'../CAM5/inputs_*')), key=sort_fn)[113:]
    y_paths = sorted(glob.glob(os.path.join(f'../CAM5/outputs_*')), key=sort_fn)[113:]

    out_dir = "CAM5"
    make_dir(out_dir)

    for i, (x_f, y_f) in tqdm(enumerate(zip(x_paths, y_paths)), total=len(y_paths)):
        x_time, y_time = x_f[-14:-4], y_f[-14:-4]

        if x_time != y_time:
            print('Wrong')
            print(x_time, y_time)
            break
        start_time = datetime.datetime.strptime(x_time, "%Y_%m_%d")

        days = start_time.day % 6
        if days == 0:
            days = 6
        start_time = start_time - datetime.timedelta(days=days-1)

        x = np.load(x_f)  # , mmap_mode='r')
        y = np.load(y_f)  # , mmap_mode='r')
        x_minmax_scaler.fit(x)
        y_minmax_scaler.fit(y)

        # Ignore last one
        x = x.reshape(-1, 96, 144, 109)[..., :-1]
        x = x.reshape(-1, 108)

        # Write a file for each day.
        n = x.shape[0] // days
        for i in range(days):
            title = start_time.strftime("%Y-%m-%d")
            inp_path = f"{out_dir}/inputs_{title}.npy"
            out_path = f"{out_dir}/outputs_{title}.npy"

            np.save(inp_path, x[i*n:(i+1)*n])
            np.save(out_path, y[i*n:(i+1)*n])
            sftp.put(inp_path, os.path.join(ucsd_dir, inp_path))
            sftp.put(out_path, os.path.join(ucsd_dir, out_path))

            os.remove(inp_path)
            os.remove(out_path)

            print(
                f"SSH {inp_path} train data shape: {n} -- {x.shape[-1]}, {y.shape[-1]}")
            start_time += datetime.timedelta(days=1)

        print(f'Processed Original File {x_time}, shapes: {x.shape} {y.shape}')

    save_object(x_minmax_scaler, f"{out_dir}/x_minmax_scaler.pkl")
    save_object(y_minmax_scaler, f"{out_dir}/y_minmax_scaler.pkl")


process_data()
sftp.close()
ssh.close()