import os
import glob

import paramiko
import matplotlib.pyplot as plt
import multiprocessing

out_dir = "SPCAM5"

data_dir = 'data/SPCAM5'
ucsd_dir = "/data/allen/climate_data/SPCAM5"

os.makedirs(data_dir, exist_ok=True)
def process_data(path):
    try:
        if os.path.exists(os.path.join(data_dir, path)):
            print(f"SSH path: {path} exists")
            return
        # ssh for each multiprocessing
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect('-', username='-', password='-', allow_agent=False)
        sftp = ssh.open_sftp()

        sftp.get(os.path.join(ucsd_dir, path), os.path.join(data_dir, path))

        print(f"SSH path: {path} done")
    except Exception as e:
        print(e)

        print(f"SSH path: {path} failed")
    finally:
        try:
            sftp.close()
            ssh.close()
        except:
            return
        

if __name__ == "__main__":
    paths = sorted(os.listdir('data/CAM5/'))

    # For each data in range
    with multiprocessing.Pool(processes=10) as pool:
        pool.map(process_data, paths)