#!/usr/bin/env python3
import os
import netCDF4 as nc
import numpy as onp
import datetime
import calendar
import paramiko
import matplotlib.pyplot as plt
import multiprocessing

out_dir = "SPCAM5"

data_dir = '/scratch1/07088/tg863871/CESM2_case/CESM2_NN2_pelayout01_ens_07/CESM2_NN2_pelayout01_ens_07/CESM2_NN2_pelayout01_ens_07.cam.h1.'
ucsd_dir = "/data/allen/climate_data/"



class SPCAM5_Frontera_to_npy:
    def __init__(self, start_date, end_date, delta):
        self.DT = 30*60
        self.start_date = start_date
        self.end_date = end_date
        self.delta = delta

        self.extensions = ["-00000.nc",  "-28800.nc", "-57600.nc"]


    def process_data(self, data):
        # try:
        # ssh for each multiprocessing
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect('-', username='-', password='-', allow_agent=False)
        sftp = ssh.open_sftp()

        date_str, n_days = data
        up_to_date = datetime.datetime.strptime(date_str, '%Y-%m-%d')
        if self.delta.days == 1:
            n = 1
        else:
            n = n_days
        for i in range(-n+1, 1):
            curr_date = up_to_date + datetime.timedelta(days=i)
            curr_date_str = curr_date.strftime("%Y-%m-%d")

            inputs, outputs = None, None
            for ext in self.extensions:
                ds = nc.Dataset(data_dir+curr_date_str+ext)

                self.lat = ds.dimensions['lat'].size
                self.time = ds.dimensions['time'].size
                self.lon = ds.dimensions['lon'].size
                self.lev = ds.dimensions['lev'].size

                np_inputs, np_outputs = self.format_NC_dataset(ds)

                if inputs is None:
                    inputs = np_inputs
                    outputs = np_outputs
                else:
                    inputs = onp.concatenate((inputs, np_inputs), axis=0)
                    outputs = onp.concatenate(
                        (outputs, np_outputs), axis=0)
        print("Processed", data_dir+curr_date_str+ext)

        inp_path = f'{out_dir}/inputs_{date_str}.npy' 
        out_path = f'{out_dir}/outputs_{date_str}.npy'

        # Index out solar insolation to match CAM5 dataset
        inputs = inputs.reshape(-1, 96, 144, 108)[1::2, :].reshape(-1, 108)
        outputs = outputs.reshape(-1, 96, 144, 112)[1::2, :].reshape(-1, 112)

        onp.save(inp_path, inputs)
        onp.save(out_path, outputs)
    
        sftp.put(inp_path, os.path.join(ucsd_dir, inp_path))
        sftp.put(out_path, os.path.join(ucsd_dir, out_path))

        os.remove(inp_path)
        os.remove(out_path)
        print(f"SSH {inp_path}", 'train data shape: ', inputs.shape, outputs.shape)

        sftp.close()
        ssh.close()
        # except Exception as e:
        #     print(e)
        #     print("probably a mistake in the date from source")
                

    def get_all_dates(self, start_date, end_date, delta):
        """Returns a list of all dates between start_date and end_date, inclusive.
        Each element is a list of [end_date_str, days_in_this_file]."""
        curr_date = start_date
        data = []
        while curr_date <= end_date:
            display_date = curr_date + delta - datetime.timedelta(days=1)
            # If next step is over the end date, then just use the end date.
            if display_date > end_date:
                display_date = end_date
                data.append([display_date.strftime("%Y-%m-%d"),
                            (display_date - curr_date).days + 1])
                break
            # Elif next step is over the end of the month, then use the end of the month.
            if display_date.day > calendar.monthrange(display_date.year, display_date.month)[1]:
                display_date = datetime.datetime(display_date.year, display_date.month, calendar.monthrange(
                    display_date.year, display_date.month)[1])
                data.append([display_date.strftime("%Y-%m-%d"),
                            (display_date - curr_date).days + 1])
                curr_date = display_date + datetime.timedelta(days=1)
            else:
                data.append([display_date.strftime("%Y-%m-%d"), delta.days])
                curr_date = curr_date + delta
        return data

    def format_NC_dataset(self, ds):
        lev = self.lev
        time = self.time
        lat = self.lat
        lon = self.lon
        DT = self.DT

        TBP = onp.transpose(ds.variables['TBP'], axes=(1, 0, 2, 3))
        QBP = onp.transpose(ds.variables['QBP'], axes=(1, 0, 2, 3))
        CLDLIQBP = onp.transpose(ds.variables['CLDLIQBP'], axes=(1, 0, 2, 3))
        CLDICEBP = onp.transpose(ds.variables['CLDICEBP'], axes=(1, 0, 2, 3))

        TBP = onp.reshape(TBP, (lev, time*lat*lon))
        QBP = onp.reshape(QBP, (lev, time*lat*lon))
        CLDLIQBP = onp.reshape(CLDLIQBP, (lev, time*lat*lon))
        CLDICEBP = onp.reshape(CLDICEBP, (lev, time*lat*lon))

        PS = onp.reshape(ds.variables['PS'], (1, time*lat*lon))
        SOLIN = onp.reshape(ds.variables['SOLIN'], (1, time*lat*lon))
        SHFLX = onp.reshape(ds.variables['SHFLX'], (1, time*lat*lon))
        LHFLX = onp.reshape(ds.variables['LHFLX'], (1, time*lat*lon))

        inputs = onp.concatenate(
            (TBP, QBP, CLDLIQBP, CLDICEBP, PS, SOLIN, SHFLX, LHFLX)).T

        TBC = onp.transpose(ds.variables['TBC'], axes=(1, 0, 2, 3))
        TBC = onp.reshape(TBC, (lev, time*lat*lon))
        TBCTEND = (TBC-TBP)/DT

        QBC = onp.transpose(ds.variables['QBC'], axes=(1, 0, 2, 3))
        QBC = onp.reshape(QBC, (lev, time*lat*lon))
        QBCTEND = (QBC-QBP)/DT

        CLDLIQBC = onp.transpose(ds.variables['CLDLIQBC'], axes=(1, 0, 2, 3))
        CLDLIQBC = onp.reshape(CLDLIQBC, (lev, time*lat*lon))
        CLDLIQBCTEND = (CLDLIQBC-CLDLIQBP)/DT

        CLDICEBC = onp.transpose(ds.variables['CLDICEBC'], axes=(1, 0, 2, 3))
        CLDICEBC = onp.reshape(CLDICEBC, (lev, time*lat*lon))
        CLDICEBCTEND = (CLDICEBC-CLDICEBP)/DT

        NN2L_FLWDS = onp.reshape(ds.variables['NN2L_FLWDS'], (1, time*lat*lon))
        NN2L_NETSW = onp.reshape(ds.variables['NN2L_NETSW'], (1, time*lat*lon))
        NN2L_PRECC = onp.reshape(ds.variables['NN2L_PRECC'], (1, time*lat*lon))
        NN2L_PRECSC = onp.reshape(
            ds.variables['NN2L_PRECSC'], (1, time*lat*lon))
        NN2L_SOLL = onp.reshape(ds.variables['NN2L_SOLL'], (1, time*lat*lon))
        NN2L_SOLLD = onp.reshape(ds.variables['NN2L_SOLLD'], (1, time*lat*lon))
        NN2L_SOLS = onp.reshape(ds.variables['NN2L_SOLS'], (1, time*lat*lon))
        NN2L_SOLSD = onp.reshape(ds.variables['NN2L_SOLSD'], (1, time*lat*lon))
        outputs = onp.concatenate((TBCTEND, QBCTEND, CLDLIQBCTEND, CLDICEBCTEND,
                                   NN2L_FLWDS, NN2L_NETSW, NN2L_PRECC, NN2L_PRECSC,
                                   NN2L_SOLL, NN2L_SOLLD, NN2L_SOLS, NN2L_SOLSD)).T

        return inputs, outputs


if __name__ == "__main__":
    # start_date = datetime.datetime(2004, 4, 1)
    start_date = datetime.datetime(2004, 5, 11)
    end_date = datetime.datetime(2005, 3, 28)
    delta = datetime.timedelta(days=1)
    convertor = SPCAM5_Frontera_to_npy(start_date, end_date, delta)

    # convertor.process_data()
    date_lists = convertor.get_all_dates(convertor.start_date, convertor.end_date, convertor.delta)

    # For each data in range
    with multiprocessing.Pool(processes=10) as pool:
        pool.map(convertor.process_data, date_lists)

