import numpy as np
import glob as glob
import dill
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

def sort_fn(filename):
    date_string = filename[-14:-4]
    datetime_object = datetime.datetime.strptime(date_string, "%Y-%m-%d")
    return datetime_object

x_scaler_minmax = dill.load(open("../scalers/x_SPCAM5_minmax_scaler.dill", 'rb'))
y_scaler_minmax = dill.load(open(f"../scalers/y_SPCAM5_minmax_scaler.dill", 'rb'))
x_scaler_minmax.min = x_scaler_minmax.min * 0
x_scaler_minmax.max = np.abs(x_scaler_minmax.max)
y_scaler_minmax.min = y_scaler_minmax.min[:26] * 0
y_scaler_minmax.max = np.abs(y_scaler_minmax.max[:26])

x_data = sorted(glob.glob(f"../../SPCAM5/inputs_*"), key=sort_fn)
y_data = sorted(glob.glob(f"../../SPCAM5/outputs_*"), key=sort_fn)

split_n = int(365*0.8)
x_train = x_data[:split_n]
y_train = y_data[:split_n]
x_valid = x_data[split_n:365]
y_valid = y_data[split_n:365]
x_test = x_data[365:]
y_test = y_data[365:]

x_train_df = pd.DataFrame()
for x_paths in tqdm(x_train, total=len(x_train)):
    x = np.load(x_paths)
    loadDf = pd.DataFrame([x.mean(axis=0), x.min(axis=0), x.max(axis=0)])
    x_train_df = pd.concat([x_train_df, loadDf], axis=0)
x_train_df = x_train_df.reset_index(drop=True)

y_train_df = pd.DataFrame()
for y_paths in tqdm(y_train, total=len(y_train)):
    y = np.load(y_paths)[:, :26]
    loadDf = pd.DataFrame([y.mean(axis=0), y.min(axis=0), y.max(axis=0)])
    y_train_df = pd.concat([y_train_df, loadDf], axis=0)
y_train_df = y_train_df.reset_index(drop=True)

x_valid_df = pd.DataFrame()
for x_paths in tqdm(x_valid, total=len(x_valid)):
    x = np.load(x_paths)
    x = x_scaler_minmax.transform(x)
    loadDf = pd.DataFrame([x.mean(axis=0), x.min(axis=0), x.max(axis=0)])
    x_valid_df = pd.concat([x_valid_df, loadDf], axis=0)
x_valid_df = x_valid_df.reset_index(drop=True)

y_valid_df = pd.DataFrame()
for y_paths in tqdm(y_valid, total=len(y_valid)):
    y = np.load(y_paths)[:, :26]
    loadDf = pd.DataFrame([y.mean(axis=0), y.min(axis=0), y.max(axis=0)])
    y_valid_df = pd.concat([y_valid_df, loadDf], axis=0)
y_valid_df = y_valid_df.reset_index(drop=True)

x_test_df = pd.DataFrame()
for x_paths in tqdm(x_test, total=len(x_test)):
    x = np.load(x_paths)
    loadDf = pd.DataFrame([x.mean(axis=0), x.min(axis=0), x.max(axis=0)])
    x_test_df = pd.concat([x_test_df, loadDf], axis=0)
x_test_df = x_test_df.reset_index(drop=True)

y_test_df = pd.DataFrame()
for y_paths in tqdm(y_test, total=len(y_test)):
    y = np.load(y_paths)[:, :26]
    loadDf = pd.DataFrame([y.mean(axis=0), y.min(axis=0), y.max(axis=0)])
    y_test_df = pd.concat([y_test_df, loadDf], axis=0)
y_test_df = y_test_df.reset_index(drop=True)

n = 9
for v in range(108 // n):
    sns.violinplot(data=x_train_df.iloc[:, v*n:(v+1)*n])
    plt.savefig(f"plots/x_train_{v*n}_violion.png")
    plt.close()
    sns.violinplot(data=x_valid_df.iloc[:, v*n:(v+1)*n])
    plt.savefig(f"plots/x_valid_{v*n}_violion.png")
    plt.close()
    sns.violinplot(data=x_test_df.iloc[:, v*n:(v+1)*n])
    plt.savefig(f"plots/x_test_{v*n}_violion.png")
    plt.close()

n = 4
for v in range(26//n+1):
    sns.violinplot(data=y_train_df.iloc[:, v*n:(v+1)*n])
    plt.savefig(f"plots/y_train_{v*n}_violion.png")
    plt.close()
    sns.violinplot(data=y_valid_df.iloc[:, v*n:(v+1)*n])
    plt.savefig(f"plots/y_valid_{v*n}_violion.png")
    plt.close()
    sns.violinplot(data=y_test_df.iloc[:, v*n:(v+1)*n])
    plt.savefig(f"plots/y_test_{v*n}_violion.png")
    plt.close()


x_train_df = pd.DataFrame()
for x_paths in tqdm(x_train, total=len(x_train)):
    x = np.load(x_paths)
    x = x_scaler_minmax.transform(x)
    loadDf = pd.DataFrame([x.mean(axis=0), x.min(axis=0), x.max(axis=0)])
    x_train_df = pd.concat([x_train_df, loadDf], axis=0)
x_train_df = x_train_df.reset_index(drop=True)

y_train_df = pd.DataFrame()
for y_paths in tqdm(x_train, total=len(x_train)):
    y = np.load(y_paths)[:, :26]
    y = y_scaler_minmax.transform(y)
    loadDf = pd.DataFrame([y.mean(axis=0), y.min(axis=0), y.max(axis=0)])
    y_train_df = pd.concat([y_train_df, loadDf], axis=0)
y_train_df = y_train_df.reset_index(drop=True)

x_valid_df = pd.DataFrame()
for x_paths in tqdm(x_valid, total=len(x_valid)):
    x = np.load(x_paths)
    x = x_scaler_minmax.transform(x)
    loadDf = pd.DataFrame([x.mean(axis=0), x.min(axis=0), x.max(axis=0)])
    x_valid_df = pd.concat([x_valid_df, loadDf], axis=0)
x_valid_df = x_valid_df.reset_index(drop=True)

y_valid_df = pd.DataFrame()
for y_paths in tqdm(y_valid, total=len(y_valid)):
    y = np.load(y_paths)[:, :26]
    y = y_scaler_minmax.transform(y)
    loadDf = pd.DataFrame([y.mean(axis=0), y.min(axis=0), y.max(axis=0)])
    y_valid_df = pd.concat([y_valid_df, loadDf], axis=0)
y_valid_df = y_valid_df.reset_index(drop=True)

x_test_df = pd.DataFrame()
for x_paths in tqdm(x_test, total=len(x_test)):
    x = np.load(x_paths)
    x = x_scaler_minmax.transform(x)
    loadDf = pd.DataFrame([x.mean(axis=0), x.min(axis=0), x.max(axis=0)])
    x_test_df = pd.concat([x_test_df, loadDf], axis=0)
x_test_df = x_test_df.reset_index(drop=True)

y_test_df = pd.DataFrame()
for y_paths in tqdm(y_test, total=len(y_test)):
    y = np.load(y_paths)[:, :26]
    y = y_scaler_minmax.transform(y)
    loadDf = pd.DataFrame([y.mean(axis=0), y.min(axis=0), y.max(axis=0)])
    y_test_df = pd.concat([y_test_df, loadDf], axis=0)
y_test_df = y_test_df.reset_index(drop=True)


n = 9
for v in range(108 // n):
    sns.violinplot(data=x_train_df.iloc[:, v*n:(v+1)*n])
    plt.savefig(f"plots/x_train_scaled_{v*n}_violion.png")
    plt.close()
    sns.violinplot(data=x_valid_df.iloc[:, v*n:(v+1)*n])
    plt.savefig(f"plots/x_valid_scaled_{v*n}_violion.png")
    plt.close()
    sns.violinplot(data=x_test_df.iloc[:, v*n:(v+1)*n])
    plt.savefig(f"plots/x_test_scaled_{v*n}_violion.png")
    plt.close()

n = 4
for v in range(26//n+1):
    sns.violinplot(data=y_train_df.iloc[:, v*n:(v+1)*n])
    plt.savefig(f"plots/y_train_scaled_{v*n}_violion.png")
    plt.close()
    sns.violinplot(data=y_valid_df.iloc[:, v*n:(v+1)*n])
    plt.savefig(f"plots/y_valid_scaled_{v*n}_violion.png")
    plt.close()
    sns.violinplot(data=y_test_df.iloc[:, v*n:(v+1)*n])
    plt.savefig(f"plots/y_test_scaled_{v*n}_violion.png")
    plt.close()