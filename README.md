# climate_processes

## Multi-Fidelity Hierarchical Attentive Neural Process

### Reference

Multi-Fidelity Hierarchical Neural Processes

https://github.com/Rose-STL-Lab/Hierarchical-Neural-Processes

Attentive Nueral Processes

https://github.com/soobinseo/Attentive-Neural-Process

Scientific Climate Model Data Description

https://ncar.github.io/CAM/doc/build/html/cam6_scientific_guide/

## Develpoment Notes

- Currently, train.py in sf_attention_process is the best training code. Will clean up the rest of the folders when compiling results toward the end.

### Packages

Conda environment was used so pip's requirements.txt won't be helpful.
Just install as you go along. Most standard packages are used.

### Hardware

\- _Need around 400 Gb (high_data) + 320 Gb (low_data) of storage for full dataset as well as deleting each original data after processing it into smaller batches._

GPU ~ 1280MiB for SFNP.

## How to Run

## No need to git clone. Just use docker. If not, just go straight to src/run.py

1. <code>```docker pull kaimj/climate_process```</code>
2. <code>```docker run -it --rm kaimj/climate_process```</code>


<hr>

1. Download the private low-fidelity and high-fidelity data into data/high_data and data/low_data.
2. <strong>CD into src and run process_data.py</strong> to split the big files into smaller chunks. process_data.py saves each time steps or chunks into batch_high_data and batch_low_data in order to ensure randomized sampling during training.
3. <strong>CD into SFNP/MFHNP/SFANP</strong>.
4. <strong>Run train.py.</strong> \*You can, run tensorboard
5. Ctrl+C to interrupt training. tqdm in terminal + logging enabled.
6. To resume training, go to <strong>config.yaml</strong> and changeload_checkpoint: folder_name from logs/{folder_name}.

<hr>

## Data Notes

1. Low fidelity data "2003_02_12", "2003_07_06", "2003_11_18" files were corrupted. Corresponding high fidelity data were removed.
2. One of the variable has been corrupted for every other index. process_data removes every other timestep.
3. Input and output data were normalized with minmax scaling 0 to 1 based on the given data range. Adjustable in process_data/save_scaler.
   - Standard scaler resulted in rmse of negative values. Minmax was more stable and better for comparison.
   - \*Note 1/22. Issues with normalizing with given data range. MinMax across 1 year was used)
4. Fully randomized sampling required loading one array from each multiple files, which took too long. Decided to load one file at a time, so batch_size = 1 has 96*144 * (feature) points for every 1 hour.

<br/>
Assuming all the data has been downloaded and processed, Use these indexes from the whole dataset to batch the files.

| date                    | # files | idxs         |
| ----------------------- | ------- | ------------ |
| 2003_01_01 ~ 2004_09_22 | 14184   | All          |
| 2003_09_13 ~ 2004_01_13 | 4176    | -8658:-4176  |
| 2003_09_13 ~ 2004_09_13 | 8352    | -8658:-216   |
| 2003_04_06 ~ 2004_03_31 | 8208    | 1872 : 10080 |
| 2004_04_01 ~ 2004_06_30 | 2160    | 10080: 12240 |
| 2004_06_30 ~ 2004_09_22 | 3888    | 12240: 14184 |

## Table of Experiments

Random seed set to n.

```python
seed = n
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
```

<hr>

20% of 1 year

| Model                 | epoch | Split | MSE      | MAE (scaled) | MAE (non-scaled) | NRMSE (scaled) | NRMSE (non-scaled) |
| --------------------- | ----- | ----- | -------- | ------------ | ---------------- | -------------- | ------------------ |
| Next-Step             | -     | Train | 0.000220 | 0.002866     | 0.679104         | 0.000178       | 0.303254           |
| -                     | -     | Valid | 0.000211 | 0.002843     | 0.626795         | 0.000183       | 0.254427           |
| Single-Fidelity       | 282   | Train | 0.002018 | 0.012892     | 3.337976         | 12.067160      | 74290.2945         |
| 114064 # parameters   | -     | Valid | 0.001929 | 0.012650     | 3.198386         | 8.696346       | 52690.9533         |
| Multi (nested)        | 171   | Train | 0.002021 | -            | 3.339059         | 1.480341       | 12539.474          |
| 243378 # parameters   | -     | Valid | 0.001928 | -            | 3.219164         | 5.389482       | 5152.517653        |
| Multi (nonnested)     | 11    | Train | 0.002029 | -            | 3.341579         | 14.255439      | 36424.380684       |
| 243378 # parameters   | -     | Valid | 0.001928 | -            | 3.222764         | 15.417424      | 60962.398692       |
| Multi-Head Attention  | 48    | Train | 0.059827 | -            | 1.901673         | 6.632925       | 229219.039199      |
| 13133960 # parameters | -     | Valid | 0.012957 | -            | 2.165103         | 1.201251       | 414917.831952      |
| Transformer Decoder   | 53    | Train | 0.000084 | -            | 0.416977         | 0.130012       | 5564.482205        |
| 3216392 # parameters  | 53    | Valid | 0.000233 | -            | 0.701086         | 0.093324       | 2403.025430        |

<hr>

Experiment results:

## Experiment Notes

1. Non-nested training scheme:
   "=" is used and "\_" is not used
   - [==== ___________] First 20% of high fidelity data used for training. Last 20% used for validation.
   - [____ ======= ___] 60% of non-overlapping low fidelity data used. Last 20% used for validation
2. Nested training scheme:
   - [==== _________] First 20% of high fidelity data used for training. Last 20% used for validation.
   - [========== ___] initial 60% of low-fidelity data used. Last 20% used for validation
   - 20 % of high fidelity scenarios are matched with the same low fidelity pair during training.
3. Context and target sets are based on 1 time step - 144 points at 1 hour interval.
4. Possible bottlenecks in dataloader :File processing and normalization.
5. For Multi-Fidelity, one epoch = 3 passes on high-fidelity dataset because nesting/non-nesting lead to multiple passes on high-fidelity dataset.
6. Transformer-Decoder
   - attention_layers: at least 6 used.
   - num_heds: at least 8 used.