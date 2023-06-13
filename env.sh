# conda activate torch
conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia -y
conda install -c conda-forge ray-tune -y
conda install pyyaml numpy tqdm scipy dill flask -y
conda install -c conda-forge matplotlib tensorboard -y