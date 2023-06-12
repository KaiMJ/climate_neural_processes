import matplotlib.pyplot as plt
from lib.eval import *
import numpy as np
import glob
import os

log_dir = "logs/lr_0.000025_seed0_lr2.5e-05_bs_1_Tue May 16 17:25:13 2023"

evaluator = Evaluator(log_dir)

train_non_mae, train_nmae, train_r = evaluator.get_metrics(evaluator.trainloader)
train_r = train_r**2
valid_non_mae, valid_nmae, valid_r = evaluator.get_metrics(evaluator.validloader)
valid_r = valid_r**2
test_non_mae, test_nmae, test_r = evaluator.get_metrics(evaluator.testloader)
test_r = test_r**2


dir = "../../notebooks/metrics/"
np.save(dir+"Transformer_train_MAE.npy", train_non_mae)
np.save(dir+"Transformer_train_NMAE.npy", train_nmae)
np.save(dir+"Transformer_train_R2.npy", train_r)
np.save(dir+"Transformer_valid_MAE.npy", valid_non_mae)
np.save(dir+"Transformer_valid_NMAE.npy", valid_nmae)
np.save(dir+"Transformer_valid_R2.npy", valid_r)
np.save(dir+"Transformer_test_MAE.npy", test_non_mae)
np.save(dir+"Transformer_test_NMAE.npy", test_nmae)
np.save(dir+"Transformer_test_R2.npy", test_r)

train_values, valid_values = evaluator.get_loss()
plt.plot(train_values, label="train")
plt.plot(valid_values, label="valid")
plt.xticks(np.arange(len(valid_values)))
plt.yscale("log")
plt.legend()
plt.savefig(dir+"Transformer_training_loss.png")
plt.show()
