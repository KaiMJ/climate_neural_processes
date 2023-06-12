import matplotlib.pyplot as plt
from lib.eval import *
import numpy as np
import glob
import os

log_dir = "logs/SFNP_train_seed0_lr0.000227_bs_1_Tue May 16 19:25:33 2023"

evaluator = Evaluator(log_dir)

train_non_mae, train_nmae, train_r = evaluator.get_metrics(evaluator.trainloader)
train_r = train_r**2
val_non_mae, val_nmae, val_r = evaluator.get_metrics(evaluator.validloader)
val_r = val_r**2
test_non_mae, test_nmae, test_r = evaluator.get_metrics(evaluator.testloader)
test_r = test_r**2

train_values, valid_values = evaluator.get_loss()
plt.plot(train_values, label="train")
plt.plot(valid_values, label="valid")
plt.xticks(np.arange(len(valid_values)))
plt.yscale("log")
plt.legend()
plt.savefig("../../notebooks/plots/SFNP_training_loss.png")
plt.show()

np.save("../../notebooks/metrics/SFNP_train_NMAE.npy", train_nmae)
np.save("../../notebooks/metrics/SFNP_train_MAE.npy", train_non_mae)
np.save("../../notebooks/metrics/SFNP_train_R2.npy", train_r)
np.save("../../notebooks/metrics/SFNP_valid_NMAE.npy", val_nmae)
np.save("../../notebooks/metrics/SFNP_valid_MAE.npy", val_non_mae)
np.save("../../notebooks/metrics/SFNP_valid_R2.npy", val_r)
np.save("../../notebooks/metrics/SFNP_test_NMAE.npy", test_nmae)
np.save("../../notebooks/metrics/SFNP_test_MAE.npy", test_non_mae)
np.save("../../notebooks/metrics/SFNP_test_R2.npy", test_r)