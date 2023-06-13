from eval import *
import matplotlib.pyplot as plt


log_dir = "/data/kai/climate_neural_processes/src/SF_Attn/logs/ReTrain_seed0_lr0.000139_bs_1_Mon Jun 12 23:03:05 2023"
evaluator = Evaluator(log_dir)

dir = "../../notebooks/metrics/"
name = "SF_ATTN"

train_values, valid_values = evaluator.get_loss()
plt.plot(train_values, label="train")
plt.plot(valid_values, label="valid")
plt.xticks(np.arange(len(valid_values)))
plt.yscale("log")
plt.legend()
plt.title(name+" Training Loss")

plt.savefig(dir+name+"_training_loss.png")
plt.show()



train_non_mae, train_nmae, train_r = evaluator.get_metrics(evaluator.train_loader)
train_r = train_r**2
valid_non_mae, valid_nmae, valid_r = evaluator.get_metrics(evaluator.valid_loader)
valid_r = valid_r**2
test_non_mae, test_nmae, test_r = evaluator.get_metrics(evaluator.test_loader)
test_r = test_r**2

np.save(dir+name+"_train_MAE.npy", train_non_mae)
np.save(dir+name+"_train_NMAE.npy", train_nmae)
np.save(dir+name+"_train_R2.npy", train_r)
np.save(dir+name+"_valid_MAE.npy", valid_non_mae)
np.save(dir+name+"_valid_NMAE.npy", valid_nmae)
np.save(dir+name+"_valid_R2.npy", valid_r)
np.save(dir+name+"_test_MAE.npy", test_non_mae)
np.save(dir+name+"_test_NMAE.npy", test_nmae)
np.save(dir+name+"_test_R2.npy", test_r)