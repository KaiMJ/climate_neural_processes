from lib.eval import *
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

log_dir = "logs/best_r2"
evaluator = Evaluator(log_dir)

non_mae, nmae, r = evaluator.get_metrics(evaluator.testloader)
r = r**2

np.save("../../notebooks/metrics/MAE_MFANP.npy", non_mae)
np.save("../../notebooks/metrics/r2_MFANP.npy", r)
np.save("../../notebooks/metrics/NMAE_MFANP.npy", nmae)