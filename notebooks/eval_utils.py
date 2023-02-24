import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
sys.path.append('../src')
from lib.loss import mae_metric, norm_rmse_loss

def plot_with_colorbar(fig, im, ax):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(im, vmin=0, vmax=1, cmap='jet')
    fig.colorbar(im, cax=cax, orientation='vertical')


def plot_variables(x, y, pred=None, loss=None, fidelity="high"):
    col = 2
    if pred is not None:
        col = 3
    fig, axs = plt.subplots(12, col, figsize=(10, 40))
    idx = 0
    n = 32
    for i in range(12):
        if i >= 4:
            n = 1
        if i < 8:
            plot_with_colorbar(fig, x[:, :, idx:idx+n].mean(-1), axs[i, 0])
            axs[i, 0].set_title("X")
        else:
            axs[i, 0].axis('off')
        plot_with_colorbar(fig, y[:, :, idx:idx+n].mean(-1), axs[i, 1])
        axs[i, 1].set_title("Y")
        if pred is not None:
            plot_with_colorbar(fig, pred[:, :, idx:idx+n].mean(-1), axs[i, 2])
            axs[i, 2].set_title("Pred")
            if loss is not None:
                axs[i, 2].set_title(f"Pred, loss: {loss[:, :, idx:idx+n].mean()}")
        idx += n

def plot_one_variable(x, y, y_pred, idx, figsize=(10, 5)):
    fig, axs = plt.subplots(2, 2, figsize=figsize, tight_layout=True)
    axs[0, 0].imshow(y[:, :, idx], vmax=1, vmin=0)
    axs[0, 0].set_title("True 0-1")
    axs[0, 1].imshow(y_pred[:, :, idx], vmax=1, vmin=0)
    axs[0, 1].set_title("Prediction 0-1")

    axs[1, 0].imshow(y[:, :, idx])
    axs[1, 0].set_title("Scaled True")
    axs[1, 1].imshow(y_pred[:, :, idx])
    axs[1, 1].set_title("Scaled Prediction")

    mae_loss = mae_metric(y_pred, y, mean=False).reshape(-1, 136).mean(0)[idx]
    norm_loss = norm_rmse_loss(y_pred, y, metric=True, mean=False).reshape(-1, 136).mean(0)[idx]
    plt.suptitle(f"MAE: {mae_loss:.4f} NRMSE: {norm_loss:.4f}")
    plt.show()
