from .utils import *
from .dataset import MultiDataset
from .model import Model
from .loss import *
import torch
from torch.utils.data import DataLoader
import yaml
import glob
import dill
from tqdm import tqdm
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")


class Evaluator():
    def __init__(self, dirpath):
        self.dirpath = dirpath
        self.config = yaml.safe_load(open(f"{dirpath}/saved_config.yaml"))
        self.init_dataloader()
        self.init_model()

    def init_model(self):
        self.device = device
        model_dict = torch.load(
            f"{self.dirpath}/best.pt", map_location=device)
        model = Model(model_dict['config']['model']).to(self.device)

        model.load_state_dict(model_dict['model'])
        model.eval()
        self.model = model

    def init_dataloader(self):
        l2_x_data = sorted(
            glob.glob(f"{self.config['data_dir']}/SPCAM5/inputs_*"), key=sort_fn)
        l2_y_data = sorted(
            glob.glob(f"{self.config['data_dir']}/SPCAM5/outputs_*"), key=sort_fn)
        l1_x_data = sorted(
            glob.glob(f"{self.config['data_dir']}/CAM5/inputs_*"), key=sort_fn)
        l1_y_data = sorted(
            glob.glob(f"{self.config['data_dir']}/CAM5/outputs_*"), key=sort_fn)

        split_n = int(365*0.8)
        l2_x_train = l2_x_data[:split_n]
        l2_y_train = l2_y_data[:split_n]
        l2_x_valid = l2_x_data[split_n:365]
        l2_y_valid = l2_y_data[split_n:365]
        l1_x_train = l1_x_data[:split_n]
        l1_y_train = l1_y_data[:split_n]
        l1_x_valid = l1_x_data[split_n:365]
        l1_y_valid = l1_y_data[split_n:365]

        l1_x_test = l1_x_data[365:]
        l1_y_test = l1_y_data[365:]
        l2_x_test = l2_x_data[365:]
        l2_y_test = l2_y_data[365:]

        l2_x_scaler_minmax = dill.load(
            open(f"../../scalers/x_SPCAM5_minmax_scaler.dill", 'rb'))
        l2_y_scaler_minmax = dill.load(
            open(f"../../scalers/y_SPCAM5_minmax_scaler.dill", 'rb'))
        l1_x_scaler_minmax = dill.load(
            open(f"../../scalers/x_CAM5_minmax_scaler.dill", 'rb'))
        l1_y_scaler_minmax = dill.load(
            open(f"../../scalers/y_CAM5_minmax_scaler.dill", 'rb'))

        # Change to first 26 variables
        # Follow Azis's process. X -> X/(max(abs(X))
        l2_x_scaler_minmax.min = l2_x_scaler_minmax.min * 0
        l2_x_scaler_minmax.max = np.abs(l2_x_scaler_minmax.max)
        l2_y_scaler_minmax.min = l2_y_scaler_minmax.min[:26] * 0
        l2_y_scaler_minmax.max = np.abs(l2_y_scaler_minmax.max[:26])
        l1_x_scaler_minmax.min = l1_x_scaler_minmax.min * 0
        l1_x_scaler_minmax.max = np.abs(l1_x_scaler_minmax.max)
        l1_y_scaler_minmax.min = l1_y_scaler_minmax.min[:26] * 0
        l1_y_scaler_minmax.max = np.abs(l1_y_scaler_minmax.max[:26])

        self.l2_x_scaler_minmax = l2_x_scaler_minmax
        self.l2_y_scaler_minmax = l2_y_scaler_minmax
        self.l1_x_scaler_minmax = l1_x_scaler_minmax
        self.l1_y_scaler_minmax = l1_y_scaler_minmax

        train_dataset = MultiDataset(l1_x_train, l1_y_train, l2_x_train, l2_y_train,
                                     l1_x_scaler=l1_x_scaler_minmax, l1_y_scaler=l1_y_scaler_minmax,
                                     l2_x_scaler=l2_x_scaler_minmax, l2_y_scaler=l2_y_scaler_minmax, variables=[26, 26])
        self.trainloader = DataLoader(
            train_dataset, self.config['batch_size'], shuffle=True, drop_last=False, num_workers=4, pin_memory=True)

        val_dataset = MultiDataset(l1_x_valid, l1_y_valid, l2_x_valid, l2_y_valid,
                                   l1_x_scaler=l1_x_scaler_minmax, l1_y_scaler=l1_y_scaler_minmax,
                                   l2_x_scaler=l2_x_scaler_minmax, l2_y_scaler=l2_y_scaler_minmax, variables=[26, 26])
        self.valloader = DataLoader(
            val_dataset, self.config['batch_size'], shuffle=False, drop_last=False, num_workers=4, pin_memory=True)

        test_dataset = MultiDataset(l1_x_test, l1_y_test, l2_x_test, l2_y_test,
                                    l1_x_scaler=l1_x_scaler_minmax, l1_y_scaler=l1_y_scaler_minmax,
                                    l2_x_scaler=l2_x_scaler_minmax, l2_y_scaler=l2_y_scaler_minmax, variables=[26, 26])

        self.testloader = DataLoader(
            test_dataset, self.config['batch_size'], shuffle=False, drop_last=False, num_workers=4, pin_memory=True)

        self.l2_y_scaler_minmax = l2_y_scaler_minmax

    def get_metrics(self, loader):
        self.get_R_stats(loader)
        self.r = self.ssxym / np.sqrt(self.ssxm * self.ssym)
        return self.non_mae, self.nmae, self.r

    def forward_pass(self, data):
        with torch.no_grad():
            n_minibatch = 24
            n_miniminibatch = 4
            l2_x_, l2_y_, l1_x_, l1_y_ = data

            l2_x_ = l2_x_.reshape(24, 96, 144, l2_x_.shape[-1])
            l2_y_ = l2_y_.reshape(24, 96, 144, l2_y_.shape[-1])
            l1_x_ = l1_x_.reshape(24, 96, 144, l1_x_.shape[-1])
            l1_y_ = l1_y_.reshape(24, 96, 144, l1_y_.shape[-1])

            non_y_all = None
            non_y_pred_all = None

            # Split 24 hour data into each hour
            for mb in range(n_minibatch):
                # split each hour into 4 minibatches for GPU memory
                for minimb in range(n_miniminibatch):
                    l2_x = l2_x_[mb].reshape(
                        1, -1, l2_x_.shape[-1])[:, minimb::n_miniminibatch].to(device)
                    l2_y = l2_y_[mb].reshape(
                        1, -1, l2_y_.shape[-1])[:, minimb::n_miniminibatch].to(device)
                    l1_x = l1_x_[mb].reshape(
                        1, -1, l1_x_.shape[-1])[:, minimb::n_miniminibatch].to(device)
                    l1_y = l1_y_[mb].reshape(
                        1, -1, l1_y_.shape[-1])[:, minimb::n_miniminibatch].to(device)

                    context_idxs, target_idxs = split_context_target(
                        l2_x, self.config['context_percentage_low'], self.config['context_percentage_high'], axis=1)

                    l2_x_context = l2_x[:, context_idxs].to(device)
                    l2_y_context = l2_y[:, context_idxs].to(device)
                    l2_x_target = l2_x[:, target_idxs].to(device)
                    l2_y_target = l2_y[:, target_idxs].to(device)

                    l1_x_context = l1_x[:, context_idxs].to(device)
                    l1_y_context = l1_y[:, context_idxs].to(device)
                    l1_x_target = l1_x[:, target_idxs].to(device)
                    l1_y_target = l1_y[:, target_idxs].to(device)

                    with torch.cuda.amp.autocast():
                        l2_output_mu, l2_output_cov, l1_output_mu, l1_output_cov, \
                            l2_z_mu_c, l2_z_cov_c, l2_z_mu_all, l2_z_cov_all, \
                            l1_z_mu_c, l1_z_cov_c, l1_z_mu_all, l1_z_cov_all = \
                            self.model(l1_x_context, l1_y_context, l1_x_target, l2_x_context, l2_y_context, l2_x_target,
                                       l1_y_target, l2_y_target)

                    non_y_pred = self.l2_y_scaler_minmax.inverse_transform(
                        l2_output_mu.squeeze().detach().cpu().numpy())
                    non_y = self.l2_y_scaler_minmax.inverse_transform(
                        l2_y_target.squeeze().detach().cpu().numpy())

                    non_y_all = np.concatenate(
                        (non_y_all, non_y), axis=0) if non_y_all is not None else non_y
                    non_y_pred_all = np.concatenate(
                        (non_y_pred_all, non_y_pred), axis=0) if non_y_pred_all is not None else non_y_pred
                # break
            return non_y, non_y_pred, context_idxs, target_idxs

    def get_loss(self):
        step_size = len(self.trainloader)
        train_event_file = sorted(glob.glob(os.path.join(
            self.dirpath, "runs/train/events.out.tfevents*")), key=os.path.getctime)[-1]
        valid_event_file = sorted(glob.glob(os.path.join(
            self.dirpath, "runs/valid/events.out.tfevents*")), key=os.path.getctime)[-1]
        train_acc = EventAccumulator(train_event_file)
        valid_acc = EventAccumulator(valid_event_file)
        train_acc.Reload()
        valid_acc.Reload()

        valid_values = [s.value for s in valid_acc.Scalars("non_mae")]
        n_epochs = len(valid_values)

        # Change each iteration to epochs
        train_values = np.array(
            [s.value for s in train_acc.Scalars("non_mae")])
        max_n = min(len(train_values) // step_size, n_epochs)
        train_values = train_values[:max_n * step_size]
        valid_values = valid_values[:max_n]
        train_values = train_values.reshape(-1, step_size).mean(axis=1)

        return train_values, valid_values

    def get_R_stats(self, loader):
        self._get_stats(loader)
        self.ssxm = 0
        self.ssxym = 0
        self.ssym = 0

        with torch.no_grad():
            for i, data in enumerate(tqdm(loader, total=len(loader))):
                non_y, non_y_pred, _, _ = self.forward_pass(data)
                self.ssxm += ((non_y - self.y_mean)**2).sum(0)
                self.ssym += ((non_y_pred - self.y_pred_mean)**2).sum(0)
                self.ssxym += ((non_y - self.y_mean) *
                               (non_y_pred - self.y_pred_mean)).sum(0)
            # Get average
            self.ssxm /= self.n_total
            self.ssym /= self.n_total
            self.ssxym /= self.n_total

    def _get_stats(self, loader):
        self.n_total = 0
        self.x_total = 0
        self.y_total = 0
        self.xy_total = 0
        self.x2_total = 0
        self.y2_total = 0
        self.y_mean = 0
        self.y_pred_mean = 0
        self.nmae = 0
        self.non_mae = 0
        self.y_max = 0

        with torch.no_grad():
            for i, data in enumerate(tqdm(loader, total=len(loader))):
                non_y, non_y_pred, _, _ = self.forward_pass(data)

                non_mae = mae_metric(non_y_pred, non_y, mean=False)

                self.y_mean += non_y.sum(axis=0)
                self.y_pred_mean += non_y_pred.sum(axis=0)
                self.non_mae += non_mae.sum(axis=0)
                self.n_total += non_y.shape[0]
                self.x_total += non_y.sum(axis=0)
                self.y_total += non_y_pred.sum(axis=0)
                self.x2_total += (non_y ** 2).sum(axis=0)
                self.y2_total += (non_y_pred ** 2).sum(axis=0)
                self.xy_total += (non_y_pred * non_y).sum(axis=0)

                self.y_max = np.maximum(self.y_max, np.abs(non_y).max(axis=0))

        self.y_mean /= self.n_total
        self.y_pred_mean /= self.n_total
        self.nmae /= self.n_total
        self.non_mae /= self.n_total
        self.nmae = self.non_mae / self.y_max

    def plot_scenario(self, day, hour=0, split="test"):
        """
            Plots xth day of the scenario.
        """
        if split == "test":
            loader = self.testloader
        elif split == "valid":
            loader = self.valloader
        else:
            loader = self.trainloader

        for i, data in enumerate(loader):
            if i < day:
                continue
            l2_x_, l2_y_, l1_x_, l1_y_ = data

            non_y_all = np.zeros((96 * 144, 26))
            non_y_pred_all = np.zeros((96 * 144, 26))
            context_idxs_all = None
            target_idxs_all = None

            n_mb = 96
            with torch.no_grad():
                l2_x_ = l2_x_.reshape(24, 1, -1, l2_x_.shape[-1])[hour]
                l2_y_ = l2_y_.reshape(24, 1, -1, l2_y_.shape[-1])[hour]

                for mb in range(n_mb):
                    l2_x = l2_x_[:, mb::n_mb].to(device)
                    l2_y = l2_y_[:, mb::n_mb].to(device)
                    l1_x = l1_x_[:, mb::n_mb].to(device)
                    l1_y = l1_y_[:, mb::n_mb].to(device)

                    context_idxs, target_idxs = split_context_target(l2_x, self.config['context_percentage_low'],
                                                                     self.config['context_percentage_high'], axis=1)

                    l2_x_context = l2_x[:, context_idxs]
                    l2_y_context = l2_y[:, context_idxs]
                    l1_x_context = l1_x[:, context_idxs]
                    l1_y_context = l1_y[:, context_idxs]
                    l2_x_target = l2_x[:, target_idxs]
                    l2_y_target = l2_y[:, target_idxs]
                    l1_x_target = l1_x[:, target_idxs]
                    l1_y_target = l1_y[:, target_idxs]

                    # Each context index is repeated n_mb times
                    mb_context_idxs = mb + n_mb * context_idxs
                    mb_target_idxs = mb + n_mb * target_idxs

                    context_idxs_all = np.concatenate(
                        (context_idxs_all, mb_context_idxs), axis=0) if context_idxs_all is not None else mb_context_idxs
                    target_idxs_all = np.concatenate(
                        (target_idxs_all, mb_target_idxs), axis=0) if target_idxs_all is not None else mb_target_idxs

                    # For target
                    with torch.cuda.amp.autocast():
                        l2_output_mu, l2_output_cov, l1_output_mu, l1_output_cov, \
                            l2_z_mu_c, l2_z_cov_c, l2_z_mu_all, l2_z_cov_all, \
                            l1_z_mu_c, l1_z_cov_c, l1_z_mu_all, l1_z_cov_all = \
                            self.model(l1_x_context, l1_y_context, l1_x_target, l2_x_context, l2_y_context, l2_x_target,
                                       l1_y_target, l2_y_target)
                    non_y_pred = self.l2_y_scaler_minmax.inverse_transform(
                        l2_output_mu.squeeze().cpu().numpy())
                    non_y_pred_all[mb::n_mb][target_idxs] = non_y_pred

                    # For Context
                    with torch.cuda.amp.autocast():
                        l2_output_mu, l2_output_cov, l1_output_mu, l1_output_cov, \
                            l2_z_mu_c, l2_z_cov_c, l2_z_mu_all, l2_z_cov_all, \
                            l1_z_mu_c, l1_z_cov_c, l1_z_mu_all, l1_z_cov_all = \
                            self.model(l1_x_context, l1_y_context, l1_x_context, l2_x_context, l2_y_context, l2_x_context,
                                       l1_y_context, l2_y_context)
                    non_y_pred = self.l2_y_scaler_minmax.inverse_transform(
                        l2_output_mu.squeeze().cpu().numpy())
                    non_y_pred_all[mb::n_mb][context_idxs] = non_y_pred

                    non_y = self.l2_y_scaler_minmax.inverse_transform(
                        l2_y.squeeze().cpu().numpy())
                    non_y_all[mb::n_mb] = non_y

            return non_y_all, non_y_pred_all, context_idxs_all, target_idxs_all

        # return self.losses
