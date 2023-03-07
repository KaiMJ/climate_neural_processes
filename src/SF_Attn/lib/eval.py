from .utils import *
from .dataset import l2Dataset
from .model import Model
from .loss import *
import torch
from torch.utils.data import DataLoader
import yaml
import glob
import dill
import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Evaluator():
    def __init__(self, dirpath):
        self.dirpath = dirpath
        self.config = yaml.safe_load(open(f"{dirpath}/saved_config.yaml"))
        self.init_dataloader()
        self.init_model()

    def init_model(self):
        self.device = torch.device('cuda')
        model_dict = torch.load(
            f"{self.dirpath}/best.pt", map_location=torch.device('cuda'))
        model = Model(model_dict['config']['model']).to(self.device)

        model.load_state_dict(model_dict['model'])
        model.eval()
        self.model = model

    def init_dataloader(self):
        l2_x_data = sorted(
            glob.glob(f"{self.config['data_dir']}/SPCAM5/inputs_*"), key=sort_fn)
        l2_y_data = sorted(
            glob.glob(f"{self.config['data_dir']}/SPCAM5/outputs_*"), key=sort_fn)

        n = int(365*0.8)
        self.l2_x_train = l2_x_data[:n]
        self.l2_y_train = l2_y_data[:n]
        self.l2_x_valid = l2_x_data[n:365]
        self.l2_y_valid = l2_y_data[n:365]
        self.l2_x_test = l2_x_data[365:]
        self.l2_y_test = l2_y_data[365:]

        l2_x_scaler_minmax = dill.load(
            open(f"../../scalers/x_SPCAM5_minmax_scaler.dill", 'rb'))
        l2_y_scaler_minmax = dill.load(
            open(f"../../scalers/y_SPCAM5_minmax_scaler.dill", 'rb'))

        # Change to first 26 variables
        l2_y_scaler_minmax.min = l2_y_scaler_minmax.min[:26]
        l2_y_scaler_minmax.max = l2_y_scaler_minmax.max[:26]

        trainset = l2Dataset(self.l2_x_train, self.l2_y_train,
                             x_scaler=l2_x_scaler_minmax, y_scaler=l2_y_scaler_minmax, variables=26)
        self.trainloader = DataLoader(trainset, batch_size=self.config['batch_size'], shuffle=True, drop_last=False,
                                      num_workers=4, pin_memory=True)
        validset = l2Dataset(self.l2_x_valid, self.l2_y_valid,
                             x_scaler=l2_x_scaler_minmax, y_scaler=l2_y_scaler_minmax, variables=26)
        self.validloader = DataLoader(validset, batch_size=self.config['batch_size'], shuffle=False, drop_last=False,
                                      num_workers=4, pin_memory=True)
        testset = l2Dataset(self.l2_x_test, self.l2_y_test,
                            x_scaler=l2_x_scaler_minmax, y_scaler=l2_y_scaler_minmax, variables=26)
        self.testloader = DataLoader(testset, batch_size=self.config['batch_size'], shuffle=False, drop_last=False,
                                     num_workers=4, pin_memory=True)

        self.l2_y_scaler_minmax = l2_y_scaler_minmax

    def get_metrics(self, loader):
        self.get_R_stats(loader)
        self.r = self.ssxym / np.sqrt(self.ssxm * self.ssym)
        return self.non_mae, self.nmae, self.r

    def forward_pass(self, data):
        with torch.no_grad():
            x, y = data

            x = x.reshape(-1, 1, x.shape[-1]).to(device)
            y = y.reshape(-1, 1, y.shape[-1]).to(device)
            context_idxs, target_idxs = split_context_target(
                x, self.config['context_percentage_low'], self.config['context_percentage_high'])

            x_context = x[context_idxs]
            y_context = y[context_idxs]
            x_target = x[target_idxs]
            y_target = y[target_idxs]

            with torch.cuda.amp.autocast():
                l2_output_mu, l2_output_cov, l2_z_mu_all, l2_z_cov_all, l2_z_mu_c, l2_z_cov_c = self.model(
                    x_context, y_context, x_target, x_all=x, y_all=y)

            non_y_pred = self.l2_y_scaler_minmax.inverse_transform(
                l2_output_mu.squeeze().cpu().numpy())
            non_y = self.l2_y_scaler_minmax.inverse_transform(
                y_target.squeeze().cpu().numpy())
            return non_y, non_y_pred, context_idxs, target_idxs
        
    def get_loss(self):
        path = f"{self.dirpath}/best.pt"

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

        self.y_mean /= self.n_total
        self.y_pred_mean /= self.n_total
        self.nmae /= self.n_total
        self.non_mae /= self.n_total
        self.nmae = np.abs(
            np.sqrt(self.non_mae / self.n_total) / np.abs(self.y_mean))

    def plot_scenario(self, idx, split="test"):
        """
            Plots idxth day of the scenario.
        """
        if split == "test":
            loader = self.testloader
        elif split == "valid":
            loader = self.validloader
        else:
            loader = self.trainloader

        for i, data in enumerate(loader):
            x, y = data

            with torch.no_grad():
                x = x.reshape(24, -1, 1, x.shape[-1])[idx].to(device)
                y = y.reshape(24, -1, 1, y.shape[-1])[idx].to(device)
                context_idxs, target_idxs = split_context_target(x, self.config['context_percentage_low'],
                                                                 self.config['context_percentage_high'])

                x_context = x[context_idxs]
                y_context = y[context_idxs]
                # x_target = x[target_idxs]
                # y_target = y[target_idxs]

                with torch.cuda.amp.autocast():
                    l2_output_mu, l2_output_cov = self.model(
                        x_context, y_context, x)

                non_y_pred = self.l2_y_scaler_minmax.inverse_transform(
                    l2_output_mu.squeeze().cpu().numpy())
                non_y_context = self.l2_y_scaler_minmax.inverse_transform(
                    y_context.squeeze().cpu().numpy())
                non_y = self.l2_y_scaler_minmax.inverse_transform(
                    y.squeeze().cpu().numpy())

            return non_y, non_y_context, non_y_pred, context_idxs, target_idxs

        # return self.losses
