import os
import torch
import numpy as np
import pandas as pd
from torch import optim
from synthcity.metrics import eval_statistical
from synthcity.metrics import eval_detection
from synthcity.metrics import eval_performance
from synthcity.plugins.core.schema import Schema
from data_utils import get_dataloader
from model.Goggle import Goggle
from model.GoggleLoss import GoggleLoss


class GoggleModel():
    def __init__(self,
                 ds_name,
                 input_dim,
                 encoder_dim=64,
                 encoder_l=2,
                 het_encoding=True,
                 decoder_dim=64,
                 decoder_l=2,
                 threshold=0.1,
                 het_decoder=False,
                 graph_prior=None,
                 prior_mask=None,
                 device='cpu',
                 alpha=0.1,
                 beta=0.1,
                 seed=42,
                 **kwargs):
        self.ds_name = ds_name
        self.device = device
        self.seed = seed
        self.learning_rate = kwargs.get('learning_rate', 5e-3)
        self.weight_decay = kwargs.get('weight_decay', 1e-3)
        self.epochs = kwargs.get('epochs', 1000)
        self.batch_size = kwargs.get('batch_size', 64)
        self.patience = kwargs.get('patience', 50)
        self.logging_epoch = kwargs.get('logging', 50)
        self.loss = GoggleLoss(alpha, beta, graph_prior, device)
        self.model = Goggle(input_dim,
                            encoder_dim,
                            encoder_l,
                            het_encoding,
                            decoder_dim,
                            decoder_l,
                            threshold,
                            het_decoder,
                            graph_prior,
                            prior_mask,
                            device).to(device)
        self.optimiser = optim.Adam(self.model.parameters(),
                                    lr=self.learning_rate,
                                    weight_decay=self.weight_decay)

    def evaluate(self, data_loader, epoch):
        with torch.no_grad():
            eval_loss, rec_loss, kld_loss, graph_loss = 0., 0., 0., 0.
            num_samples = 0
            for _, data in enumerate(data_loader):
                self.model.eval()
                data = data[0].to(self.device)
                x_hat, adj, mu_z, logvar_z = self.model(data, epoch)
                loss, loss_rec, loss_kld, loss_graph = self.loss(
                    x_hat, data, mu_z, logvar_z, adj)

                eval_loss += loss.item()
                rec_loss += loss_rec.item()
                kld_loss += loss_kld.item()
                graph_loss += loss_graph.item()*data.shape[0]
                num_samples += data.shape[0]

            eval_loss /= num_samples
            rec_loss /= num_samples
            kld_loss /= num_samples
            graph_loss /= num_samples

            return eval_loss, rec_loss, kld_loss, graph_loss

    def fit(self, data):

        data_loaders = get_dataloader(data, self.batch_size, self.seed)
        model_save_path = f'tmp/{self.ds_name}.pt'

        if not os.path.exists('tmp/'):
            os.makedirs('tmp/')
            print('created path to save models...')

        train_loader = data_loaders['train']
        best_loss = np.inf

        for epoch in range(self.epochs):

            train_loss, num_samples = 0., 0
            for _, data in enumerate(train_loader):
                data = data[0].to(self.device)
                self.model.train()

                self.optimiser.zero_grad()

                x_hat, adj, mu_z, logvar_z = self.model(data, epoch)
                loss, _, _, _ = self.loss(
                    x_hat, data, mu_z, logvar_z, adj)

                loss.backward(retain_graph=True)
                self.optimiser.step()

                train_loss += loss.item()
                num_samples += data.shape[0]

            train_loss /= num_samples

            val_loss = self.evaluate(data_loaders['val'], epoch)

            if val_loss[1] < best_loss:
                best_loss = val_loss[1]
                patience = 0
                torch.save(self.model.state_dict(), model_save_path)
            else:
                patience += 1

            if (epoch+1) % self.logging_epoch == 0:
                print(
                    f'[Epoch {(epoch+1):3}/{self.epochs}, patience {patience:2}] train: {train_loss:.3f}, val: {val_loss[0]:.3f}')

            if patience == self.patience:
                self.model.load_state_dict(torch.load(
                    model_save_path), strict=False)
                print(f'Training terminated after {epoch} epochs')
                break

    def enforce_constraints(self, X_synth, X_test):
        schema = Schema(data=X_test)
        X_synth = pd.DataFrame(X_synth, columns=schema.features())
        for rule in schema.as_constraints().rules:
            if rule[1] == 'in':
                X_synth[rule[0]] = X_synth[rule[0]].apply(
                    lambda x: min(rule[2], key=lambda z: abs(z-x)))
            elif rule[1] == 'eq':
                raise Exception('not yet implemented')
            else:
                pass
        return X_synth.values

    def sample(self, X_test):
        count = X_test.shape[0]
        X_synth = self.model.sample(count)
        X_synth = X_synth.cpu().detach().numpy()
        X_synth = self.enforce_constraints(X_synth, X_test)
        X_synth = pd.DataFrame(X_synth, columns=X_test.columns)
        return X_synth

    def evaluate_synthetic(self, X_synth, X_test):
        quality_evaluator = eval_statistical.AlphaPrecision()
        qual_res = quality_evaluator.evaluate(X_test, X_synth)
        qual_score = np.mean(list(qual_res.values()))

        xgb_evaluator = eval_performance.PerformanceEvaluatorXGB()
        linear_evaluator = eval_performance.PerformanceEvaluatorLinear()
        mlp_evaluator = eval_performance.PerformanceEvaluatorMLP()
        xgb_score = xgb_evaluator.evaluate(X_test, X_synth)
        linear_score = linear_evaluator.evaluate(X_test, X_synth)
        mlp_score = mlp_evaluator.evaluate(X_test, X_synth)
        gt_perf = (xgb_score['gt'] + linear_score['gt'] + mlp_score['gt'])/3
        synth_perf = (xgb_score['syn'] +
                      linear_score['syn'] + mlp_score['syn'])/3

        xgb_detector = eval_detection.SyntheticDetectionXGB()
        mlp_detector = eval_detection.SyntheticDetectionMLP()
        gmm_detector = eval_detection.SyntheticDetectionGMM()
        xgb_det = xgb_detector.evaluate(X_test, X_synth)
        mlp_det = mlp_detector.evaluate(X_test, X_synth)
        gmm_det = gmm_detector.evaluate(X_test, X_synth)
        det_score = (xgb_det['mean'] + mlp_det['mean'] + gmm_det['mean'])/3

        return qual_score, (gt_perf, synth_perf), det_score
