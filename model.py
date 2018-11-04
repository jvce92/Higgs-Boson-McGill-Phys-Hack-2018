import xgboost as xgb
import numpy as np
from matplotlib import pyplot as plt

class Model:
    def __init__(self, params):
        self.params = params
        self.evals_result = {}
        self.metrics = params['eval_metric']

    def fit(self, X_train, y_train, X_val=None, y_val=None, num_rounds=100, clean=True):
        if clean:
            dtrain = xgb.DMatrix(self.clean_features(X_train), label=self.clean_labels(y_train))
        else:
            dtrain = xgb.DMatrix(X_train, label=y_train)
        
        evallist = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(self.clean_features(X_val), label=self.clean_labels(y_val))
            evallist.append((dval, 'eval'))
        self.bst = xgb.train(self.params, dtrain, num_rounds, evallist, evals_result=self.evals_result)
    
    def predict(self, X, cutoff=None, clean=True):
        if clean:
            dtest = xgb.DMatrix(self.clean_features(X))
        else:
            dtest = xgb.DMatrix(X)
        
        preds = self.bst.predict(dtest)
        if cutoff is None:
            return preds
        else:
            return (preds >= cutoff).astype('float32')

    def cv(self, params, X_train, y_train, num_boost_round=10, n_fold=3, stratified=False, verbose=None):
        dtrain = xgb.DMatrix(self.clean_features(X_train), label=self.clean_labels(y_train))
        self.cv_eval_hist = xgb.cv(params, dtrain, metrics=params['eval_metric'],
                                   num_boost_round=num_boost_round, nfold=n_fold, stratified=stratified,
                                   verbose_eval=verbose)

    def score(self, W, y_pred, y_true, cutoff):
        s = np.sum(W * (y_pred >= cutoff).astype('float32') * (y_true["Label"] == 1))
        b = np.sum(W * (y_pred >= cutoff).astype('float32') * (y_true["Label"] == 0))
        
        ams = np.sqrt(2 * ((s + b) * np.log(1 + (s / b)) - s))

        return ams
                      
    def get_score(self):
        return self.bst.get_score()

    def get_fscore(self):
        return self.bst.get_fscore()

    def save_model(self, fname):
        self.bst.save_model(fname)

    def load_model(self, fname):
        self.bst = xgb.Booster()
        self.bst.load_model(fname)

    def clean_features(self, X):
        columns =  ['DER_mass_MMC',
                    'DER_mass_transverse_met_lep',
                    'DER_mass_vis',
                    'DER_pt_h',
                    'DER_deltaeta_jet_jet',
                    'DER_mass_jet_jet',
                    'DER_prodeta_jet_jet',
                    'DER_deltar_tau_lep',
                    'DER_pt_tot',
                    'DER_sum_pt',
                    'DER_pt_ratio_lep_tau',
                    'DER_met_phi_centrality',
                    'DER_lep_eta_centrality',
                    'PRI_tau_pt',
                    'PRI_tau_eta',
                    'PRI_tau_phi',
                    'PRI_lep_pt',
                    'PRI_lep_eta',
                    'PRI_lep_phi',
                    'PRI_met',
                    'PRI_met_phi',
                    'PRI_met_sumet',
                    'PRI_jet_num',
                    'PRI_jet_leading_pt',
                    'PRI_jet_leading_eta',
                    'PRI_jet_leading_phi',
                    'PRI_jet_subleading_pt',
                    'PRI_jet_subleading_eta',
                    'PRI_jet_subleading_phi',
                    'PRI_jet_all_pt']
        
        return X[columns]

    def clean_labels(self, y):
        return y["Label"]

    def plot_evals_result(self):
        fig, ax = plt.subplots(len(self.metrics), 1, figsize=(12,9))
        
        for i, metric in enumerate(self.metrics):
            ax[i].set_title(metric)
            ax[i].plot(self.evals_result['train'][metric], '-b', lw=2, label='Training')
            ax[i].plot(self.evals_result['eval'][metric], '-r', lw=2, label='Validation')
            ax[i].legend()

        plt.show()

    def plot_ams(self, W, y_pred, y_true):
        ams_data = [self.score(W, y_pred, y_true, cutoff) for cutoff in np.linspace(0.1, .95, 100)]
        plt.figure(figsize=(12,9))

        plt.title("AMS")
        plt.plot(np.linspace(0.1, .95, 100), ams_data, lw=2)
        plt.ylabel("AMS")
        plt.xlabel("Cutoff Probability")

        plt.show()

    def plot_histogram(self, W, X_train, y_train, cutoff):
        plt.figure(figsize=(12,9))

        y_pred = self.predict(X_train)
        y_pred_signal = self.bst.predict(xgb.DMatrix(self.clean_features(X_train.iloc[np.where(y_train >= 0.5)[0], :])))
        y_pred_bg = self.bst.predict(xgb.DMatrix(self.clean_features(X_train.iloc[np.where(y_train < 0.5)[0], :])))

        ams = self.score(W, y_pred, y_train, cutoff)
        # y_pred_hard = np.zeros_like(y_pred)
        # y_pred_hard[np.where(y_pred >= cutoff)] = 1.0 
        # error = np.sum(y_pred != y_train) / y_pred.shape[0]

        range_max = np.max(np.vstack((y_pred_signal.reshape((-1, 1)), y_pred_bg.reshape(-1, 1))))
        range_min = np.min(np.vstack((y_pred_signal.reshape((-1, 1)), y_pred_bg.reshape(-1, 1))))
        
        vals_bg, bins_bg, _ = plt.hist(y_pred_bg, bins=50, color='red', alpha=.5)
        vals_signal, bins_signal, _ = plt.hist(y_pred_signal, bins=50, color='blue', alpha=.5)
        
        plt.axvspan(range_min, cutoff, color='red', alpha=0.18, label="Background")
        plt.axvspan(cutoff, range_max, color='blue', alpha=0.18, label="Signal")

        max_y = np.max(np.vstack((vals_bg.reshape((-1, 1)), vals_signal.reshape((-1, 1)))))
        plt.text(range_max - 0.3, max_y * 0.8, "AMS Score = {0:.4f}".format(ams), fontsize=16)
        # plt.text(range_max - 0.4, max_y * 0.7, "Classification Error = {0:.4f}".format(error), fontsize=16)

        plt.legend()
        plt.axis([range_min, range_max, 
                  0, 1000 + max_y])

        plt.show()