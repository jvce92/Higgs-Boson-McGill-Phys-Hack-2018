import xgboost as xgb
import numpy as np

class Model:
    def __init__(self, params):
        self.params = params

    def fit(self, X_train, y_train, X_val, y_val, num_rounds=10):
        dtrain = xgb.DMatrix(self.clean_features(X_train), label=self.clean_labels(y_train))
        dval = xgb.DMatrix(self.clean_features(X_val), label=self.clean_labels(y_val))
        evallist = [(dval, 'eval'), (dtrain, 'train')]
        self.bst = xgb.train(self.params, dtrain, num_rounds, evallist)
    
    def predict(self, X):
        dtest = xgb.DMatrix(self.clean_features(X))
        return self.bst.predict(dtest)

    def score(self, W, y_pred, y_true, cutoff):
        s = np.sum(W * (y_pred >= cutoff).astype('float32') * (y_true["Label"] == 1))
        b = np.sum(W * (y_pred >= cutoff).astype('float32') * (y_true["Label"] == 0))
        
        ams = np.sqrt(2 * ((s + b) * np.log(1 + (s / b)) - s))

        return ams
                      
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