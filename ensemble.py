import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from model import Model
from prepare import prepare_data
from sklearn.model_selection import StratifiedKFold

class Ensemble:
    def __init__(self, num_of_classifiers, params_list, blender_params):
        self.params_list = params_list
        self.num_of_classifiers = num_of_classifiers
        self.blender_params = blender_params
        self.clfs = [Model(params) for params in params_list]
        self.blender_clf = Model(blender_params)

    def fit_predict(self, X, y, X_sub, splits=5, num_rounds=100):
        skf = StratifiedKFold(splits)

        X_train_blend = np.zeros((X.shape[0], self.num_of_classifiers))
        X_test_blend = np.zeros((X_sub.shape[0], self.num_of_classifiers))
        
        for i, clf in enumerate(self.clfs):
            print("Classifier {0}:".format(i))
            X_test_blend_i = np.zeros((X_sub.shape[0], splits))
            for j, (train, test) in enumerate(skf.split(clf.clean_features(X), clf.clean_labels(y))):
                print()
                print("Fold {0}:".format(j+1))
                X_train = X.iloc[train, :]
                y_train = y.iloc[train, :]
                X_val = X.iloc[test, :]
                y_val = y.iloc[test, :]
                clf.fit(X_train, y_train, X_val, y_val, num_rounds=num_rounds)
                y_pred = clf.predict(X_val)
                X_train_blend[test, i] = y_pred
                X_test_blend_i[:, j] = clf.predict(X_sub)
            X_test_blend[:, i] = X_test_blend_i.mean(1)
            print()

        print("Classifier Blender")
        self.blender_clf.fit(X_train_blend, y["Label"].values, clean=False)
        y_sub = self.blender_clf.predict(X_test_blend, clean=False)

        return y_sub

    def predict(self, X_sub, splits=5):
        X_test_blend = np.zeros((X_sub.shape[0], self.num_of_classifiers))
        
        for i, clf in enumerate(self.clfs):
            X_test_blend_i = np.zeros((X_sub.shape[0], splits))
            for j in range(splits):
                X_test_blend_i[:, j] = clf.predict(X_sub)
            X_test_blend_i[:, j] = X_test_blend_i.mean(1)

        y_sub = self.blender_clf.predict(X_test_blend, clean=False)

        return y_sub
            
    def load_ensemble(self, clfs_fname, blender_fname):
        for i, fname in enumerate(clfs_fname):
            self.clfs[i].load_model(fname)

        self.blender_clf.load_model(blender_fname)

    def score(self, W, y_pred, y_true, cutoff):
        s = np.sum(W * (y_pred >= cutoff).astype('float32') * (y_true["Label"] == 1))
        b = np.sum(W * (y_pred >= cutoff).astype('float32') * (y_true["Label"] == 0))
        
        ams = np.sqrt(2 * ((s + b) * np.log(1 + (s / b)) - s))

        return ams

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
        y_pred_signal = self.predict(X_train.iloc[np.where(y_train >= 0.5)[0], :])
        y_pred_bg = self.predict(X_train.iloc[np.where(y_train < 0.5)[0], :])

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