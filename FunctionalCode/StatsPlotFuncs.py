# -*- encoding: utf-8 -*-
"""
@brief: commonly used statistical plot funcs

@author: guanshikang

@type: class

Created on Mon Mar 24 10:13:10 2025, HONG KONG

! Marked by this clolor need adjustments for different tasks.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statistics import mean
from matplotlib import rcParams
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

config = {
    "font.family": 'Times New Roman',
    "font.size": 12,
    "mathtext.fontset": 'stix',
    "axes.unicode_minus": False
}
rcParams.update(config)


class StatsPlot:
    def __init__(self, save_dir=None):
        self.save_dir = os.getcwd() if save_dir is None else save_dir

    def _calculate_stats(self, x, y):
        MAE = mean_absolute_error(x, y)
        RMSE = root_mean_squared_error(x, y)
        R2 = r2_score(x, y)

        return MAE, RMSE, R2

    def __best_fit_slope_and_intercept(self, x, y):
        k = (((mean(x) * mean(y)) - mean(x * y)) /
             ((mean(x) * mean(x)) - mean(x * x)))
        b = mean(y) - k * mean(x)

        return k, b

    def point_scatter_density(self, label_y, pred_y, file_name=None):
        MAE, RMSE, R2 = self._calculate_stats(label_y, pred_y)

        # 计算点密度
        xy = np.vstack([label_y, pred_y])
        z = stats.gaussian_kde(xy)(xy)

        # 按点排序
        idx = z.argsort()
        x, y, z = label_y[idx], pred_y[idx], z[idx]
        k, b = self.__best_fit_slope_and_intercept(x, y)
        regression_line = []
        for a in x:
            regression_line.append((k * a + b))

        # 绘图
        fig, ax = plt.subplots(facecolor='w', figsize=(7, 6))
        scatter = ax.scatter(
            x, y,
            c=z * 1000,
            s=2,
            marker='o',
            edgecolors=None,
            cmap='jet'
        )
        cbar = plt.colorbar(
            scatter,
            orientation='vertical',
            pad=0.05,
            # label='Density',
            aspect=15
        )
        cbar.set_ticks(np.arange(0, 1, 0.2))  # TODO: y_lim
        cbar.ax.tick_params(
            left=True, leftlabel=False,
            right=True, rightlabel=True,
            size=3, width=1, direction='in',
            labelsize=10
        )

        ax.plot([0, 1], [0, 1], c='k', ls='--', lw=1, dashes=(16, 16))
        ax.plot(x, regression_line, c='k', ls='-', lw=1)
        ax.axis([0, 1, 0, 1])

        ax.set_xlabel("Real Refelctance")
        ax.set_ylabel("Predicted Reflectance")
        ax.set_xticks(np.arange(0, 1 + 1e-8, 0.2))
        ax.set_yticks(np.arange(0, 1 + 1e-8, 0.2))
        ax.tick_params(
            size=3,
            width=1,
            direction='in',
            labelsize=12,
            right=True,
            top=True
        )

        # ax.text(0.2, 0.85, r"$\rm N=%d$" % len(y), fontsize=28, transform=ax.transAxes)
        # ax.text(0.2, 0.80, r"$\rm Y=%.2fX+%.2f$" % (k, b), fontsize=28, transform=ax.transAxes)
        # ax.text(0.2, 0.75, r"$\rm R^2=%.2f$" % float(R2), fontsize=28, transform=ax.transAxes)
        # ax.text(0.2, 0.70, r"$\rm RMSE=%.2f$" % float(RMSE), fontsize=28, transform=ax.transAxes)
        # ax.text(0.2, 0.65, r"$\rm MAE=%.2f$" % float(MAE), fontsize=28, transform=ax.transAxes)

        file_name = file_name if file_name is not None else "scatter_density.png"
        save_path = os.path.join(self.save_dir, file_name)
        fig.savefig(save_path, dpi=600, bbox_inches='tight')

    def line_plot(self, metrics, key, col=3, file_name=None):
        fig = plt.figure(facecolor='w', figsize=(10, 6))
        num = len(key)
        row = num // col + 1 if (num > col) else 1
        for i in range(row):
            for j in range(col):
                count = i * col + j
                if count + 1 > num:
                    break
                ax = fig.add_subplot(row, col, count + 1)
                train_key = "train_{}".format(key[count])
                val_key = "valid_{}".format(key[count])
                y1 = metrics[train_key]
                y2 = metrics[val_key]
                x1 = [i for i in range(len(y1))]
                x2 = [i * 2 for i in range(len(y2))]
                ax.plot(x1, y1, label=train_key)
                ax.plot(x2, y2, label=val_key)
                ax.legend(ncol=1, frameon=False)
                ax.set_xlabel("epoch", fontsize=12)
                ax.set_ylabel(key[count], fontsize=12)

        file_name = file_name if file_name is not None else "metric_plot.png"
        save_path = os.path.join(self.save_dir, file_name)
        fig.savefig(save_path, dpi=600, bbox_inches='tight')

    def hist_plot(self, y):
        fig, ax = plt.subplots(facecolor='w', figsize=(7, 6))
        hist = ax.hist(y, bins=50)

        ax.tick_params(
            size=3,
            width=1,
            direction='in',
            labelsize=12,
        )
