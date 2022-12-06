import numpy as np
import matplotlib.pyplot as plt

import matplotlib.font_manager as fm
font_times = fm.FontProperties(family='Times New Roman', stretch=0)

import time
current_time = time.strftime('%Y%m%d_%H%M%S')

import os
import sys

TITLE_FONT = 25
LABEL_FONT = 22
LEGEND_FONT = 20
TICK_FONT = 20


class PlotModule:
    def __init__(self):
        pass

    @staticmethod
    def create_multi_fig(fig_size=(55, 30), fig_dpi=30):
        fig = plt.figure(figsize=fig_size, dpi=fig_dpi)
        plt.subplots_adjust(left=0.03, bottom=0.05, right=0.99, top=0.95, wspace=0.25, hspace=0.25)
        return fig

    @staticmethod
    def plot_sub_fig(x_axis, y, fig, num_row, num_col, sub_fig,
                     title, x_label, y_label, x_lim=[0], y_lim=[0],
                     title_font=TITLE_FONT, label_font=LABEL_FONT, tick_font=TICK_FONT, legend_font=LEGEND_FONT):
        ax = fig.add_subplot(num_row, num_col, sub_fig)
        ax.set_title('Fig.' + str(sub_fig) + ' - ' + title,
                     fontsize=title_font, fontproperties=font_times)
        ax.set_xlabel(x_label, fontsize=label_font, fontproperties=font_times)
        ax.set_ylabel(y_label, fontsize=label_font, fontproperties=font_times)
        assert np.ndim(y) <= 2
        if y.ndim == 1:
            ax.plot(x_axis, y, 'b-', linewidth=1)
        else:
            for k in range(np.shape(y)[0]):
                ax.plot(x_axis, y[k], 'b-', linewidth=0.5)
        # plt.legend(fontsize=legend_font)
        plt.tick_params(labelsize=tick_font)
        plt.grid('on')
        if len(x_lim) == 2:
            plt.xlim(x_lim[0], x_lim[1])
        else:
            pass

        if len(y_lim) == 2:
            plt.ylim(y_lim[0], y_lim[1])
        else:
            pass
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

    @staticmethod
    def save_fig(path, fig, name, pdf_enable):
        if pdf_enable:
            fig.savefig(path + '/' + name + '.jpg', transparent=True)
            fig.savefig(path + '/' + name + '.pdf', transparent=True)
        else:
            fig.savefig(path + '/' + name + '.jpg', transparent=True)

    @staticmethod
    def save_data(path, data, name):
        np.save(path + '/' + name + '.npy', data)


