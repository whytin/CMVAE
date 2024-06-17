import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

def draw_plot_line(dt, indicator, save_name):
    colors = ['g', 'c', 'm', 'y', 'k', 'gray', 'brown',  'orange', 'b', 'r']
    markers = ['o', 'v', '>', '<', 's', 'p', '*', '^', '+', 'D']
    labels = ['BSV', 'Concat', 'DCCA', 'DCCAE', 'VCCAP',  'UEAF',  'CPM', 'COMPLETER', 'VMVAE', 'CMVAE']
    model_ind = {0:'BSV', 1:'Concat', 2:'DCCA', 3:'DCCAE', 4:'VCCAP',  5:'UEAF',  6:'CPM', 7:'COMPLETER', 8:'VMVAE', 9:'CMVAE'}

    plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5])
    X = [0.1, 0.2, 0.3, 0.4, 0.5]
    ax = plt.subplot(111)
    for i in range(10):
        plt.plot(X, dt[model_ind[i]][indicator]['means'], c=colors[i], label=labels[i])
        plt.scatter(X, dt[model_ind[i]][indicator]['means'], c=colors[i], marker=markers[i])
        plt.fill_between(X, dt[model_ind[i]][indicator]['lower_bound'], dt[model_ind[i]][indicator]['upper_bound'], facecolor=colors[i], alpha=0.2)
        plt.legend(loc='lower left' ,fancybox=True, framealpha=0.5)
    plt.xlabel('Missing Rate')
    plt.ylabel(indicator)

    #box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig('./visualization/'+save_name)


def main():
    path='performance_std.json'
    with open(path, 'r') as f:
        data = json.load(f)
    draw_plot_line(data['UCI'], 'NMI', 'UCI_NMI.png')
      




if __name__ == '__main__':
    main()