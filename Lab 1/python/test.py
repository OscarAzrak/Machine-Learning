import monkdata as m
import dtree as d
import random
import matplotlib.pyplot as plt
from prettytable import PrettyTable

import numpy as np

def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]

def prune_tree(data, test):
    fractions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    pruned = []
    #Check which fraction has best error rate
    for frac in fractions:
        train, val = partition(data, frac)
        tree = d.buildTree(train, m.attributes)
        forest = d.allPruned(tree)
        best_perf = d.check(tree, val)

        best_tree = tree
        temp_tree = 0
        for t in forest:
            temp_perf = d.check(t, val)
            if temp_perf > best_perf:
                best_perf = temp_perf
                best_tree = t
        pruned.append(1-d.check(best_tree, test))
    return pruned


def evaluate_pruning():
    fractions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    monk1_pruned = []
    monk3_pruned = []


    for i in range(100):
        monk1_pruned.append(prune_tree(m.monk1, m.monk1test))
        monk3_pruned.append(prune_tree(m.monk3, m.monk3test))

    monk1_pruned = np.transpose(monk1_pruned)
    monk3_pruned = np.transpose(monk3_pruned)

    mean1 = np.mean(monk1_pruned, axis=1)
    mean3 = np.mean(monk3_pruned, axis=1)
    std1 = np.std(monk1_pruned, axis=1)
    std3 = np.std(monk3_pruned, axis=1)

    stat_table = PrettyTable(['Dataset/Stat', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6'])
    stat_table.add_row(np.concatenate((['MONK-1 - MEAN'], np.around(mean1, decimals=6)), axis=0))
    stat_table.add_row(np.concatenate((['MONK-3 - MEAN'], np.around(mean3, decimals=6)), axis=0))
    stat_table.add_row(np.concatenate((['MONK-1 - STDEV'], np.around(std1, decimals=6)), axis=0))
    stat_table.add_row(np.concatenate((['MONK-1 - STDEV'], np.around(std3, decimals=6)), axis=0))
    print(stat_table)

    complete_tree1 = d.buildTree(m.monk1, m.attributes)
    complete_tree3 = d.buildTree(m.monk3, m.attributes)

    prn_table = PrettyTable(['Dataset', 'Error on Complete Tree', 'Error on Pruned Tree (mean)'])
    prn_table.add_row(['MONK-1', 1 - d.check(complete_tree1, m.monk1test), np.amin(mean1)])
    prn_table.add_row(['MONK-3', 1 - d.check(complete_tree3, m.monk3test), np.amin(mean3)])
    print(prn_table)

    plt.plot(fractions, mean1, color='#49abc2', marker='o', label="Means")
    plt.title("Mean Error vs Fractions on MONK-1")
    plt.xlabel("Fractions")
    plt.ylabel("Means of Error")
    plt.legend(loc='upper right', frameon=False)
    plt.show()

    plt.plot(fractions, mean3, color='#fe5f55', marker='o', label="Means")
    plt.title("Mean Error vs Fractions on MONK-3")
    plt.xlabel("Fractions")
    plt.ylabel("Means of Error")
    plt.legend(loc='upper right', frameon=False)
    plt.show()

evaluate_pruning()
