import monkdata as m
import dtree as d
import random
import matplotlib.pyplot as plt

fracaction = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]

# Example of how to call use partition


def bestTree(treeList):
    "Get tree with the highest score (most correct)"
    scoremap = dict()
    for tree in treeList:
        score = d.check(tree, monk1val)
        scoremap[score] = tree

    return scoremap.get(max(scoremap.keys()))

non_pruned = []
pruned = []
for f in fracaction:
    print(f)
    n = 50000
    tot = 0
    tot_pruned = 0
    for _ in range(n):
        monk1train, monk1val = partition(m.monk1, f) # Change the input to dataset

        t = d.buildTree(monk1train, m.attributes)
        tot += (1 - d.check(t, monk1val))  # Un-prouned value
        prune_old = d.check(t, monk1val)  # Värdet på förra trädet
        t_new = bestTree(d.allPruned(t))
        prune_new = d.check(t_new, monk1val)

        while prune_new > prune_old:
            t = t_new
            t_new = bestTree(d.allPruned(t))
            prune_old = d.check(t, monk1val)
            prune_new = d.check(t_new, monk1val)

        tot_pruned += (1 - prune_new)

    mean_no = tot / n
    non_pruned.append(mean_no)

    mean_pruned = tot_pruned / n
    pruned.append(mean_pruned)

    print("NON-PRUNED SCORE", mean_no)
    print("PRUNED SCORE", mean_pruned)

plt.ylabel("Classification error")
plt.xlabel("Fraction")
plt.plot(fracaction, non_pruned, marker="o", linestyle="dotted", label="Non-pruned")
plt.plot(fracaction, pruned, marker="v", linestyle="dotted", label="Pruned")
plt.title("MONK-1") # Change this with dataset
plt.legend()
plt.show()



hej = 0x555555555280



