import matplotlib.pyplot as plt
import pickle
import os

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))

allfiles = os.listdir()
for i in allfiles[:-1]:
    res = pickle.load(open(i, 'rb'))
    x = []
    y = []
    z = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    for key in res.keys():
        x.append(key)
        y.append(res[key])
    j = i.split('_')
    print(i)
    print(x)
    print(y)
    dataset_name = j[0]
    model = (j[1].split('.'))[0]
    if model == 'AirGNN':
        marker = 'o'
        color = 'red'
    elif model == 'APPNP':
        marker = '^'
        color = 'purple'
    elif model == 'GAT':
        marker = '*'
        color = 'blue'
    elif model == 'GCN':
        marker = 's'
        color = 'orange'
    else:
        continue
    if dataset_name == 'Cora':
        axes[0].plot(z, y, marker=marker, label=model, c=color)
    if dataset_name == 'CiteSeer':
        axes[1].plot(z, y, marker=marker, label=model, c=color)


fig.tight_layout(pad=3)
axes[0].legend()
axes[1].legend()
axes[0].set_xlabel('Ratio of noisy nodes')
axes[1].set_xlabel('Ratio of noisy nodes')
axes[0].set_ylabel('Accuracy')
axes[1].set_ylabel('Accuracy')
axes[1].grid(True)
axes[0].grid(True)
axes[1].set_xticks(z, labels=x)
axes[0].set_xticks(z, labels=x)
plt.savefig('noisy.png')
plt.show()