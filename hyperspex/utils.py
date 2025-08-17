import numpy as np

def merge_spectra(x, y, increasing_x=True):
    index_list = []
    counting = True
    for i in range(1, len(x)):
        if increasing_x:
            if x[i]<x[:i].max():
                if counting:
                    index_list.append(i-1)
                    counting = False
            else:
                if not counting:
                    index_list.append(i)
                counting = True
        else:
            if x[i]>x[:i].min():
                if counting:
                    index_list.append(i-1)
                    counting = False
            else:
                if not counting:
                    index_list.append(i)
                counting = True
    new_x = x[0:index_list[0]]
    new_y = y[0:index_list[0]]
    for i in range(1, int(len(index_list)/2)):
        new_x = np.append(new_x, x[index_list[i*2-1]:index_list[i*2]])
        new_y = np.append(new_y, y[index_list[i*2-1]:index_list[i*2]]+new_y[-1]-y[index_list[i*2-1]])
    return [new_x, new_y]

def remove_spikes(data, iterations, e=1e-2):
    for n in range(iterations):
        for i in range(1, len(data)-1):
            if np.abs(data[i+1]-data[i])>e:
                data[i] = data[i-1]
    return data