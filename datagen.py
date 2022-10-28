import numpy as np
import pandas as pd
from random import random
import math
import sys
import os

import matplotlib.pyplot as plt
import seaborn as sns

def set_seed(i):
    np.random.seed(i)

def random_ball_num(center, radius, d, n, clunum):
    u = np.random.normal(0,1,(n,d+1))  # an array of d normally distributed random variables
    norm=np.sqrt(np.sum(u**2,1))
    r = np.random.random(n)**(1.0/d)
    normed = np.divide(u,norm[:, None])
    x= r[:, None]*normed
    x[:,:-1] = center + x[:,:-1]*radius
    x[:,-1] = clunum
    return x

def random_ball_num_noclu(center, radius, d, n):
    u = np.random.normal(0,1,(n,d))  # an array of d normally distributed random variables
    norm=np.sqrt(np.sum(u**2,1))
    r = np.random.random(n)**(1.0/d)
    normed = np.divide(u,norm[:, None])
    x= r[:, None]*normed
    x = center + x*radius
    return x


def spreader_improv(n, d, cln, c_reset, min_size, num_noise, domain_size, r_sphere, r_shift, min_subspace,
                    num_connections, con_density, seed):
    set_seed(seed)
    datanum = 0
    pos = np.random.random(d) * domain_size
    data = []
    nonoise = n - num_noise
    clunum = 1

    while (True):
        restarts = np.ceil(np.sort(np.random.random(cln) * nonoise)).astype('int32')
        dist = 0
        newrun = False
        for i in range(cln):
            if i == 0:
                dist = restarts[i]
            elif i == cln - 1:
                dist = nonoise - restarts[i]
            else:
                dist = restarts[i] - restarts[i - 1]
            if dist < min_size:
                newrun = True
                break
        if not newrun:
            break

    # print(restarts)

    nextCluster = False

    while datanum < nonoise:
        c_rand_reset = np.ceil(np.random.normal(c_reset, 2, 1)).astype('int32')
        runlength = max(min(c_rand_reset[0], nonoise - datanum), c_reset / 10)

        if (runlength >= restarts[clunum - 1] - datanum and clunum <= cln - 1):
            runlength = restarts[clunum - 1] - datanum
            nextCluster = True

        # print(runlength)

        data_new = random_ball_num(pos, r_sphere, d, runlength, clunum - 1)
        data.extend(data_new.tolist())
        datanum = datanum + runlength

        if (nextCluster):
            clunum += 1
            pos = np.random.random(d) * domain_size
            nextCluster = False
        else:
            shift_dir = np.random.random(d) - 0.5
            pos = pos + (shift_dir / (np.sum(shift_dir ** 2) ** (0.5)) * np.random.normal(r_shift, 2, 1))

    data = np.array(data)

    print("Data generated")

    maxdata = []
    for i in range(d):
        data[:, i] = data[:, i] - np.min(data[:, i])
    noise = np.random.random([n - nonoise, d + 1])
    for i in range(d):
        maxdata.append(np.max(data[:, i]))
        noise[:, i] = noise[:, i] * maxdata[i]
    noise[:, -1] = -1

    subspaces = {}
    for c in range(cln):
        c_subspace_num = np.round(np.random.random(1) * (d - min_subspace)).astype(int)
        c_subspace = np.random.choice(range(d), c_subspace_num, replace=False)
        subspaces[c] = c_subspace

    rand_subspace = np.random.random([len(data), d])
    for xi in range(len(data)):
        x = data[xi]
        if x[-1] > -1:
            for i in subspaces[x[-1]]:
                x[i] = rand_subspace[xi, i] * maxdata[i]

    print("Subspaces applied")

    r_outlier = (2 * r_shift) ** 2
    for n in noise:
        for x in data:
            dist = 0
            for i in range(d):
                dist += (x[i] - n[i]) ** 2
                if dist > r_outlier:
                    break
            if dist < r_outlier:
                n[-1] = x[-1]
                # print(n[-1])
                break

    data_main = data.tolist()
    data_main.extend(noise.tolist())
    data_main = np.array(data_main)

    print("Outliers generated")

    conpoints = []
    for i in range(num_connections):
        curconpoints = []
        startclu = np.random.choice(range(cln), 1)
        stopclu = np.random.choice(range(cln), 1)
        while (stopclu == startclu):
            stopclu = np.random.choice(range(cln), 1)

        # print(np.where(data_main[:,d] == startclu)[0])
        startpoints = data[np.where(data[:, d] == startclu)[0]][:, :-1]
        startpoint = startpoints[np.random.choice(len(startpoints), 1)][0]

        stoppoints = data[np.where(data[:, d] == stopclu)[0]][:, :-1]

        np.delete(stoppoints, -1, axis=0)

        target = stoppoints[np.random.choice(len(stoppoints), 1)][0]
        newcenter = startpoint

        dist = np.sum((target - newcenter) ** 2) ** (0.5)
        while (dist > r_shift):
            shift_dir = target - newcenter
            newcenter = newcenter + (shift_dir / (np.sum(shift_dir ** 2) ** (0.5)) * np.random.normal(r_shift, 2, 1))
            newpoints = random_ball_num_noclu(newcenter, r_sphere, d, con_density)

            target = stoppoints[np.random.choice(len(stoppoints), 1)][0]
            curconpoints.extend(newpoints.tolist())
            for pottarget in stoppoints:
                dist = min(dist, np.sum((pottarget - newcenter) ** 2) ** (0.5))

        #print(curconpoints[0])
        for point in curconpoints:
            dist = r_outlier * 2
            for potstart in startpoints:
                dist = min(dist, np.sum((potstart - point) ** 2) ** (0.5))
            if (dist > r_shift):
                # if(True):
                conpoints.append(point)

        print("Connection " + str(i+1) + " built")

    conpoints_array = np.column_stack((conpoints, [-1] * len(conpoints)))

    data_final = data_main.tolist()
    data_final.extend(conpoints_array.tolist())
    data_final = np.array(data_final)

    return data_final

if __name__ == '__main__':
    args = sys.argv
#n, d, cln, c_reset, min_size, p_noise, domain_size, r_sphere, r_shift, min_subspace,
#                    num_connections, con_density, seed
    print(args)

    seed = int(args[1])
    c_reset = 100
    r_sphere = 100
    r_shift = 100
    num_con = 0
    den_con = 1
    num_nonoise = int(args[2])

    dim = int(args[3])
    cln = int(args[4])

    num_noise = int (math.ceil(0.2 * num_nonoise / cln))
    min_sub = dim

    if (len(args) > 5):
        num_noise = int(args[5])

    if(len(args) > 8):
        c_reset = int(args[6])
        r_sphere = int(args[7])
        r_shift = int(args[8])


    if (len (args) > 10):
        num_con = int(args[9])
        den_con = int(args[10])


    if (len (args) > 11):
        min_sub = int(args[11])

    min_size = int(math.ceil(num_nonoise / cln**2))

    num = num_nonoise + num_noise
    domain_size = 10000

    synthdata = spreader_improv(num, dim, cln, c_reset, min_size, num_noise, domain_size, r_sphere, r_shift, min_sub, num_con, den_con, seed)

    #color = plt.cm.rainbow(np.linspace(0, 1, np.max(synthdata[:,-1]).astype('int32') +2))
    path = "data/synth"
    filename = "synth_data_" + str(num) + "_" + str(cln) + "_" + str(dim) + "_" + str(seed) + ".npy"
    np.save(os.path.join(path, filename), synthdata)


    #print(color[x3[:,2].astype('int32')])

    #plt.figure(figsize=(10,10))
    #plt.scatter(synthdata[:,0], synthdata[:,1], c=color[synthdata[:,2].astype('int32')], alpha=0.1)
    #plt.ylim(np.min(synthdata[:,1]-100), np.max(synthdata[:,1]+100))
    #plt.xlim(np.min(synthdata[:,0]-100), np.max(synthdata[:,0]+100))

    #plt.show()