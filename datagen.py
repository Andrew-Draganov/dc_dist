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

    
# obtain n uniformly sampled points within a d-sphere with a fixed radius around a given point. Assigns all points to given cluster
# code partially based on code provided here http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
def random_ball_num(center, radius, d, n, clunum):
    d = int(d)
    n = int(n)
    u = np.random.normal(0,1,(n,d+1))  # an array of d normally distributed random variables
    norm=np.sqrt(np.sum(u**2,1))
    r = np.random.random(n)**(1.0/d)
    normed = np.divide(u,norm[:, None])
    x= r[:, None]*normed
    x[:,:-1] = center + x[:,:-1]*radius
    x[:,-1] = clunum
    return x

# obtain n uniformly sampled points within a d-sphere with a fixed radius around a given point. Does not assign all points to given cluster
def random_ball_num_noclu(center, radius, d, n):
    u = np.random.normal(0,1,(n,d))  # an array of d normally distributed random variables
    norm=np.sqrt(np.sum(u**2,1))
    r = np.random.random(n)**(1.0/d)
    normed = np.divide(u,norm[:, None])
    x= r[:, None]*normed
    x = center + x*radius
    return x

# detect if point within minimal distance of a set of points
def tooclose (pos, cluid, points, labels, factors, mindist, d):
    #if (len(points) > 0):
    #    points = points[0]
    for j in range(len(points)):
        x = points[j]
        label = labels[j]
        factor = max(factors[label], factors[cluid])
        dist = 0
        mindistj = mindist * factor
        for i in range(d):
            dist += (x[i] - pos[i]) ** 2
            if dist **0.5 > mindistj:
                break
        if dist **0.5 < mindistj:
            return True
    return False

    
# obtain index of closesest point
def getclosest (pos, points, startdist, d):
    #if (len(points) > 0):
    #    points = points[0]
    mindist = startdist
    minj = 0
    for j in range(len(points)):
        x = points[j]
        dist = 0
        for i in range(d):
            dist += (x[i] - pos[i]) ** 2
            if dist **0.5 > mindist:
                break
        if dist **0.5 < mindist:
            mindist = dist ** 0.5
            minj = j
            
    return minj

    
# Seed Spreader improved on description from DBSCAN Revisited: Mis-Claim, Un-Fixability, and Approximation by Junhao Gan and Yufei Tao
def spreader_improv(n, d, cln, c_reset, min_size, num_noise, domain_size, r_sphere, r_shift, min_subspace,
                    num_connections, con_density, seed, vardensity):
    set_seed(seed)
    datanum = 0
    pos = np.random.random(d) * domain_size
    data = []
    nonoise = n - num_noise
    clunum = 1

    center_store_other = []
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
     
    clufactors = []
    if vardensity:
        for _ in restarts:
            factor = (np.random.rand() * 4.5) + 0.5
            factor = round(factor,2)
            print(factor)
            clufactors.append(factor)
        clufactors.append(0)
    else:
        for _ in restarts:
            clufactors.append(1)
        clufactors.append(0)
            
    center_store_new = [pos]
    center_clunum = [clunum-1]
    print(clunum)

    # print(restarts)

    nextCluster = False
    
    corecounter = 0
    startpos = pos
    
    while datanum < nonoise:
        corecounter = corecounter + 1
        c_rand_reset = np.ceil(np.random.normal(c_reset, 2, 1)).astype('int32')
        runlength = max(min(c_rand_reset[0], nonoise-datanum), c_reset/10)
        
        if (runlength >= restarts[clunum-1] - datanum and clunum <= cln-1):
            runlength = restarts[clunum-1] - datanum
            nextCluster = True
        
        r_sphere_i = r_sphere * clufactors[clunum-1]
        
        #print(runlength)
        data_new = random_ball_num(pos, r_sphere_i, d, runlength, clunum-1)
        data.extend(data_new.tolist())
        datanum = datanum + runlength
            
        if (nextCluster):
            center_store_other.extend(center_store_new)
            center_clunum.append(clunum-1)
            clunum += 1
            nextCluster = False
            #print(center_store_new)
            #print(center_store_other)
            center_store_new = [pos]
            startpos = pos
            print(clunum)
            pos = np.random.random(d)*domain_size
            attempts = 0
            while(tooclose(pos, clunum-1, center_store_other, center_clunum, clufactors, (r_sphere*4), d)):
                if (attempts > 100):
                    domain_size = domain_size + r_sphere*2
                    attempts = 0
                    print("expand domain size")
                pos = np.random.random(d)*domain_size
                attempts = attempts + 1

                

                        
        else:
            #print("Center_store")
            #print(center_store_new)
            #print("add pos")
            center_store_new.append(pos)
            center_clunum.append(clunum-1)
            #print(center_store_new)
            shift_dir = np.random.random(d) - 0.5
            pos_old = pos
            r_shift_i = r_shift * clufactors[clunum-1]
            pos = pos_old + (shift_dir/(np.sum((shift_dir*2)**2) **(0.5))*np.random.normal(r_shift_i, 2, 1))
            attempts = 0
            while(tooclose(pos, clunum-1, center_store_other, center_clunum, clufactors, (r_sphere*2.5), d)):
                if (attempts > 100):
                    pos_old = startpos
                    attempts = 0
                    print("reset to startpos")
                #print("tooclose")
                shift_dir = np.random.random(d) - 0.5
                pos = pos_old + (shift_dir/(np.sum(shift_dir**2) **(0.5))*np.random.normal(r_shift, 2, 1))
                attempts = attempts + 1

    data = np.array(data)
    #print(center_store_other)
    center_store_other.extend(center_store_new)
    #print("Centers")
    #print(center_clunum)
    #print(data)
    #for x in center_store_other:
    #    print(x)
        
    #print(corecounter)
    #print(len(center_store_other))
    #print(len(center_clunum))
    
    #print(len(np.array(center_store_other)))
    #print(np.array(center_store_other))
    print("Data generated")

    center_store_other_shift = np.array(center_store_other)
    
    maxdata = []
    for i in range(d):
        minimal = np.min(data[:, i])
        data[:, i] = data[:, i] - minimal
        center_store_other_shift[:, i] = center_store_other_shift[:,i] - minimal
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

    r_outlier = r_sphere
    
    for n in noise:
        if (tooclose(n, -1, center_store_other_shift, center_clunum, clufactors, r_outlier*1.5, d)):
            closestcore = getclosest(n, center_store_other_shift, r_outlier*8, d)
            n[-1] = center_clunum[closestcore]
    
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

    return data_final, center_store_other_shift
    
    
def run(args):
    seed = int(args[1])
    c_reset = 10
    r_sphere = 25
    num_con = 0
    den_con = 1
    num_nonoise = int(args[2])

    dim = int(args[3])
    cln = int(args[4])
    r_shift = 25

    num_noise = int (math.ceil(0.001 * num_nonoise))
    min_sub = dim
    vardensity = False

    if(len(args) > 5):
        if args[5] == "true" or args[5] == "True" or args[5] == '1':
            vardensity = bool(args[5])


    if (len(args) > 6):
        num_noise = int(args[6])

    if(len(args) > 8):
        c_reset = int(args[7])
        r_sphere = int(args[8])
        r_shift = int(args[8])


    if (len (args) > 10):
        num_con = int(args[9])
        den_con = int(args[10])


    if (len (args) > 11):
        min_sub = int(args[11])

    min_size = int(math.ceil(num_nonoise / cln**2))

    num = num_nonoise + num_noise
    domain_size = 500
    synthdata, centers = spreader_improv(num, dim, cln, c_reset, min_size, num_noise, domain_size, r_sphere, r_shift, min_sub, num_con, den_con, seed, vardensity)
    path = "data/synth"
        
    filename = "synth_data_" + str(num) + "_" + str(cln) + "_" + str(dim) + "_"
    if vardensity:
         filename = filename + "vardensity_"
    filename = filename + str(seed) + ".npy"
    np.save(os.path.join(path, filename), synthdata)
    
    color = plt.cm.tab20(np.linspace(0, 1, np.max(synthdata[:,-1]).astype('int32') +2))
    plt.figure(figsize=(15,15))
    plt.scatter(synthdata[:,0], synthdata[:,1], c=color[synthdata[:,2].astype('int32')], alpha=0.4)
    plt.scatter(centers[:,0], centers[:,1], c='black', alpha=1)

    plt.ylim(np.min(synthdata[:,1]-100), np.max(synthdata[:,1]+100))
    plt.xlim(np.min(synthdata[:,0]-100), np.max(synthdata[:,0]+100))

    plt.show()

if __name__ == '__main__':
    args = sys.argv
    
    # seed, number of points (without noise), dimensionality, cluster number
    # variable cluster density (default: False; active with "true", "True" or "1")
    # number of noise points (default: 0.001 * points)
    # cluster sphere point number, cluster sphere size = cluster sphere shift (default: 100,100)
    # number of connections (default: 0), density of connections
    # minimal number of dimensions for the cluster subspace (default: all dims -> no subspaces)

    print(args)

    run(args)