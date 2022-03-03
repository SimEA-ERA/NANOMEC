from __future__ import print_function
from __future__ import division
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;
from lammpsreader import LammpsTrajReader, isLammpsDataSorted, sortLammpsData
from auxil import *
from agg3dm import Aggregator3DM

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="Measure strain")
    argparser.add_argument("--trajectory", type=str, required=True, help="source lammps trajecotry")
    argparser.add_argument("--frame", type=int, required=True, help="frame of the trajectory to visualize")
    argparser.add_argument("--strain", type=str, required=True, help="file with calculated strain")
    argparser.add_argument("--aggsize", type=float, default=6.0, help="frame of the trajectory to visualize")
    argparser.add_argument("--row", type=int, default=0, help="row of strain to visualize")
    argparser.add_argument("--col", type=int, default=0, help="column of strain to visualize")
    argparser.add_argument("--outfname", type=str, required=False, help="column of strain to visualize")

    args = argparser.parse_args()

    if not os.path.isfile(args.trajectory):
        eprint('*** Could not locate trajectory file.')
        quit()
    if not os.path.isfile(args.strain):
        print('*** Could not locate strain file')
        quit()

    strain_data = np.load(args.strain)

    reader = LammpsTrajReader(args.trajectory)
    print("Loading trajectory frame")
    conf = seek_to_trajectory_step(reader, args.frame)    
    if conf == None:
        eprint("*** Could not seek to frame", args.frame)
        quit()
    if not isLammpsDataSorted(conf):
        print('Sorting frame')
        if not sortLammpsData(conf):
            eprint('*** Could not sort Lammps data')
            quit()

    box_params = calc_box_params(conf)
    xlo = box_params[0,0]
    xhi = box_params[0,1]
    ylo = box_params[1,0]
    yhi = box_params[1,1]
    zlo = box_params[2,0]
    zhi = box_params[2,1]

    # print('xlo:', xlo, 'xhi:', xhi)
    # print('ylo:', ylo, 'yhi:', yhi)
    # print('zlo:', zlo, 'zhi:', zhi)

    xs = conf['x']
    ys = conf['y']
    zs = conf['z']
    nats = len(xs)

    agg = Aggregator3DM(xlo,xhi,ylo,yhi,zlo,zhi, args.aggsize, (3,3))
    
    for iat in range(nats):
        p = np.array([xs[iat], ys[iat], zs[iat]])
        s = strain_data[iat,:,:]
        agg.add(p,s)

    averages = agg.calc_averages_by_count()
    nx = averages.shape[0]
    ny = averages.shape[1]
    nz = averages.shape[2]
    iz = int(nz//2)
    plot_data = np.zeros((ny,nx))
    for ix in range(nx):
        for iy in range(ny):
            plot_data[ny-iy-1,ix] = averages[ix,iy,iz,args.row,args.col]

    fig = plt.figure(figsize=(8,6))
    sns.set_theme()
    ax = sns.heatmap(plot_data, annot=True, cmap="mako", annot_kws={"fontsize":7})
    if args.outfname is not None:
        plt.savefig(args.outfname)
    else:
        plt.show()


