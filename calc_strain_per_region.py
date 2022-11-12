from __future__ import print_function
from __future__ import division
import numpy as np
from lammpsreader import LammpsTrajReader, isLammpsDataSorted, sortLammpsData
import scipy.optimize 
from numba import njit
from auxil import *
import argparse
import os
import time
from neicelllist import NeiCellList

natsInSil = 3069
natsInPb = 400
nPbsPerCell = 23

def calc_sil_start_ends():
    natsInCell = natsInSil + natsInPb * nPbsPerCell
    ses = []
    s = 0
    e = s + natsInSil - 1
    ses.append([s, e])      # warning, zero based
    return np.array(ses, np.uint)

def calc_pb_start_ends_full():
    natsInCell = natsInSil + natsInPb * nPbsPerCell
    ses = []
    iStart = 0
    s = iStart + natsInSil 
    e = s + nPbsPerCell * natsInPb - 1
    ses.append([s, e])      # warning, zero based
    return np.array(ses, np.uint)

def calc_np_cms(conf, silStartEnds):
    masses = conf['mass']
    uxs = conf['xu']
    uys = conf['yu']
    uzs = conf['zu']

    nNps = silStartEnds.shape[0]    
    npCms = np.zeros((nNps, 3))

    for iNp in range(nNps):
        istart = int(silStartEnds[iNp,0])
        iend = int(silStartEnds[iNp,1])
        cm = np.zeros((3))
        m = 0.0
        for iAt in range(istart, iend+1):
            cm += masses[iAt] * np.array([uxs[iAt], uys[iAt], uzs[iAt]])
            m += masses[iAt]
        npCms[iNp] = cm / m

    return npCms

def isAtomOfMolecule(index, startEnds):
    for se in startEnds:
        if index >= se[0] and index <= se[1]:
            return True
    return False        

def select_atoms(coords, rin, rout, cm, box_size, startEnds):
    xu = coords['xu']
    yu = coords['yu']
    zu = coords['zu']
    n = len(xu)
    atoms = []
    for i in range(n):
        if isAtomOfMolecule(i, startEnds):
            pos = np.array([xu[i], yu[i], zu[i]])
            dx = periodicDistance(xu[i], cm[0], box_size[0])
            if dx > 0:
                dr = periodicDistance3(pos, cm, box_size)
                if dr >= rin and dr < rout:
                    atoms.append(i)
    return atoms

def pick_conf_coords_full(conf):
    return {'x': conf['x'].copy(), 'y': conf['y'].copy(), 'z': conf['z'].copy(),
            'xu': conf['xu'].copy(), 'yu': conf['yu'].copy(), 'zu': conf['zu'].copy()}

def make_nei_cell_list(xs,ys,zs,box_size,rcut):
    ncl = NeiCellList()
    ncl.makeCells(box_size[0], box_size[1], box_size[2], rcut)
    ncl.makeList(xs,ys,zs)
    return ncl

def calc_dxs(atm, neins, coords, box_size):
    x = coords['x']
    y = coords['y']
    z = coords['z']
    posm = np.array([x[atm], y[atm], z[atm]])
    result = np.zeros((3,len(neins)))
    i = 0
    for nein in neins:
        posn = np.array([x[nein], y[nein], z[nein]])
        dx = periodicDistanceVec(posn,posm,box_size)
        result[:,i] = dx
        i += 1
    return result

def calc_strain(intf_atoms, blk_atoms, prevCoords, prevBoxOffest, prevBoxSize, newCoords, newBoxSize, rcut):
    intf_atoms_set = set(intf_atoms)
    blk_atoms_set = set(blk_atoms)
    all_atoms_set = intf_atoms_set.union(blk_atoms_set)

    progressSyncSteps = 200
    print('Calculating strain...')
    prevProgressStr = ''
    startTime = time.time()

    x = prevCoords['x'] - prevBoxOffest[0]
    y = prevCoords['y'] - prevBoxOffest[1]
    z = prevCoords['z'] - prevBoxOffest[2]

    intf_strain = np.zeros((3,3))
    blk_strain = np.zeros((3,3))

    ncl = make_nei_cell_list(x,y,z,prevBoxSize,rcut)

    cnt = 0
    for atm in all_atoms_set:
        neis = ncl.calcNeisOf(atm,x,y,z,rcut)
        DXS = calc_dxs(atm, neis, prevCoords, prevBoxSize)
        dxs = calc_dxs(atm, neis, newCoords, newBoxSize)
        f0 = np.linalg.solve(np.kron(np.eye(3),DXS@DXS.T),(dxs@DXS.T).flatten()).reshape(3,3)
        e = 0.5 * (np.matmul(f0.transpose(),f0) - np.identity(3))

        if atm in intf_atoms_set:
            intf_strain += e
        if atm in blk_atoms_set:
            blk_strain += e

        cnt += 1

        if cnt % progressSyncSteps == 0:
            percentDone = float(cnt)/float(len(all_atoms_set)) * 100.0
            progressStr = '%.1f%%' % percentDone
            if progressStr != prevProgressStr:
                if percentDone > 0.0:
                    timeElapsed = time.time() - startTime
                    timeRemaining = timeElapsed * (100.0 / percentDone - 1.0)
                    print('Done', progressStr, 'time remaining is', time_length_str(timeRemaining))
                    prevProgressStr = progressStr
    
    return intf_strain / len(intf_atoms_set), blk_strain / len(blk_atoms_set)

def main_function():
    argparser = argparse.ArgumentParser(description="Measure interface and bulk strain")
    argparser.add_argument("--trajectory", type=str, required=True, help="source lammps trajecotry")
    argparser.add_argument("--intf_rin", type=float, default=21.0, help="interface rin")
    argparser.add_argument("--intf_rout", type=float, default=25.0, help="interface rout")
    argparser.add_argument("--blk_rin", type=float, default=25.0, help="bulk rin")
    argparser.add_argument("--blk_rout", type=float, default=29.0, help="bulk rout")
    argparser.add_argument("--rcut", type=float, default=8.0, help="cutoff distance")
    argparser.add_argument("--outfile", type=str, default='per_region.txt', help="destination filename")
    args = argparser.parse_args()

    if not os.path.isfile(args.trajectory):
        eprint("*** Source file " + args.trajectory + " does not exist")
        quit()

    reader = LammpsTrajReader(args.trajectory)
    print("Loading starting frame")
    start_conf = reader.readNextStep()
    if start_conf == None:
        eprint("*** Could not read starting step")
        quit()

    if not isLammpsDataSorted(start_conf):
        print('Sorting frame')
        if not sortLammpsData(start_conf):
            eprint('*** Could not sort Lammps data')
            quit()

    prevCoords = pick_conf_coords_full(start_conf)
    prevBoxOffset = calc_box_offset(start_conf)
    prevBoxSize = calc_box_size(start_conf)

    silStartEnds = calc_sil_start_ends()
    pbStartEnds = calc_pb_start_ends_full()
    cm = calc_np_cms(start_conf, silStartEnds)[0]
    intf_atoms = select_atoms(prevCoords, args.intf_rin, args.intf_rout, cm, prevBoxSize, pbStartEnds)
    blk_atoms = select_atoms(prevCoords, args.blk_rin, args.blk_rout, cm, prevBoxSize, pbStartEnds)

    outfile = open(args.outfile, 'w')

    while True:
        end_conf = reader.readNextStep()
        if end_conf is None:
            print("Finished reading from trajectory")
            break        

        if not isLammpsDataSorted(end_conf):
            print('Sorting frame')
            if not sortLammpsData(end_conf):
                eprint('*** Could not sort Lammps data')
                quit()

        print("Read conf", end_conf['step_no'])

        newCoords = pick_conf_coords_full(end_conf)
        newBoxSize = calc_box_size(end_conf)

        intf_strain_per_atom, blk_strain_per_atom = calc_strain(intf_atoms, blk_atoms, prevCoords, prevBoxOffset, prevBoxSize, newCoords, newBoxSize, args.rcut)    
        cm = calc_np_cms(end_conf, silStartEnds)[0]

        outfile.write('%d %s %s\n' % (end_conf['step_no'],
                                      np.format_float_scientific(intf_strain_per_atom[0,0]), 
                                      np.format_float_scientific(blk_strain_per_atom[0,0])))

    outfile.close()
    print('Goodbye')


if __name__ == "__main__":
    main_function()
