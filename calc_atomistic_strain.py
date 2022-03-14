from __future__ import print_function
from __future__ import division
import numpy as np
from lammpsreader import LammpsTrajReader, isLammpsDataSorted, sortLammpsData
import scipy.optimize 
from numba import njit
from auxil import *
from mpi4py import MPI
import argparse
import os
import time
from neicelllist import NeiCellList, check_same_neis

mpi_comm = None
mpi_rank = None
mpi_size = None
g_check_only = False
g_at_range = None

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

@njit()
def obj_func(f,DXS,dxs):
    W = 0.0
    for i in range(dxs.shape[1]):
        dx0 = dxs[0,i]
        dx1 = dxs[1,i]
        dx2 = dxs[2,i]
        DX0 = DXS[0,i]
        DX1 = DXS[1,i]
        DX2 = DXS[2,i]
        tmp0 = dx0 - f[0]*DX0 - f[1]*DX1 - f[2]*DX2
        tmp1 = dx1 - f[3]*DX0 - f[4]*DX1 - f[5]*DX2
        tmp2 = dx2 - f[6]*DX0 - f[7]*DX1 - f[8]*DX2
        W += (tmp0*tmp0 + tmp1*tmp1 + tmp2*tmp2)
    return W

def calc_strain(startAt, atCount, prevCoords, prevBoxOffest, prevBoxSize, newCoords, newBoxSize, rcut):
    global mpi_comm, mpi_rank, g_check_only, g_at_range
    
    minAtCount = mpi_comm.allreduce(atCount, MPI.MIN)
    progressSyncSteps = 500
    if mpi_rank == 0:
        print('Calculating strain...')
        prevProgressStr = ''
        startTime = time.time()

    x = prevCoords['x'] - prevBoxOffest[0]
    y = prevCoords['y'] - prevBoxOffest[1]
    z = prevCoords['z'] - prevBoxOffest[2]
    strains = np.zeros((atCount,3,3), dtype=np.float64)
    residuals = np.zeros((atCount,), dtype=np.float64) 

    ncl = make_nei_cell_list(x,y,z,prevBoxSize,rcut)

    for atm in range(startAt, startAt+atCount):
        at_in_range = g_at_range is None or (atm >= g_at_range[0] and atm <= g_at_range[1])
        if not g_check_only and at_in_range:
            neis = ncl.calcNeisOf(atm,x,y,z,rcut)
            DXS = calc_dxs(atm, neis, prevCoords, prevBoxSize)
            dxs = calc_dxs(atm, neis, newCoords, newBoxSize)
            f0 = np.linalg.solve(np.kron(DXS@DXS.T,np.eye(3)),(dxs@DXS.T).flatten()).reshape(3,3)
            res0 = obj_func(f0.flatten(),DXS,dxs)
            # res = scipy.optimize.minimize(obj_func, f0, args=(DXS,dxs), method='BFGS')
            # f = res.x.reshape(3,3)
            e = 0.5 * (np.matmul(f0.transpose(),f0) - np.identity(3))
        else:
            e = np.eye(3) * atm
            res0 = atm

        zeroBasedIndex = atm-startAt 
        strains[zeroBasedIndex,:,:] = e
        residuals[zeroBasedIndex] = res0
        
        if zeroBasedIndex < minAtCount and zeroBasedIndex % progressSyncSteps == 0:
            mpi_comm.Barrier()
            if mpi_rank == 0:
                percentDone = float(zeroBasedIndex)/float(minAtCount) * 100.0
                progressStr = '%.1f%%' % percentDone
                if progressStr != prevProgressStr:
                    if percentDone > 0.0:
                        timeElapsed = time.time() - startTime
                        timeRemaining = timeElapsed * (100.0 / percentDone - 1.0)
                        print('Done', progressStr, 'time remaining is', time_length_str(timeRemaining))
                        prevProgressStr = progressStr
    
    return strains, residuals   

def main_function():
    global mpi_comm, mpi_rank, mpi_size, g_check_only, g_at_range

    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()

    if mpi_rank == 0:
        argparser = argparse.ArgumentParser(description="Measure strain")
        argparser.add_argument("--trajectory", type=str, required=True, help="source lammps trajecotry")
        argparser.add_argument("--startframe", type=int, default=0, help="initial frame of the trajectory")
        argparser.add_argument("--endframes", type=int, nargs='+', required=True, help="frames for which to calculate the strain")
        argparser.add_argument("--rcut", type=float, default=8.0, help="cutoff distance")
        argparser.add_argument("--dest_folder", type=str, required=False, help="destination folder")
        argparser.add_argument("--atrange", type=int, nargs=2, help="calculate only this range of atoms")
        argparser.add_argument("--check_only", action="store_const", const=True, help="does not perform the actual calculations")
        args = argparser.parse_args()

        if not os.path.isfile(args.trajectory):
            eprint("*** Source file " + args.trajectory + " does not exist")
            mpi_comm.Abort()

        cmdparams = {'trajectory': args.trajectory, 'startframe': args.startframe, 
                     'endframes': args.endframes, 'rcut': args.rcut, 
                     'dest_folder': args.dest_folder, 'atrange': args.atrange, 
                     'check_only': args.check_only == True}
        mpi_comm.bcast(cmdparams, root=0)
    else:
        cmdparams = mpi_comm.bcast(None, root=0)

    trajectory_filename = cmdparams['trajectory']
    start_frame = cmdparams['startframe']
    end_frames = sorted(cmdparams['endframes'])
    rcut = cmdparams['rcut']
    dest_folder = cmdparams['dest_folder']

    g_at_range = cmdparams['atrange']
    if g_at_range is not None: g_at_range.sort()
    g_check_only = cmdparams['check_only']

    if mpi_rank == 0:
        reader = LammpsTrajReader(trajectory_filename)
        print("Loading starting frame")
        start_conf = seek_to_trajectory_step(reader, start_frame)
        if start_conf == None:
            eprint("*** Could not seek to starting step", start_frame)
            mpi_comm.Abort()

        if not isLammpsDataSorted(start_conf):
            print('Sorting frame')
            if not sortLammpsData(start_conf):
                eprint('*** Could not sort Lammps data')
                mpi_comm.Abort()

        mpi_comm.bcast(start_conf, root=0)
    else:
        start_conf = mpi_comm.bcast(None, root=0)

    prevCoords = pick_conf_coords(start_conf)
    prevBoxOffset = calc_box_offset(start_conf)
    prevBoxSize = calc_box_size(start_conf)

    nats = len(prevCoords['x'])
    nats_per_rank = nats // mpi_size
    nats_per_last_rank = nats_per_rank + nats % mpi_size

    at_counts = []
    for irank in range(mpi_size):
        if irank < mpi_size-1:
            at_counts.append(nats_per_rank)
        else:
            at_counts.append(nats_per_last_rank)
    at_displacements = [0]
    for irank in range(1,mpi_size):
        at_displacements.append(at_displacements[irank-1] + at_counts[irank-1])

    data_counts = [at_count*3*3 for at_count in at_counts]
    data_displacements = [at_displacement*3*3 for at_displacement in at_displacements]

    residual_data_counts = [at_count for at_count in at_counts]
    residual_data_displacements = [at_displacement for at_displacement in at_displacements]

    for end_frame in end_frames:
        if mpi_rank == 0:        
            end_conf = seek_to_trajectory_step(reader, end_frame)
            
            if end_conf == None:
                eprint("*** Could not seek to ending step", end_frame)
                mpi_comm.Abort()

            if not isLammpsDataSorted(end_conf):
                print('Sorting frame')
                if not sortLammpsData(end_conf):
                    eprint('*** Could not sort Lammps data')
                    mpi_comm.Abort()

            mpi_comm.Barrier()
            mpi_comm.bcast(end_conf, root=0)
        else:
            mpi_comm.Barrier()
            end_conf = mpi_comm.bcast(None, root=0)

        newCoords = pick_conf_coords(end_conf)
        newBoxSize = calc_box_size(end_conf)

        startAt = at_displacements[mpi_rank]
        atCount = at_counts[mpi_rank]

        if mpi_rank == 0:
            startTime = time.time()

        strains, residuals = calc_strain(startAt, atCount, prevCoords, prevBoxOffset, prevBoxSize, newCoords, newBoxSize, rcut)

        if mpi_rank == 0:
            global_strains = np.zeros((nats,3,3), dtype=np.float64)
            global_residuals = np.zeros((nats,), dtype=np.float64)
        else:
            global_strains = None 
            global_residuals = None

        mpi_comm.Gatherv(sendbuf=strains, recvbuf=(global_strains, data_counts, data_displacements, MPI.DOUBLE), root=0)
        mpi_comm.Gatherv(sendbuf=residuals, recvbuf=(global_residuals, residual_data_counts, residual_data_displacements, MPI.DOUBLE), root=0)

        if mpi_rank == 0:
            timeElapsed = time.time() - startTime
            print('Calculation took', time_length_str(timeElapsed))

        if mpi_rank == 0:
            if dest_folder is None: dest_folder = ''
            filename = os.path.join(dest_folder, 'strain_from_%d_to_%d.npy' % (start_frame, end_frame))
            np.save(filename, global_strains)
            print('Saved strain to', filename)
            # filename = os.path.join(dest_folder, 'residual_from_%d_to_%d.npy' % (start_frame, end_frame))
            # np.save(filename, global_residuals)
            # print('Saved residuals to', filename)

    if mpi_rank == 0:        
        print('Goodbye')


if __name__ == "__main__":
    main_function()
