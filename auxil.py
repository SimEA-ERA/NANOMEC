from __future__ import print_function
from __future__ import division
import math
import numpy as np
from numba import njit
import sys


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    sys.stderr.flush()

def calc_box_params(conf):
    box_bounds = conf['box_bounds']
    xlo_bound = box_bounds[0,0]
    xhi_bound = box_bounds[0,1]
    xy = box_bounds[0,2]
    ylo_bound = box_bounds[1,0]
    yhi_bound = box_bounds[1,1]
    xz = box_bounds[1,2]
    zlo_bound = box_bounds[2,0]
    zhi_bound = box_bounds[2,1]
    yz = box_bounds[2,2]

    xlo = xlo_bound - min(0.0, xy, xz, xy+xz)
    xhi = xhi_bound - max(0.0, xy, xz, xy+xz)
    ylo = ylo_bound - min(0.0, yz)
    yhi = yhi_bound - max(0.0, yz)
    zlo = zlo_bound
    zhi = zhi_bound

    result = np.zeros((3,3))
    result[0,0] = xlo
    result[0,1] = xhi
    result[0,2] = xy
    result[1,0] = ylo
    result[1,1] = yhi
    result[1,2] = xz
    result[2,0] = zlo
    result[2,1] = zhi
    result[2,2] = yz

    return result

def calc_box_size(conf):
    params = calc_box_params(conf)
    lx = params[0,1] - params[0,0]
    ly = params[1,1] - params[1,0]
    lz = params[2,1] - params[2,0]
    return np.array([lx,ly,lz])  

@njit()
def py2round(x):
    if x >= 0.0:
        return math.floor(x + 0.5)
    else:
        return math.ceil(x - 0.5)

@njit("float64(float64,float64,float64)")
def periodicDistance(x1, x2, length):
    delta = x1 - x2
    return delta - py2round(delta/length)*length
    
@njit()
def periodicDistance3(v1, v2, box_size):
    dx = periodicDistance(v1[0], v2[0], box_size[0])
    dy = periodicDistance(v1[1], v2[1], box_size[1])
    dz = periodicDistance(v1[2], v2[2], box_size[2])
    return math.sqrt(dx*dx + dy*dy + dz*dz)

@njit()
def periodicDistanceVec(v1, v2, box_size):
    dx = periodicDistance(v1[0], v2[0], box_size[0])
    dy = periodicDistance(v1[1], v2[1], box_size[1])
    dz = periodicDistance(v1[2], v2[2], box_size[2])
    return np.array([dx,dy,dz])           

def time_length_str(delta):
    hour = int(delta / 3600)
    delta -= hour * 3600
    minute = int(delta / 60)
    delta -= minute * 60
    seconds = delta
    return '%02d' % hour + ':%02d' % minute + ':%02d' % seconds     

def seek_to_trajectory_step(lammps_reader, target_step):
    print('Seeking to trajectory step', target_step)
    while True:
        conf = lammps_reader.readNextStep()
        if conf is None:
            return None
        step_no = conf['step_no']
        print('Read conf', step_no)
        if step_no == target_step:
            return conf
        if step_no > target_step:
            return None   