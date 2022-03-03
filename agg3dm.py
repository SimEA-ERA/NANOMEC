from __future__ import print_function
from __future__ import division
import numpy as np

class Aggregator3DM:
    def __init__(self, xlo, xhi, ylo, yhi, zlo, zhi, aggsize, float_shape):
        if len(float_shape) != 2:
            raise Exception('bad shape')
        self.float_shape = float_shape
        self.startx = xlo
        self.starty = ylo
        self.startz = zlo
        self.dx = aggsize
        self.dy = aggsize
        self.dz = aggsize
        self.lx = xhi - xlo
        self.ly = yhi - ylo
        self.lz = zhi - zlo
        self.nx = int(self.lx//self.dx)
        if self.lx % self.dx > 0: self.nx += 1
        self.ny = int(self.ly//self.dy)
        if self.ly % self.dy > 0: self.ny += 1
        self.nz = int(self.lz//self.dz)
        if self.lz % self.dz > 0: self.nz += 1
        self.lx = self.nx*self.dx
        self.ly = self.ny*self.dy
        self.lz = self.nz*self.dz
        self.float_sums = np.zeros((self.nx, self.ny, self.nz, float_shape[0], float_shape[1]), np.float64)
        self.counts = np.zeros((self.nx, self.ny, self.nz), np.uint64)
        self.total_count = 0

    def add(self,dr,float_value):
        ix = int((dr[0]-self.startx)//self.dx)
        iy = int((dr[1]-self.starty)//self.dy)
        iz = int((dr[2]-self.startz)//self.dz)
        if ix >= 0 and ix < self.nx and iy >= 0 and iy < self.ny and iz >= 0 and iz < self.nz:
            self.float_sums[ix,iy,iz,:,:] += float_value
            self.counts[ix,iy,iz] += 1
            self.total_count += 1
        # else:
        #     print(dr)
        #     print(self.startx, self.starty, self.startz)
        #     print(self.dx, self.dy, self.dz)
        #     print(ix,iy,iz)
        #     #quit()

    def calc_averages_by_count(self):
        result = np.zeros((self.nx, self.ny, self.nz, self.float_shape[0], self.float_shape[1]))
        for ix in range(self.nx):
            for iy in range(self.ny):
                for iz in range(self.nz):
                    if self.counts[ix,iy,iz] > 0:
                        # sprint('--->', self.float_sums[ix,iy,iz,:,:] / self.counts[ix,iy,iz])
                        result[ix,iy,iz,:,:] = self.float_sums[ix,iy,iz,:,:] / self.counts[ix,iy,iz]
                    else:
                        result[ix,iy,iz,:,:] = np.NaN
        return result

    def show_axis(self):
        self.means = self.values.copy()
        for ix in range(self.means.shape[0]):
            for iy in range(self.means.shape[1]):
                for iz in range(self.means.shape[2]):
                        self.means[ix,iy,iz,:,:] = None

        for ix in range(self.means.shape[0]):
            for iz in range(self.means.shape[2]):
                self.means[ix,0,iz] = 1

        for iy in range(self.means.shape[1]):
            for iz in range(self.means.shape[2]):
                self.means[0,iy,iz] = 2
