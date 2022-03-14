import math
import numpy as np
from auxil import periodicDistance, periodicDistance3
from numba import int32, int64, float64, jitclass, types, typed
from numba.experimental import jitclass

int_list = types.ListType(int64)

spec = [
    ('lx', float64),
    ('ly', float64),
    ('lz', float64),
    ('rcut', float64),
    ('nx', int64),
    ('ny', int64),
    ('nz', int64),
    ('nyz', int64),
    ('nxyz', int64),
    ('clx', float64),
    ('cly', float64),
    ('clz', float64),
    ('dnx', int64),
    ('dny', int64),
    ('dnz', int64),
    ('cellAtoms', types.ListType(int_list)),
    ('cellNeis', types.ListType(int_list)),
]

@jitclass(spec)
class NeiCellList:

    def __init__(self):
        self.makeCells(10.0,10.0,10.0, 6.0)#

    def makeCells(self, lx,ly,lz, rcut):
        if rcut > lx or rcut > ly or rcut > lz:
            raise Exception('rcut too large')

        self.lx = lx*1.0
        self.ly = ly*1.0
        self.lz = lz*1.0

        self.rcut = rcut*1.0

        self.nx = int(math.floor(lx / (rcut/2.0)))
        self.ny = int(math.floor(ly / (rcut/2.0)))
        self.nz = int(math.floor(lz / (rcut/2.0)))
        self.nyz = self.ny * self.nz
        self.nxyz = self.nx*self.ny*self.nz

        self.clx = self.lx / self.nx
        self.cly = self.ly / self.ny
        self.clz = self.lz / self.nz

        self.dnx = int(math.ceil(self.rcut / self.clx))
        self.dny = int(math.ceil(self.rcut / self.cly))
        self.dnz = int(math.ceil(self.rcut / self.clz))

        self.cellAtoms = typed.List.empty_list(int_list)
        for i in range(self.nx*self.ny*self.nz):
            self.cellAtoms.append(typed.List.empty_list(int64))
        self.cellNeis = typed.List.empty_list(int_list)
        for i in range(self.nx*self.ny*self.nz):
            self.cellNeis.append(typed.List.empty_list(int64))

        for ix in range(self.nx):
            for iy in range(self.ny):
                for iz in range(self.nz):
                    xmin = ix - self.dnx
                    xmax = ix + self.dnx
                    ymin = iy - self.dny
                    ymax = iy + self.dny
                    zmin = iz - self.dnz
                    zmax = iz + self.dnz

                    neis = set()

                    for x in range(xmin,xmax+1):
                        if x < 0: x += self.nx
                        if x >= self.nx: x -= self.nx
                        for y in range(ymin,ymax+1):
                            if y < 0: y += self.ny
                            if y >= self.ny: y -= self.ny
                            for z in range(zmin,zmax+1):
                                if z < 0: z += self.nz
                                if z >= self.nz: z -= self.nz

                                index = self.calcCellIndex(x,y,z)
                                neis.add(index)

                    index = self.calcCellIndex(ix,iy,iz)
                    l = self.cellNeis[index]
                    for n in neis:
                        l.append(n)

    def calcCellIndex(self, x,y,z):
        return x*self.nyz + y*self.nz + z

    def calcCellCoords(self, i):
        x = int(i) // self.nyz
        y = (int(i) - x * self.nyz) // self.nz
        z = int(i) - x * self.nyz - y * self.nz
        return (x,y,z) 

    def calcCellIndexOfAtom(self, x,y,z):
        x = x - math.floor(x / self.lx) * self.lx
        y = y - math.floor(y / self.ly) * self.ly
        z = z - math.floor(z / self.lz) * self.lz

        cellx = int(x // self.clx)
        celly = int(y // self.cly)
        cellz = int(z // self.clz)

        if cellx < 0 or celly < 0 or cellz < 0 or cellx >= self.nx or celly >= self.ny or cellz >= self.nz:
            raise Exception('oops')

        return self.calcCellIndex(cellx, celly, cellz)

    def makeList(self, xs, ys, zs):
        self.cellAtoms = typed.List.empty_list(int_list)
        for i in range(self.nx*self.ny*self.nz):
            self.cellAtoms.append(typed.List.empty_list(int64))        

        for i in range(len(xs)):
            x = xs[i]
            y = ys[i]
            z = zs[i]
            self.cellAtoms[self.calcCellIndexOfAtom(x,y,z)].append(i)

    def calcNeisOf(self, atomIndex, xs, ys, zs, rcut):
        if rcut > self.rcut:
            raise Exception("rcut > list rcut")
        neis = []
        x, y, z = xs[atomIndex], ys[atomIndex], zs[atomIndex]
        rootCellIndex = self.calcCellIndexOfAtom(x,y,z) 
        neiCellsIndices = self.cellNeis[rootCellIndex]
        for neiCellIndex in neiCellsIndices:
            neiAtomIndexList = self.cellAtoms[neiCellIndex]
            for neiAtomIndex in neiAtomIndexList:
                if neiAtomIndex != atomIndex:
                    neix, neiy, neiz = xs[neiAtomIndex], ys[neiAtomIndex], zs[neiAtomIndex]
                    dx = periodicDistance(neix,x,self.lx)
                    dy = periodicDistance(neiy,y,self.ly)
                    dz = periodicDistance(neiz,z,self.lz)
                    if math.sqrt(dx*dx + dy*dy + dz*dz) <= rcut:
                        neis.append(neiAtomIndex)
        return neis


def check_same_neis(list_a, list_b, at, xu, yu, zu, boxSize):
    set_a = set(list_a)
    set_b = set(list_b)
    posi = np.array([xu[at], yu[at], zu[at]])
    for a in set_a:
        if not a in set_b:
            posj = np.array([xu[a], yu[a], zu[a]]) 
            dist = periodicDistance3(posi, posj, boxSize)
            print('--->', dist)
            # raise Exception('missing nei')
        else:
            set_b.remove(a)
    if len(set_b) > 0:
        for b in set_b:
            posj = np.array([xu[b], yu[b], zu[b]]) 
            dist = periodicDistance3(posi, posj, boxSize)
            print('--->', dist)
            # raise Exception(str(len(set_b)) + ' extra neis')    