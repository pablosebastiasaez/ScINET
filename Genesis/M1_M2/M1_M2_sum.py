from mpi4py import MPI
import numpy as np
import sys
from scatter_cython import sum_integral

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank==0: 
    print('rank: ',rank)
    print('size: ',size)
    print('initializing')

    nx=251; ny=251; nz=251
    Lx = 25000.; Ly = 25000.; Lz = 2000.
    x = (2 * (np.arange(nx) + 1) - 1 - nx) * np.pi / Lx
    y = (2 * (np.arange(ny) + 1) - 1 - ny) * np.pi / Ly
    z = (2 * (np.arange(nz) + 1) - 1 - nz) * np.pi / Lz
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    dz = z[1]-z[0]

    eps_h = dx/2
    eps_v = dz/2
    dV=dx*dy*dz
    xmax = x[-1]
    ymax = y[-1]
    zmax = z[-1]
    ZZ0,XX0 = np.meshgrid(z[int((nz+1)/2 -1):],x[int((nx+1)/2 -1):], indexing='ij')
    ZZ0 = ZZ0.flatten()
    XX0 = XX0.flatten()

    ZZ,YY,XX = np.meshgrid(z,y,x, indexing='ij')
    Z,X,Y = ZZ.flatten(), XX.flatten(), YY.flatten()
    cond = (X**2+Y**2+Z**2 > 0) & (X**2+Y**2 < xmax**2+1e-12 )
    X = X[cond]
    Y = Y[cond]
    Z = Z[cond]
    del YY, ZZ, XX

    f0 = 1e-4; N0 = 5e-3
    dt=10*24*60*60
else:
    xmax=None
    ymax=None
    zmax=None
    XX0=None
    ZZ0=None
    Z=None
    X=None
    Y=None
    dV=None
    dt=None
    f0=None
    N0=None
    eps_h=None
    eps_v=None

# Broadcast the global variable to all processes
zmax = comm.bcast(zmax, root=0)
xmax = comm.bcast(xmax, root=0)
ymax = comm.bcast(ymax, root=0)
XX0 = comm.bcast(XX0, root=0)
ZZ0 = comm.bcast(ZZ0, root=0)
Z = comm.bcast(Z, root=0)
X = comm.bcast(X, root=0)
Y = comm.bcast(Y, root=0)
dV = comm.bcast(dV, root=0)
dt = comm.bcast(dt, root=0)
f0 = comm.bcast(f0, root=0)
N0 = comm.bcast(N0, root=0)
eps_h = comm.bcast(eps_h, root=0)
eps_v = comm.bcast(eps_v, root=0)
it=1

#N_len=int(len(x)*len(y)*len(z))
N = XX0.shape[0]
i_blk = (N-1)//(size) + 1        # extent of each block
my_blk = np.mod(rank,(size))+1  # index of this block
is_pe =  (N*(it-1)) + (my_blk-1)*i_blk        # start index of this block
ie_pe =   (N*(it-1)) +  my_blk*i_blk   
if rank==0: 
    print('total points :',N)
    print('total triads :',N*N)
    print('points per cpu :',i_blk)
    print('triads per cpu :',i_blk*N)
    print('starting loop')
    sys.stdout.flush()

X0 = np.zeros(i_blk, 'f')
Y0 = X0.copy()
Z0 = X0.copy()
wave_sum = X0.copy()
wave_diff = X0.copy()
wave_sum_f_re = X0.copy()
wave_sum_f_nr = X0.copy()
wave_sum_N_re = X0.copy()
wave_sum_N_nr = X0.copy()
wave_sum_rest_re = X0.copy()
wave_sum_rest_nr = X0.copy()
wave_sum_mix_re = X0.copy()
wave_sum_mix_nr = X0.copy()

for i in range(is_pe,min(ie_pe,N*it)):
    if rank==0 and (i-is_pe) % int(10) == 0:
        print(int((((i-is_pe))/int(i_blk))*100), '%')
        sys.stdout.flush()
    X0[i-is_pe],Y0[i-is_pe],Z0[i-is_pe],wave_sum[i-is_pe],wave_sum_f_re[i-is_pe],wave_sum_f_nr[i-is_pe],wave_sum_N_re[i-is_pe],wave_sum_N_nr[i-is_pe], wave_sum_mix_re[i-is_pe], wave_sum_mix_nr[i-is_pe],wave_sum_rest_re[i-is_pe],wave_sum_rest_nr[i-is_pe] =  sum_integral(XX0[i],0.0,ZZ0[i],X,Y,Z,xmax,ymax,zmax,f0,N0,dt,dV,eps_h,eps_v)
# Gather results from all processes
comm.barrier()  # Ensure all processes finish saving before moving on
comm.barrier()  # Adding a second barrier to be extra sure

name=f'scatter_{rank}_sum_{it}.npz'
np.savez(name,
        kx=X0,
        ky=Y0,
        kz=Z0,
        ww_sum=wave_sum,
        ww_sum_mix_re=wave_sum_mix_re,
        ww_sum_mix_nr=wave_sum_mix_nr,
        ww_sum_f_re=wave_sum_f_re,
        ww_sum_f_nr=wave_sum_f_nr,
        ww_sum_N_re=wave_sum_N_re,
        ww_sum_N_nr=wave_sum_N_nr,
        ww_sum_rest_re=wave_sum_rest_re,
        ww_sum_rest_nr=wave_sum_rest_nr)
print(f'Rank {rank}: After saving')
sys.stdout.flush()

