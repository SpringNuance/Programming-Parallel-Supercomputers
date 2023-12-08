
Further instructions on code usage
==================================

The compilation and job submission scripts for MPI+OpenMP codes are
provided in scripts/job_hybrid_example.sh.

The number of MPI ranks has to be a factor of the grid dimension (default 
dimension is 2000). The default initial temperature field is a disk. Initial
temperature field can be read also from a file, the provided **bottle.dat** 
illustrates what happens to a cold soda bottle in sauna.


 * If the file `HEAT_RESTART.dat` exists, it will be read and produce
   the initial field and remember the last iteration step.  No other
   options will be used. (To run a restart with a certain number of
   iterations, use:
     `srun ./heat_mpi - N_ITERATIONS`, with `-` as input filename.)
 * Running with defaults: `srun <options in your batch file>./heat_mpi`
 * Bottle in sauna: `srun <options> ./heat_mpi bottle.dat`
 * Bottle in sauna, given number of time steps:
   `srun <options> ./heat_mpi bottle.dat 1000`
 * Default pattern with given dimensions and time steps:
   `srun <options> ./heat_mpi 800 800 1000`

  The program produces a series of `heat_XXXX.png` files which show the
  time development of the temperature field.

You can visualize the png files with any image viewer, the `display`
command line program on Triton (if you have graphics forwarding set
up), or from Python using the following code:
```console
$ module load anaconda
$ pip3 install matplotlib
```
```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread('heat_1000.png')
imgplot = plt.imshow(img)
plt.show()
```

