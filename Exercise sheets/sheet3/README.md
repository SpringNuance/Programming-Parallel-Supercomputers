# Exercise 3 - Point-to-point communication

If the equations don't render in your viewer, you can view them at
least on Gitlab: https://version.aalto.fi/gitlab/manterm1/pps-example-codes/-/blob/main/exercise_sheets/sheet3/sheet.md

## Communication-time measurement 

Implement a simple point-to-point communication involving just two MPI processes and measure the communication time as a function of data package size for three different combinations out of the spectrum of the MPI send/recv routines.
For time measurement, use MPI_Wtime. (Consider, that the OpenMPI library doesn’t provide synchronization of the starting time across processes. So, you have to organize a proper synchronisation yourself.)

In particular, compare the case "both processes on the same node” with “processes on different nodes”.
The number of processes per node can be controlled by the SLURM option `--ntasks-per-node`.

Execution on a cluster is affected by random effects, hence the results will in general be subject to scatter.
Hence, for obtaining reliable mean values, you have to repeat the measurement at an appropriate rate.

Document your work in a separate PDF document containing
- a short description of the code,
- a graphical representation of the measured latency/bandwidth - package size relation,
- a short discussion of the results.

Bonus task: Compare communication performance (Bytes per second communicated) with compute
performance (FLOPS). FLOPS can be estimated by running and timing a long loop, e.g. adding two large arrays.

## A physical application case

Typical application cases of stencil computations are partial differential equations. A physical application case of passive scalar transport is described in detail in
the document `phys_appl.md`. Your task is to complete the code template `advec_wave_2D_skel.c` to obtain a working 2D-solver for that problem using **one-sided** MPI communication.
This requires
- defining a mapping of the $N$ MPI processes (ranks) to the $N$ equally-sized subdomains, into which the computational domain is decomposed
- figuring out the neighboring relationships of the MPI processes and implementing corresponding functions (see code template)
- establishing MPI windows 
- choosing a scheme of non-blocking communication
- defining a convenient data type for MPI_Get or MPI_Put
- splitting the evaluation of the right-hand-side of the PDE to allow maximum concurrency with the communication
- verifying the obtained solution against the analytical one.

Establish the level of concurrency by timing runs with communication and time integration against runs with communication only and with time integration only (which would be physically incorrect, of course).
Write a short report, accounting on your design decisions and the results (solution verification and concurrency).

