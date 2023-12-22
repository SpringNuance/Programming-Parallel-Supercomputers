# Exercise 1 - HPC landscape

The first thing that we have to establish is **what are we
going to program** computer-architecture-wise during this course. For this learning outcome, we
need to map out the HPC landscape.


The posters produced during the first lecture session are distributed as the
correct solution to this exercise sheet.

Your tasks
==========

Inspect the list of [top 500 computers in the
world](https://www.top500.org/).

### 1. What kind of trends between HPC computing paradigms can be seen? 

Hints: Go to "Statistics" -> "Development over time", and build graphs
using variable criteria for "Performance share". You can extend the
time bar with the slider on top of the graph, and select/de-select
data by clicking on the squares below the graph. Use the widest
possible time range.

Guiding questions:

1. What can you observe if you use "Architecture" as a
search keyword? Hints: SMP refers to shared memory multiprocessors and MPP to
massively parallel processors without shared memory, constellations to grid
computing, and clusters to architectures that use both SMP and MPP concepts.

2. What can you observe if you use "Accelerator/CP family" as a
keyword?

3. What can you observe if you use "Interconnect family" as a keyword?

4. What can you observe if you use "Cores per socket" as a keyword?

The posters collect all essential answers to these questions.

-----

### 2. How is the “power wall” phenomenon seen in the list?

Hints: see the hints and guiding questions in task 1, and go through the
steps to interpret your observations in the light of the power wall
phenomenon.

You are also free to use other keywords, if you think they would be
more indicative to show the power wall phenomenon.

The posters have very good answers to this question.

------

Inspect the [technical specifications of the Triton cluster](https://scicomp.aalto.fi/triton/overview/)

### 3. What kind of processor(s), accelerators, and interconnect does it have?

Hints: from the Triton specs you see that it has many generations of
computing cores, spanning over roughly 5 years or so. This gives us
some opportunity to inspect how fast important components (clock
frequency, co-proc. capabilities, core count, memory, memory
bandwidth, interconnect speed) improve over time. In the memory
bandwidth specs (DDRx-yyyy) the important number is yyyy, which tells
you the number of transactions [unit of million] per second. Multiply
by 8 to get the bandwidth in MB/s. Clocking frequenciess you need to search
from internet by architecture.

Guiding questions:

1. How has the clock frequency of CPUs changed over time in the hardware options provided by Triton?

2. How has the memory bandwidth changed over time?

3. How has the number of co-proc. capabilities changed over time?

4. How has the ratio of compute capabilities over memory transactions changed?

The essential observation here is that the compute capabilities continue increasing, more or less following the Moore's first law, but now thanks to the cores per socket and GPUs instead of the clocking frequency of the CPUs increasing. At the same time, interconnect and memory bandwidths ARE NOT increasing with the same rate. 


------

Choose any of the mobile devices that you carry. Find out its technical
specifications from the internet.

### 4. What kind of processor(s), accelerators, …, does your mobile device have, how many cores and threads they run, what memory type and what caches they use.

Hints for the group work: please collect specs of different devices, and compare them to the Triton specs.

Hints for the learning diary: you can select only one device, and compare it to the Triton specs.

Guiding question: per each new device, discuss what are the
differences to a supercomputer with respect to processor clock
frequency, memory, anything relevant you can think of, and write down
your observations.

The posters include an interesting collection of mobile device specs. There are naturally differences in the scale and energy consumption between supercomputers and mobile devices, but as per the basic building blocks (CPU capabilities, GPUs, memory, bandwidth, ...) mobile devices are not so different from those wherefrom supercomputers are built from. 

-------

### 5. Synthesis: What are we going to program during this course?

Guiding questions: based on the material you went through in this exercise

1. what are the programming paradigms we need to use in making efficient parallel code in Triton?

2. what are the main bottlenecks that we need to design our codes to cope with?

Successful applications to be run on supercomputers require algorithms that optimize both for shared and distributed memory usage. The interconnect and memory bandwidths are not increasing as rapidly as the compute capacity, hence the bottleneck will be in the former.