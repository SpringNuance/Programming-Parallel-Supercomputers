# Exercise 2 - Taxonomies and Definitions

The learning goal of this exercise is to understand **the most important
definitions** that you need when programming parallel supercomputers. We will
use these extensively in the coding exercises, so it is very important to
master these concepts.

We undertake this exercise during the second lecture session in the
format of group work; everyone participating will get full points.
If you cannnot make it, you can also write a learning diary and return
it to
[MyCourses](https://mycourses.aalto.fi/mod/assign/view.php?id=1091757). The
diary will be graded by the course personnel. The group work outcomes
will be the distributed as the
correct solution to this sheet.

If you decide to submit this exercise as a learning diary: the
tasks have equal weight in the grading.

Your tasks
==========


### 1. Inspect the list of [top 500 computers in the world](https://www.top500.org/)

Hints: use the same method (inspect evolution over time) as in sheet 1 and at least the same keywords. You are encouraged to inspect further keywords.

Guiding questions:

1. What kind of trends in terms of Flynn’s taxonomy (SISD, MISD, SIMD, MIMD) can be seen, if any? 

2. What does that mean for the actual programming concepts?

Very good answers on the posters!

-------

### 2. Inspect the Mahti interconnect network either from CSC docs or lecture slides (slide 19).

Hint: you can also find a short description of it from CSC docs. 

Guiding questions: Analyze it as per 

1. Topology

2. Connection type

3. Bandwidth 

Answers on the posters are quite good. What I was looking after in addition to the posters is "heavily multilayered".

---------

### 3. You have a code of which 80% can be efficiently parallelized.

Hint: Remind yourself of Amdahl's (p. 30 and 32 of lecture slides) and Gustafsson's laws (p. 33-34).

1. Draw a schematic strong scaling plot, with ideal scaling and the expected one: T (serial)/T (p processors) as function of p.

2. Draw a schematic weak scaling plot, when the proportion of work is kept fixed per processor count p (ideal and expected). 

Answers:

1. Drawings in nearly all posters are correct.

2. Only one group got this right: the speed of the computation should remain constant, hence we are expecting to see a flat line. I hope you could convince yourselves with sheet5 scaling tests!

-------

### 4. We have a 2D iterative stencil loop problem to program, see lecture slide 37.

Now we need to use a 4th order Moore stencil instead of the von Neumann one. If your 2D subdomain would have the dimension of 16 points, what would be the ratio of grid points that you can compute and communicate at the same time? Analyse the compute/communication performance according to the ACC model (slide 28).

Here one group got it right: the ratio is one.

-------

### 5. Synthesis: why do you think these concepts are important to master?

Programming of supercomputers is nowadays concerned with programming heterogeneous architectures. You need to mix shared mem and distributed mem programming models, MIMD, SIMD, and even SISD type of parallelism, and in addition you need to worry about concurrency of computations and communications over the interconnect network.
