# Exercise 2 - Taxonomies and Definitions

The learning goal of this exercise is to understand **the most important
definitions** that you need when programming parallel supercomputers. We will
use these extensively in the coding exercises, so it is very important to
master these concepts.

We undertake this exercise during the second lecture session in the
format of a gallery walk; everyone participating will get full points.
If you cannnot make it, you can also write a learning diary and return
it to
[MyCourses](https://mycourses.aalto.fi/mod/assign/view.php?id=1091757). The
diary will be graded by the course personnel. The posters produced
during the second exercise session will be the distributed as the
correct solution to this exercise sheet.

If you decide to submit this exercise as a learning diary: the
excercises have equal weight in the grading.

Your tasks
==========


### 1. Inspect the list of [top 500 computers in theworld](https://www.top500.org/).
What kind of trends in terms of Flynn’s taxonomy can be seen, if any? What does that mean for the actual programming concepts?

### 2. Inspect the Mahti interconnect network either from CSC docs or lecture slides (slide 19).

Yeah, you can also find a short description of it from CSC docs. But, analyze it using the terms in the lecture material. 

### 3. Draw a schematic plot of weak and strong scaling based on Amdahl and Gustafson law's, respectively. 

### 4. We have a 2D iterative stencil loop problem to program, see lecture slide 37.

Now we need to use a 4th order Moore stencil instead of the von Neumann one. If your 2D subdomain would have the dimension of 16 points, what would be the ratio of grid points that you can compute and communicate at the same time? Analyse the compute/communication performance according to the ACC model (slide 28).

### 5. Synthesis: why do you think these concepts are important to master?