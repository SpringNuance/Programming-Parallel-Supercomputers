# Exercise 6 - Distributed Computing on GPUs

In this exercise, you implemented functions to find the maximum integer in an array distributed across multiple GPUs (devices). Your submissions have been graded using an autograder that is included in the model solutions. This script can be run as follows.

'''
 python sheet6/grade.py --submissions /scratch/courses/programming_parallel_supercomputers_2023/<your username>/<your path to the zip-file>/<your studentnumber>.zip --model-directory /home/manterm1/PPS/pps-example-codes/exercise_sheets/sheet6 --exercise-files reduce-single.cu reduce-multi.cu reduce-mpi.cu --job-scripts run-single.sh run-multi.sh run-mpi.sh
 '''

Half of the points for each task have come from the autograder output, and the rest from a cursory check whether a solution has been attempted, and whether it contains the correct elements. Further debugging has not been possible by the course staff, but we are currently negotiating whether you could continue testing your solutions against the model solution in Triton (if they did not work properly).

