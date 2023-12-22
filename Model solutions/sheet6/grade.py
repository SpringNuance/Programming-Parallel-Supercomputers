#!/bin/env python3
import os
import argparse
import zipfile
from pathlib import Path
import subprocess
import time

grades='grades.csv'

parser = argparse.ArgumentParser(description='A tool for performing automatic grading')
parser.add_argument('--submissions', type=str, nargs='+', help='A list of zip files containing the submissions to the exercises. Should be in format studentnumber.zip.')
parser.add_argument('--exercise-files', type=str, nargs='+', help='A list of files that each submission should contain. These files are copied over the model solution.')
parser.add_argument('--model-directory', required=True, type=str, help='A directory that contains the model solution')
parser.add_argument('--job-scripts', type=str, nargs='+', help='A list of job scripts used to test a submission ')
parser.add_argument('--dryrun', action='store_true', help='Do a dryrun without compiling or running.')
args = parser.parse_args()


print(f'Got {args.submissions}')
print(f'Got {args.exercise_files}')
print(f'And {args.model_directory}')
print(f'And {args.job_scripts}')

# Load modules
os.system('module load gcc cuda cmake openmpi')

# Copy the model solution and clean up its exercise files
os.system('rm -rf model && mkdir -p model')
os.system(f'cp -r {args.model_directory}/* model/')
#for file in args.exercise_files:
#    os.system(f'rm model/src/{file}')

# Output .csv labeling
with open(grades, "a") as fp:
    print('studentnumber,score,notes', file=fp)

for submission in args.submissions:
    student_number = Path(submission).stem
    print(f'Student number {student_number}')

    # Copy the model solution
    print(f'Copying model files to {student_number}')
    os.system(f'mkdir -p {student_number}/src/')
    os.system(f'cp -r model/* {student_number}')

    # Check the zip file and copy the student solution to src/{file}
    try:
        with zipfile.ZipFile(submission, 'r') as zip_ref:
            nl = zip_ref.namelist()
            for file in args.exercise_files:
                namefound=False
                for name in nl:
#                    print('zip-name:',name, ', exercise_file: ',file)
                    if file in name:
#                        os.system(f'rm model/{file}')
                        zip_ref.extract(name, path=student_number)
                        if name!='src/'+file:
                            os.system(f'cp -p {student_number}/{name} {student_number}/src/{file}')
                        print(f'{student_number}: Replaced model solution for {file} by studet suvmission {name}.')
                        namefound = True
                    continue
                if namefound==False:
                    print(f'{student_number}: {file} not found in student submission.')
                    with open(grades, "a") as fp:
                        print(f'{student_number},0,"{file} not contained in zip-file"', file=fp)
    except:
        os.system(f'rm -rf {student_number}/build/*')
        with open(grades, "a") as fp:
            print(f'{student_number},0,"Invalid zip"', file=fp)
        continue

    # Compile
    # Note that must load the proper modules before calling this script
    # `module load gcc cuda cmake openmpi`
    if args.dryrun:
        print(f'mkdir -p {student_number}/build')
        print(f'module load gcc cuda cmake openmpi && rm -rf {student_number}/build/* && cmake -S {student_number} -B {student_number}/build')
        print(f'make --directory="{student_number}/build" -j')
    else:
        os.system(f'mkdir -p {student_number}/build')
        os.system(f'module load gcc cuda cmake openmpi && rm -rf {student_number}/build/* && cmake -S {student_number} -B {student_number}/build')
        os.system(f'make --directory="{student_number}/build" -j')

    # Queue a batch job
    for script in args.job_scripts:
        if args.dryrun:
            print(f'sbatch --chdir="{student_number}/build" {student_number}/job-scripts/{script}')
        else:
            os.system(f'sbatch --chdir="{student_number}/build" {student_number}/job-scripts/{script}')

        njoblimit = 2
        njobs = int(subprocess.check_output('squeue --me | wc -l', shell=True)) - 1
        while njobs >= njoblimit:
            os.system('squeue --me')
            time.sleep(2)
            njobs = int(subprocess.check_output('squeue --me | wc -l', shell=True))
        print('Jobs completed')


njobs = int(subprocess.check_output('squeue --me | wc -l', shell=True)) - 1
while njobs > 0:
    os.system('squeue --me')
    time.sleep(2)
    njobs = int(subprocess.check_output('squeue --me | grep -v JOBID | wc -l', shell=True))
print('Jobs completed')

for submission in args.submissions:
    student_number = Path(submission).stem
    if not os.path.isdir(student_number):
        continue

    score = int(subprocess.check_output(f'cat {student_number}/build/*.result | grep OK | wc -l', shell=True))
    #assert(score <= len(args.job_scripts))

    notes = ''
    nresults = int(subprocess.check_output(f'cat {student_number}/build/*.result | wc -l', shell=True))
    if nresults:
        notes = subprocess.check_output(f'grep "" -H {student_number}/build/*.result', shell=True)

    with open(grades, "a") as fp:
        print(f'{student_number},{score},"{notes}"', file=fp)

    
