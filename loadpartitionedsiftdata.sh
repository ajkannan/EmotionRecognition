#!/bin/bash
# The name of the job, can be anything, simply used when displaying the list of running jobs
#$ -N loadsiftdata
# Combining output/error messages into one file
#$ -j y
# Set memory request:
#$ -l vf=10G
# Set walltime request:
#$ -l h_rt=10:00:00
# ironfs
#$ -l ironfs
# One needs to tell the queue system to use the current directory as the working directory
# Or else the script may fail as it will execute in your top level home directory /home/username
#$ -cwd
# then you tell it retain all environment variables (as the default is to scrub your environment)
#$ -V
# Now comes the command to be executed

export LM_LICENSE_FILE=27000@anthill.cs.dartmouth.edu:1711@rclserv1.dartmouth.edu
matlab2012b -nodesktop -nosplash -r loadpartitionedsiftdata

exit 0
