#!/bin/bash

module load gromacs/5.1.4
gmx dump -s md_1.tpr > test.dat 2> log.log # Dump tpr file to test.dat in readable format
grep ' m= ' test.dat > test1.dat  # Extract lines containing 'm='
sed 's/^.*m=//g' test1.dat | sed 's/,.*$//g' > mass.dat # Remove everything before and including 'm=' and remove everyting after ',' (after m=)
#This essentially leaves only the string after 'm=' and before the next comma.

