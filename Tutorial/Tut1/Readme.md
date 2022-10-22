# Tutorial 1: Calculating Nanoparticle Properties

Please read Tut1.py

## Summary of commands

### Initial steps that were peformed before the tutorial
Load gromacs, python (Dependent on your system conigurations).
```bash
module load gromacs/5.1.4
module load python/3.7
source ~/ENV/bin/activate
```

Write index file to remove water. In this case we are keeping DNA, PEI and ION. You can also remove ION.
```bash 
gmx make_ndx -f big_fle.gro -o index.ndx << EOF
1|2|8
q
EOF
```
Extract the smaller system. The name of the group will be different in your systems.

```bash
echo "DNA_PEI_ION" | gmx trjconv -f big_file.xtc -s big_file.tpr -n index.ndx -o md_1.xtc
``` 

Convert the tpr.

```bash
echo "DNA_PEI_ION" | gmx convert-tpr -s big_file.tpr -n index.ndx -o md_1.tpr
``` 

The files `md_1.tpr` and `md_1.xtc` are provided in the tutorial.

### Starting the tutorial

Extract trajectories while making molecules whole.

```bash
mkdir -p Whole
echo "0" | gmx trjconv -f md_1.xtc -s md_1.tpr -pbc whole -o Whole/DP.gro -sep
gmx dump -s md_1.tpr > tpr. dump
```

Create a `constants.dat` file. This has been created for the tutorial.

run the python script.
```bash
python calc.py
``` 
