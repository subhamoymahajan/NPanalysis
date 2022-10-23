# Tutorial 2: Calculating Nanoparticle Properties

Please read Tut2.pdf

## Command summary

### Load gromacs and python

```bash
module load gromacs/5.1.4
module load python/3.7
source ~/ENV/bin/activate
```

### Run commands for tutorial
Extract `NPwhole.xtc`
```bash
mkdir -p NPwhole
echo "0" | gmx trjconv -f NPwhole.xtc -s md_1.tpr -o NPwhole/DP.gro -sep 
```

Run the python script
```bash
python calc.py
```
