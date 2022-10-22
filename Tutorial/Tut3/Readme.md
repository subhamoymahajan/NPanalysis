## Tutorial 3: Generating Principal Transition Diagram

files: connected.pickle, time.pickle have been calculated from first 150 ns of 
 acidification simulation of α = 10 and instant acidificaiton (Figure 8a in [1])

### Calculating Principal Transition Diagram 
```bash
python run.py
```

`gen_GPT()` calculates GPT and  `plot_GPT()` shows the GPT (Graph of prinicipal transition; Same as principal transition diagram in [1]).
 The plot has to be zoomed in different locations to clearly see the time for each transition. 
The tranition time is in microseconds (same as time.pickle).  


A simple representation of GPT is,

0->7->9->13->39->24->28->(-)->20,36,42

1->8->(-)->30,31

`->` represents principal transitions and `(-)` represents dissociation of nanoparticle. The numbers are unique IDs of a nanoparticle. 

For aggregation and restructuring, respectively `(+)` and `(x)` would be used 

### Calculating & Plotting Individual NP structure Graph (NSG).
```bash
python run2.py
```
First the code will calculate weighted NSGs. 

Second, we determine some visually appealing representations for NSGs. 
We only have to do this for  GPT starting nodes (0, 1). We first determine the representation for GPT node 0 (or U0). You will see a plot with randomly initialized representation of U0. You can accept this represtation by answering `y` followed by `save`. However, we will try to generate same represntation as [1] in the following steps.


First answer no.
```bash
Keep existing positions (y/n): n
```

We have to initialize one node in NSG to the origin. It can be any node that you can see in the plot. Here, we will fix 3.

```bash
Fix an initial node. Enter node number: 3
```

Then the script will ask you to place certain nodes on the edges of the polygon.

```bash
Enter nodes. These will be placed CCW in a regular polygon vertex: 14 17 8 -1 3 5
Rotational angle on the polygon in deg: 0
```

Here, node -1 is a dummy. Any negative numbers can be used to set position of a dummy node. The rotation angle refers to the polar angle of node the first node (14). Node 17 is placed at an polar angle of 60 deg, node 8 in 120 deg, and son on.

We have to continue fixing nodes of the NSG till we get the desired representation.

```bash
Keep existing positions (y/n): n
Enter nodes. These will be placed CCW in a regular polygon vertex: 3 0 12
Rotational angle on the polygon in deg: 0
Keep existing positions (y/n): n
Enter nodes. These will be placed CCW in a regular polygon vertex: 0 22 12
Rotational angle on the polygon in deg: 90
Keep existing positions (y/n): n
Enter nodes. These will be placed CCW in a regular polygon vertex: 4 1 5 -1 19 11
Rotational angle on the polygon in deg: 0
Keep existing positions (y/n): n
Enter nodes. These will be placed CCW in a regular polygon vertex: 1 5 2
Rotational angle on the polygon in deg: 0
Keep existing positions (y/n): n
Enter nodes. These will be placed CCW in a regular polygon vertex: 11 19 10
Rotational angle on the polygon in deg: 0
```

Note that for polygons `1 5 2` and `11 19 10` the angle specified is in consequential as two of the speciefied nodes have known positions. When mentioning polygons, if more than two node positions are known, only the first two nodes are fixed. Other node positions are recalculated. You cannot UNDO or REPEAT. 

Not that the positions and NSG representation is acceptable, we choose to keep the existing positions. Then we scale the positions to ensure the distance between NSG nodes is 1.0 and then we save.  

```bash
Keep existing positions (y/n): y
...
Enter: scale
...
Keep existing positions (y/n): y
...
Enter: save
```

Using the same ideas, one can generate a representation for U1. 

The script `run2.py` would then automatically generate .png images for each NSG at different time. Using the plot of `run.py` as refrence and graphics editing software, one can generate publication quality principal transition diagram.

## References
[1]  https://pubs.acs.org/doi/abs/10.1021/acs.langmuir.2c00952
