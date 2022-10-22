import NPanalysis as NPa

#Note UniqNSGs in GT is not the same as GPT. 

NPa.cluster.gen_GT(GT_in_pickle=None, #Input 
    UniqNSGs_in_pickle = None, #Input
    main_mol=0, #DNA
    GT_out_pickle='GT.pickle', #Output
    UniqNSGs_out_pickle = 'UniqNSGs.pickle', #Output 
    connected_pickle='connected.pickle', #Input
    time_pickle = 'time.pickle', #Input
)


NPa.cluster.plot_GT(GT_pickle='GT.pickle',
    GPT_pickle='GPT.pickle',
    time1=0, time2=0.15, #In microseconds (based on time.pickle)
    node_pos_pickle='GT_nodepos.pickle', #If the pickle file does not exist, it will be created.
    # If the pickle file exists, the node pos will be read. Typically iters=0 should be used in this case. 
    iters=2000,
    node_size=500,
    show=True,
    show_gt_edge=True, 
#   figname='GT.png', #When saving figure and show=False
#   dpi=1200, #When saving figure and show=False
)

