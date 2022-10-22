import NPanalysis as NPa

#NPa.cluster.gen_GPT(GPT_in_pickle=None, 
#    UniqNSGs_in_pickle = None, 
#    main_mol=0, #DNA
#    GPT_out_pickle='GPT.pickle', #Output
#    UniqNSGs_out_pickle = 'UniqNSGs.pickle', #Output 
#    connected_pickle='connected.pickle', #Input
#    time_pickle = 'time.pickle', #Input 
#    log=True,
#    logfile='GPT.log', #Output
#)


NPa.cluster.plot_GPT(GPT_pickle='GPT.pickle', 
    node_size=500,
    pos_node=None, 
    show=True,
    edge_label=True, 
    time_pickle='time.pickle'
)
