import NPanalysis as NPa

#NPa.cluster.get_weighted_NSGs(wNSG_pickle='wNSG.pickle',  #Output
#                              GPT_pickle='GPT.pickle', #Input
#                              UniqNSGs_pickle='UniqNSGs.pickle', #Input
#                              time_pickle='time.pickle', #Input
#                              connected_pickle='connected.pickle', #Input 
#                              main_mol=0, 
#)
#

#NPa.cluster.plot_GPT_NSG(0, 
#                         GPT_pickle='GPT.pickle', #Input
#                         wNSG_pickle='wNSG.pickle', #Input
#                         time_pickle='time.pickle', #Input
#                         NSG_pos_pickle='NSG_pos.pickle', #Output in this case.
#                         node_size=600,
#                         labels=True, #Will help interactively set the positions
#                         time_scale=1,
#                         rnd_off=4,
#                         ref_UIDs=None,
#)
#
#NPa.cluster.plot_GPT_NSG(1, 
#                         GPT_pickle='GPT.pickle', #Input
#                         wNSG_pickle='wNSG.pickle', #Input
#                         time_pickle='time.pickle', #Input
#                         NSG_pos_pickle='NSG_pos.pickle', #Output in this case.
#                         node_size=600,
#                         labels=True, #Will help interactively set the positions
#                         time_scale=1,
#                         rnd_off=4,
#                         ref_UIDs=None,
#)

NPa.cluster.plot_GPT_NSG_all(GPT_pickle='GPT.pickle', #Input
                         wNSG_pickle='wNSG.pickle', #Input
                         time_pickle='time.pickle', #Input
                         NSG_pos_pickle='NSG_pos.pickle', #Output in this case.
                         node_size=600,
                         labels=False,
                         log='GPT_NSG_plot.log', 
                         time_scale=1,
                         rnd_off=4,
                         figname='GPT-NSG',
                         dpi=200,
                         arrow=True,
                         ref_UIDs=[0,1], #Determined in previous steps
)

