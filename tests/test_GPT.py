import NPanalysis as NPa
import numpy as np
import unittest
import networkx as nx
import os
import copy
class TestGPTs(unittest.TestCase):
  
    def setUp(self):
        self.D1=nx.Graph()
        self.D1.add_edges_from([(0,1),(0,2),(3,4),(5,6)])
        self.D2=nx.Graph()
        self.D2.add_edges_from([(0,1),(0,2)])
        self.D2.add_nodes_from([3,4,5,6])
        self.D3=nx.Graph()
        self.D3.add_edges_from([(0,1),(0,2),(5,6)])
        self.D3.add_nodes_from([3,4])
        self.D4=nx.Graph()
        self.D4.add_edges_from([(0,1),(0,2),(3,4)])
        self.D4.add_nodes_from([5,6])

       
    def test_change(self):
        self.assertEqual(sorted(NPa.cluster.get_changes(self.D1,self.D2)),[[3,4],[5,6]])    
        self.assertEqual(sorted(NPa.cluster.get_changes(self.D1,self.D3)),[[3,4]])    
        self.assertEqual(sorted(NPa.cluster.get_changes(self.D1,self.D4)),[[5,6]])    
        self.assertEqual(sorted(NPa.cluster.get_changes(self.D2,self.D3)),[[5,6]])    
        self.assertEqual(sorted(NPa.cluster.get_changes(self.D2,self.D4)),[[3,4]])    
        self.assertEqual(sorted(NPa.cluster.get_changes(self.D3,self.D4)),[[3,4],[5,6]])    
        
        self.assertEqual(NPa.cluster.get_changes(self.D2,self.D1),[[3,4],[5,6]])    
        self.assertEqual(NPa.cluster.get_changes(self.D3,self.D1),[[3,4]])    
        self.assertEqual(NPa.cluster.get_changes(self.D4,self.D1),[[5,6]])    
        self.assertEqual(NPa.cluster.get_changes(self.D3,self.D2),[[5,6]])    
        self.assertEqual(NPa.cluster.get_changes(self.D4,self.D2),[[3,4]])    
        self.assertEqual(NPa.cluster.get_changes(self.D4,self.D3),[[3,4],[5,6]])   

        self.assertEqual(NPa.cluster.get_changes(self.D1,self.D1),[])    
        self.assertEqual(NPa.cluster.get_changes(self.D2,self.D2),[])    
        self.assertEqual(NPa.cluster.get_changes(self.D3,self.D3),[])    
        self.assertEqual(NPa.cluster.get_changes(self.D4,self.D4),[])    
 
    def test_graphisequal(self):
        self.assertTrue(NPa.cluster.graph_is_equal(self.D1,self.D1))
        self.assertFalse(NPa.cluster.graph_is_equal(self.D1,self.D2))
        self.assertFalse(NPa.cluster.graph_is_equal(self.D3,self.D4))
        D5=copy.deepcopy(self.D4)
        D5.remove_node(5)
        self.assertFalse(NPa.cluster.graph_is_equal(D5,self.D4))

    def test_updateUniqGs(self):
        UDs=[]
        NPa.cluster.update_UniqGs(self.D1,UDs)
        self.assertEqual(len(UDs),3)
        NPa.cluster.update_UniqGs(self.D2,UDs)
        self.assertEqual(len(UDs),7)
        NPa.cluster.update_UniqGs(self.D3,UDs)
        self.assertEqual(len(UDs),7)
       
        UDs=[]
        NPa.cluster.update_UniqGs(self.D2,UDs)
        self.assertEqual(len(UDs),5)

        UDs=[]
        NPa.cluster.update_UniqGs(self.D3,UDs)
        self.assertEqual(len(UDs),4)

        UDs=[]
        NPa.cluster.update_UniqGs(self.D4,UDs)
        self.assertEqual(len(UDs),4)

        UDs=[]
        NPa.cluster.update_UniqGs(self.D3,UDs)
        NPa.cluster.update_UniqGs(self.D3,UDs)
        self.assertEqual(len(UDs),4)

        NPa.cluster.update_UniqGs(self.D3,UDs)
        NPa.cluster.update_UniqGs(self.D4,UDs)
        self.assertEqual(len(UDs),7)

        UDs=[]
        D=nx.Graph()
        NPa.cluster.update_UniqGs(D,UDs)
        self.assertEqual(len(UDs),0)

    def test_getUID(self):
        UDs=[self.D1,self.D2,self.D3,self.D4]
        self.assertEqual(NPa.cluster.get_UID(self.D1,UDs),0)
        self.assertEqual(NPa.cluster.get_UID(self.D2,UDs),1)
        self.assertEqual(NPa.cluster.get_UID(self.D3,UDs),2)
        self.assertEqual(NPa.cluster.get_UID(self.D4,UDs),3)
        
        UDs=[self.D1,self.D2,self.D4]
        with self.assertRaises(Exception) as expt:
            NPa.cluster.get_UID(self.D3,UDs)
        self.assertTrue('The graph provided is not in UniqGs. Update UniqGs!' in str(expt.exception)) 
    
    def test_hascommon(self):
        #Example aggregation
        L1=[[0],[2]]
        L2=[[1],[2]] 
        self.assertTrue(NPa.cluster.has_common_change(L1,L2))
        #Example dissociation
        L1=[[0],[1]]
        L2=[[0],[2]] 
        self.assertTrue(NPa.cluster.has_common_change(L1,L2))
        #Example internal restructuring
        L1=[[0],[1]]
        L2=[[1],[2]] 
        self.assertFalse(NPa.cluster.has_common_change(L1,L2))

    def test_reachable(self):
        # Reachability with simple internal restructuring
        G=nx.DiGraph()
        G.graph['Snodes']=[0]
        G.graph['dt']=1
        G.graph['time']=12

        G.add_edge(0,2,time=3)
        G.nodes[0]['time']=0
        G.nodes[2]['time']=6
        G.add_node(1)
        G.nodes[1]['time']=9
        reach=NPa.cluster.reachables(G)
        self.assertEqual(sorted(reach),[0,2])

    def test_reachable2(self):
        # Reachability with aggregation restructuring  (0,1)->+->2
        G=nx.DiGraph()
        G.graph['Snodes']=[0,1]
        G.graph['dt']=1
        G.graph['time']=12

        G.add_edge('ad0',2,time=3)
        G.add_edge(1,'ad0',time=3)
        G.add_edge(0,'ad0',time=3)
        G.nodes[0]['time']=0
        G.nodes[1]['time']=0
        G.nodes[2]['time']=3
        G.nodes['ad0']['num']=[2,1]
        reach=NPa.cluster.reachables(G)
        self.assertEqual(len(reach),4)
        for x in reach:
            self.assertTrue(x in [0,1,2,'ad0'])

    def test_reachable3(self):
        # Reachability with aggregation restructuring  (0,1)->x->(2,3) (4)
        G=nx.DiGraph()
        G.graph['Snodes']=[0,1]
        G.graph['dt']=1
        G.graph['time']=12

        G.add_edge('re0',2,time=3)
        G.add_edge('re0',3,time=3)
        G.add_edge(1,'re0',time=3)
        G.add_edge(0,'re0',time=3)
        G.nodes[0]['time']=0
        G.nodes[1]['time']=0
        G.nodes[2]['time']=3
        G.nodes[3]['time']=3
        G.add_node(4)
        G.nodes[4]['time']=3
        G.nodes['re0']['num']=[2,2]
        reach=NPa.cluster.reachables(G)
        self.assertEqual(len(reach),5)
        for x in reach:
            self.assertTrue(x in [0,1,2,3,'re0'])


    def test_simple_GPT(self):
        # Simple GT where GT is same as GPT with only connector nodes
        G=nx.DiGraph()
        #GT is 0 -> 1 -> 2 -> 3 
        G.add_edge(0,1,time=3)
        G.add_edge(1,2,time=6)
        G.add_edge(2,3,time=9)
        G.add_edge(0,1,time=3)
        G.nodes[0]['time']=0
        G.nodes[1]['time']=3
        G.nodes[2]['time']=6
        G.nodes[3]['time']=9
        G.graph['time']=9
        G.graph['dt']=1
        G.graph['Snodes']=[0]
        
        New_NSG_IDs=[3]
        G=NPa.cluster.remove_loop(G,New_NSG_IDs)
        #GPT should be 0 -> 1 -> 2 -> 3

        self.assertEqual(set(G.nodes()),set([0,1,2,3]))
        self.assertEqual(set(G.edges()),set([(0,1),(1,2),(2,3)]))

        self.assertTrue(G.has_edge(0,1)) 
        self.assertTrue(G.has_edge(1,2)) 
        self.assertTrue(G.has_edge(2,3))
        self.assertEqual(G[0][1]['time'],3) 
        self.assertEqual(G[1][2]['time'],6) 
        self.assertEqual(G[2][3]['time'],9) 

    def test_simple_loop_GPT(self):
        # Simple loop removal with no connector node
        # GT: 0 -> 1 -> 2 -> 0  
        G=nx.DiGraph()
        G.add_edge(0,1,time=3)
        G.add_edge(1,2,time=6)
        G.add_edge(2,0,time=9)
        G.nodes[0]['time']=9
        G.nodes[1]['time']=3
        G.nodes[2]['time']=6
        G.graph['time']=9
        G.graph['dt']=1
        G.graph['Snodes']=[0]
        
        New_NSG_IDs=[0]
        G=NPa.cluster.remove_loop(G,New_NSG_IDs)
        #GPT should be 0

        self.assertEqual(set(G.nodes()),set([0]))
    
    def test_agg_diss_GPT(self): 
        # Simple GT with Aggregation/Dissociation, where GT is same as GPT
        #
        #                       1  -> 3
        #                     /        \
        # GT, GPT is 0 -> 'ad0' -> 2->'ad1' -> 4
        G=nx.DiGraph()
        G.add_edge(0,'ad0',time=2)
        G.add_edge('ad0',1,time=2)
        G.add_edge('ad0',2,time=2)
        G.add_edge(1,3,time=4)
        G.add_edge(2,'ad1',time=6)
        G.add_edge(3,'ad1',time=6)
        G.add_edge('ad1',4,time=6)
        
        G.nodes[0]['time']=0
        G.nodes[1]['time']=2
        G.nodes[2]['time']=2
        G.nodes[3]['time']=4
        G.nodes[4]['time']=6
        G.nodes['ad0']['num']=[1,2]
        G.nodes['ad1']['num']=[2,1]
 
        G.graph['time']=6
        G.graph['dt']=1
        G.graph['Snodes']=[0]
        
        New_NSG_IDs=[4]
        G=NPa.cluster.remove_loop(G,New_NSG_IDs)

        self.assertEqual(set(G.nodes()),set([0,1,2,3,4,'ad0','ad1']))
        self.assertEqual(set(G.edges()),set([(0,'ad0'),(1,3),(2,'ad1'),
                                     (3,'ad1'),('ad0',1),('ad0',2),('ad1',4)]))

        self.assertTrue(G.has_edge(0,'ad0'))
        self.assertTrue(G.has_edge('ad0',1))
        self.assertTrue(G.has_edge('ad0',2))
        self.assertEqual(G[0]['ad0']['time'],2) 
        self.assertEqual(G['ad0'][1]['time'],2) 
        self.assertEqual(G['ad0'][2]['time'],2) 

        self.assertTrue(G.has_edge(1,3))
        self.assertEqual(G[1][3]['time'],4) 

        self.assertTrue(G.has_edge(2,'ad1'))
        self.assertTrue(G.has_edge(3,'ad1'))
        self.assertTrue(G.has_edge('ad1',4))
        self.assertEqual(G[3]['ad1']['time'],6) 
        self.assertEqual(G[2]['ad1']['time'],6) 
        self.assertEqual(G['ad1'][4]['time'],6) 

    def test_loop_agg_diss_GPT(self): 
        # GT with aggregation/dissociation and a loop. Loop is remove to get GPT
        #
        #                     1 -> 3
        #                    /      \
        # GT  is  be 0 -> 'ad0' -> 2 ->'ad1' -> 4 -> 0
        #

        G=nx.DiGraph()
        G.add_edge(0,'ad0',time=2)
        G.add_edge('ad0',1,time=2)
        G.add_edge('ad0',2,time=2)
        G.add_edge(1,3,time=4)
        G.add_edge(2,'ad1',time=6)
        G.add_edge(3,'ad1',time=6)
        G.add_edge('ad1',4,time=6)
        G.add_edge(4,0,time=8)
        
        G.nodes[0]['time']=8
        G.nodes[1]['time']=2
        G.nodes[2]['time']=2
        G.nodes[3]['time']=4
        G.nodes[4]['time']=6
        G.nodes['ad0']['num']=[1,2]
        G.nodes['ad1']['num']=[2,1]
        
        G.graph['time']=8
        G.graph['dt']=1
        G.graph['Snodes']=[0]
        
        New_NSG_IDs=[0]
        G=NPa.cluster.remove_loop(G,New_NSG_IDs)
        # GPT: 0

        self.assertEqual(set(G.nodes()),set([0]))
        
    def test_loop_agg_diss_GPT2(self): 
        # GPT with one loop begining from two starting nodes.
        # GT:
        # {0,1} -> 'ad0' -> 2
        #   ^               |
        #   |               |
        #    ----- 'ad1' <--
        # 
        G=nx.DiGraph()
        G.add_edge(0,'ad0',time=3)
        G.add_edge(1,'ad0',time=3)
        G.add_edge('ad0',2,time=3)
        G.add_edge(2,'ad1',time=6)
        G.add_edge('ad1',0,time=6)
        G.add_edge('ad1',1,time=6)
        
        G.nodes[0]['time']=6
        G.nodes[1]['time']=6
        G.nodes[2]['time']=3
        G.nodes['ad0']['num']=[2,1]
        G.nodes['ad1']['num']=[1,2]
        
        G.graph['time']=6
        G.graph['dt']=1
        G.graph['Snodes']=[0,1]
        New_NSG_IDs=[0,1]
        
        G=NPa.cluster.remove_loop(G,New_NSG_IDs)
        # GPT: {0,1}

        self.assertEqual(set(G.nodes()),set([0,1]))
        self.assertEqual(set(G.edges()),set([]))
    
    def test_exchange_GPT(self):
        # GPT and GT are the same with multiple NPs exchanging DNAs and dissociation.
        #
        #                             1 ----      4
        #                            /      \    /
        # GPT, GT should be 0 -> 'ad0' -> 2 ->'re0' -> 3
        #
        G=nx.DiGraph()
        G.add_edge(0,'ad0',time=2)
        G.add_edge('ad0',1,time=2)
        G.add_edge('ad0',2,time=2)
        G.add_edge(1,'re0',time=4)
        G.add_edge(2,'re0',time=4)
        G.add_edge('re0',3,time=4)
        G.add_edge('re0',4,time=4)
        
        G.nodes[0]['time']=0
        G.nodes[1]['time']=2
        G.nodes[2]['time']=2
        G.nodes[3]['time']=4
        G.nodes[4]['time']=4
        G.nodes['ad0']['num']=[1,2]
        G.nodes['re0']['num']=[2,2]
        
        G.graph['time']=4
        G.graph['dt']=1
        G.graph['Snodes']=[0]
        
        New_NSG_IDs=[3,4]
        G=NPa.cluster.remove_loop(G,New_NSG_IDs)

        self.assertEqual(set(G.nodes()),set([0,1,2,3,4,'ad0','re0']))
        self.assertEqual(set(G.edges()),set([(0,'ad0'),('ad0',1),('ad0',2),
                                     (1,'re0'),(2,'re0'),('re0',3),('re0',4)]))
        self.assertTrue(G.has_edge(0,'ad0'))
        self.assertTrue(G.has_edge('ad0',1))
        self.assertTrue(G.has_edge('ad0',2))
        self.assertEqual(G[0]['ad0']['time'],2) 
        self.assertEqual(G['ad0'][1]['time'],2) 
        self.assertEqual(G['ad0'][2]['time'],2) 

        self.assertTrue(G.has_edge(1,'re0'))
        self.assertTrue(G.has_edge(2,'re0'))
        self.assertTrue(G.has_edge('re0',3))
        self.assertTrue(G.has_edge('re0',4))
        self.assertEqual(G[1]['re0']['time'],4) 
        self.assertEqual(G[2]['re0']['time'],4) 
        self.assertEqual(G['re0'][3]['time'],4) 
        self.assertEqual(G['re0'][4]['time'],4) 
        
    def test_loop_exchange_GPT(self):
        # GT containing loops w.r.t exhange connector. 
        #
        #                          1 ----      4-------
        #                         /      \    /        \
        # GT  should be 0 -> 'ad0' -> 2 ->'re0' -> 3 -> 0
        #
        G=nx.DiGraph()
        G.add_edge(0,'ad0',time=2)
        G.add_edge('ad0',1,time=2)
        G.add_edge('ad0',2,time=2)
        G.add_edge(1,'re0',time=4)
        G.add_edge(2,'re0',time=4)
        G.add_edge('re0',3,time=4)
        G.add_edge('re0',4,time=4)
        G.add_edge(4,0,time=6)
        
        G.nodes[0]['time']=6
        G.nodes[1]['time']=2
        G.nodes[2]['time']=2
        G.nodes[3]['time']=4
        G.nodes[4]['time']=4
        G.nodes['ad0']['num']=[1,2]
        G.nodes['re0']['num']=[2,2]
        
        G.graph['time']=6
        G.graph['dt']=1
        G.graph['Snodes']=[0]
        
        New_NSG_IDs=[0]
        G=NPa.cluster.remove_loop(G,New_NSG_IDs)
        # GPT: {0}
        self.assertEqual(set(G.nodes()),set([0]))

    def test_exchange_GPT2(self): 
        # GT and GPT are same where GT contains an exchange node with multiple particles rearranging
        # TODO: Check for a similar case in "get_changes()"
        # GPT, GT: {0,1,2} -> 're0' -> {3,4}
        G=nx.DiGraph()
        G.add_edge(0,'re0',time=2)
        G.add_edge(1,'re0',time=2)
        G.add_edge(2,'re0',time=2)
        G.add_edge('re0',3,time=2)
        G.add_edge('re0',4,time=2)
        
        G.nodes[0]['time']=0
        G.nodes[1]['time']=0
        G.nodes[2]['time']=0
        G.nodes[3]['time']=2
        G.nodes[4]['time']=2
        G.nodes['re0']['num']=[3,2]
        
        G.graph['time']=2
        G.graph['dt']=1
        G.graph['Snodes']=[0,1,2]
        New_NSG_IDs=[3,4]
        
        G=NPa.cluster.remove_loop(G,New_NSG_IDs)

        self.assertEqual(set(G.nodes()),set([0,1,2,3,4,'re0']))
        self.assertEqual(set(G.edges()),set([(0,'re0'),(1,'re0'),(2,'re0'),('re0',3),('re0',4)]))
        self.assertEqual(G[0]['re0']['time'],2) 
        self.assertEqual(G[1]['re0']['time'],2) 
        self.assertEqual(G[2]['re0']['time'],2) 
        self.assertEqual(G['re0'][3]['time'],2) 
        self.assertEqual(G['re0'][4]['time'],2) 
      
    def test_complex_loop_GPT(self): 
        # GT with two subgraphs, one loop reduces other doesnt.
        # GT:
        # 0--                --> 3
        #    \              /
        # 3-->(ad0)-> 4 -> (ad2) --> 6
        #
        # 1--                 -->1                   1
        #    \               /             
        # 2--> (ad1) -> 5 ->(ad3) --> 2   
        #
        G=nx.DiGraph()
        G.add_edge(0,'ad0',time=3)
        G.add_edge(3,'ad0',time=3)
        G.add_edge('ad0',4,time=3)
        G.add_edge(4,'ad2',time=6)
        G.add_edge('ad2',3,time=6)
        G.add_edge('ad2',6,time=6)
        
        G.add_edge(1,'ad1',time=3)
        G.add_edge(2,'ad1',time=3)
        G.add_edge('ad1',5,time=3)
        G.add_edge(5,'ad3',time=6)
        G.add_edge('ad3',1,time=6)
        G.add_edge('ad3',2,time=6)
        
        G.nodes[0]['time']=0
        G.nodes[3]['time']=6
        G.nodes[4]['time']=3
        G.nodes[6]['time']=6
        G.nodes['ad0']['num']=[2,1]
        G.nodes['ad2']['num']=[1,2]
        
        G.nodes[1]['time']=6
        G.nodes[2]['time']=6
        G.nodes[5]['time']=3
        G.nodes['ad3']['num']=[1,2]
        G.nodes['ad1']['num']=[2,1]
        
        G.graph['time']=6
        G.graph['dt']=1
        G.graph['Snodes']=[0,1,2,3]
        New_NSG_IDs=[1,2,3,6]
        
        G=NPa.cluster.remove_loop(G,New_NSG_IDs)
        # GPT:
        # 0--                --> 3
        #    \              /
        # 3-->(ad0)-> 4 -> (ad2) --> 6
        #
        # {1,2}
        self.assertEqual(set(G.nodes()),set([0,1,2,3,4,6,'ad0','ad2']))
        self.assertEqual(set(G.edges()),set([(0,'ad0'),(3,'ad0'),('ad0',4),
                                               (4,'ad2'),('ad2',6),('ad2',3)]))
        self.assertEqual(G[0]['ad0']['time'],3) 
        self.assertEqual(G[3]['ad0']['time'],3) 
        self.assertEqual(G['ad0'][4]['time'],3) 
        self.assertEqual(G[4]['ad2']['time'],6) 
        self.assertEqual(G['ad2'][6]['time'],6) 
        self.assertEqual(G['ad2'][3]['time'],6) 

    def test_complex_loop_GPT(self):
        # GT contains two loops. The old loop should should not  be removed,
        # while new loop should be removed.
        # GT:
        #  3 ns: {0,1} -> ad0 -> 2
        #  6 ns: 2 -> ad1 -> {1,3}
        #  9 ns: {1,3} -> ad2 -> 2
        G=nx.DiGraph()
        G.add_edge(0,'ad0',time=3)
        G.add_edge(1,'ad0',time=3)
        G.add_edge('ad0',2,time=3)
        G.add_edge(2,'ad1',time=6)
        G.add_edge('ad1',1,time=6)
        G.add_edge('ad1',3,time=6)
        G.add_edge(1,'ad2',time=9)
        G.add_edge(3,'ad2',time=9)
        G.add_edge('ad2',2,time=9)
        
        
        G.nodes[0]['time']=0
        G.nodes[1]['time']=6
        G.nodes[2]['time']=9
        G.nodes[3]['time']=6
        G.nodes['ad1']['num']=[1,2]
        G.nodes['ad0']['num']=[2,1]
        G.nodes['ad2']['num']=[2,1]
        
        G.graph['time']=9
        G.graph['dt']=1
        G.graph['Snodes']=[0,1]
        New_NSG_IDs=[2]
        
        G=NPa.cluster.remove_loop(G,New_NSG_IDs)
        # GPT:
        #
        # 0--                              
        #    \                       
        # 1--> (ad0) -> 2
        #
        self.assertEqual(set(G.nodes()),set([0,1,2,'ad0']))
        self.assertEqual(set(G.edges()),set([(0,'ad0'),(1,'ad0'),('ad0',2)]))
        self.assertEqual(G[0]['ad0']['time'],3) 
        self.assertEqual(G[1]['ad0']['time'],3) 
        self.assertEqual(G['ad0'][2]['time'],3) 
 
    def test_complex_loop_GPT2(self):
        # GT containing a loop that should not be remove with a starting node. 
        # The same starting node forms a aggregation/dissociaiton loop which
        # should be removed.
        # GT:
        #
        # 3 ns: {0,1} -> 'ad0' -> 3
        # 6 ns: 3 -> 'ad1' -> {1,4}
        # 9 ns: {1,2} -> 'ad2' -> 5
        # 12 ns: 5 -> 'ad3' -> {1,2}
        G=nx.DiGraph()
        G.add_edge(0,'ad0',time=3)
        G.add_edge(1,'ad0',time=3)
        G.add_edge('ad0',3,time=3)
        G.add_edge(3,'ad1',time=6)
        G.add_edge('ad1',1,time=6)
        G.add_edge('ad1',4,time=6)
        G.add_edge(1,'ad2',time=9)
        G.add_edge(2,'ad2',time=9)
        G.add_edge('ad2',5,time=9)
        G.add_edge(5,'ad3',time=12)
        G.add_edge('ad3',1,time=12)
        G.add_edge('ad3',2,time=12)
        
        G.nodes[0]['time']=0
        G.nodes[1]['time']=12
        G.nodes[2]['time']=12
        G.nodes[3]['time']=3
        G.nodes[4]['time']=6
        G.nodes[5]['time']=9
        G.nodes['ad0']['num']=[2,1]
        G.nodes['ad1']['num']=[1,2]
        G.nodes['ad2']['num']=[2,1]
        G.nodes['ad3']['num']=[1,2]
        
        G.graph['time']=12
        G.graph['dt']=1
        G.graph['Snodes']=[0,1,2]
        New_NSG_IDs=[1,2,4]
        
        G=NPa.cluster.remove_loop(G,New_NSG_IDs)
        # GPT:
        # 3 ns: {0,1} -> 'ad0' -> 3
        # 6 ns: 3 -> 'ad1' -> {1,4}
        #
        self.assertEqual(set(G.nodes()),set([0,1,2,3,4,'ad0','ad1']))
        self.assertEqual(set(G.edges()),set([(0,'ad0'),(1,'ad0'),('ad0',3),
                                               (3,'ad1'),('ad1',1),('ad1',4)]))
        self.assertEqual(G[0]['ad0']['time'],3) 
        self.assertEqual(G[1]['ad0']['time'],3) 
        self.assertEqual(G['ad0'][3]['time'],3) 
        self.assertEqual(G[3]['ad1']['time'],6) 
        self.assertEqual(G['ad1'][1]['time'],6) 
        self.assertEqual(G['ad1'][4]['time'],6) 

    def test_complex_loop_GPT3(self):
        # GT continaing 3 loops. When the first two loops are formed it cannot be removed.
        # formation of the third loop makes all the loops removable.
        # GT:
        #
        # 3 ns: {0,1} -> 'ad0' -> 3
        # 6 ns: 3 -> 'ad1' -> {1,4}
        # 9 ns: {4,2} -> 'ad2' -> 5
        # 12 ns: 5 -> 'ad3' -> {0,6}
        # 15 ns: 6 -> 2
        G=nx.DiGraph()
        G.add_edge(0,'ad0',time=3)
        G.add_edge(1,'ad0',time=3)
        G.add_edge('ad0',3,time=3)
        G.add_edge(3,'ad1',time=6)
        G.add_edge('ad1',1,time=6)
        G.add_edge('ad1',4,time=6)
        G.add_edge(4,'ad2',time=9)
        G.add_edge(2,'ad2',time=9)
        G.add_edge('ad2',5,time=9)
        G.add_edge(5,'ad3',time=12)
        G.add_edge('ad3',0,time=12)
        G.add_edge('ad3',6,time=12)
        G.add_edge(6,2,time=15)
        
        G.nodes[0]['time']=12
        G.nodes[1]['time']=6
        G.nodes[2]['time']=15
        G.nodes[3]['time']=3
        G.nodes[4]['time']=6
        G.nodes[5]['time']=9
        G.nodes[6]['time']=12
        G.nodes['ad0']['num']=[2,1]
        G.nodes['ad1']['num']=[1,2]
        G.nodes['ad2']['num']=[2,1]
        G.nodes['ad3']['num']=[1,2]
        
        G.graph['time']=15
        G.graph['dt']=1
        G.graph['Snodes']=[0,1,2]
        New_NSG_IDs=[0,1,2]
        
        G=NPa.cluster.remove_loop(G,New_NSG_IDs)
        # GPT:
        # {0,1,2}
        #
        self.assertEqual(set(G.nodes()),set([0,1,2]))
        self.assertEqual(set(G.edges()),set([]))

if __name__ == '__main__':
    unittest.main()
