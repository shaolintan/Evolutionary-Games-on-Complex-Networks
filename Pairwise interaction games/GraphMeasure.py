import networkx as nx
import operator as op

# calculate the degree heterogeneity of a graph
def Heter_graph(g):
    deg=g.degree()
    aver_deg=2*float(g.size())/g.order()
    heter=0.0
    for d in g.nodes_iter():
        heter+=pow((deg[d]-aver_deg),2)
    heter=float(heter)/g.order()
    return heter

# Tranlate a non-weighted graph to weighted graph
def to_weighted(g):
    if not nx.is_directed(g): g=g.to_directed()
    b={}
    for e in g.edges():
        b[e]=1
    nx.set_edge_attributes(g,'weight',b)           
    outdegree=g.out_degree(weight='weight')
    for d1 in g.nodes_iter():
        for d2 in g.successors_iter(d1):
            g[d1][d2]['weight']=g[d1][d2]['weight']/float(outdegree[d1])      
    return g
        
# calculate the temperature sequence of a graph
def Temp(g):
    g=to_weighted(g)
    temp1=g.in_degree(weight='weight')
    temp=[]
    for d in g.nodes():
        temp.append(temp1[d])
    return temp

#calculate the average temperature of a weighted Digraph
def Aver_Temp(g):
    g=to_weighted(g)
    temp=g.in_degree(weight='weight')
    aver=0
    node_num=g.order()
    for d in g.nodes_iter():
        aver+=temp[d]
    aver=float(aver)/node_num
    return aver

# calculate the temperature variance of a weighted Digraph
def Temp_var(g):
    g=to_weighted(g)
    temp=g.in_degree(weight='weight')
    aver=Aver_Temp(g)
    var=0
    for d in g.nodes_iter():
        var+=pow((temp[d]-aver),2)
    var=float(var)/g.order()
    return var

# calculate the inverse degree of an undirected graph
def inverse_degree(g):
    deg=g.degree()
    inv_deg=[]
    mid=0
    for d in g.nodes_iter():
        mid=1.0/deg[d]
        inv_deg.append((d,mid))
    return inv_deg

# calculate the average inverse_degree of an undirectedd graph
def aver_invdeg(g):
    inv_deg=inverse_degree(g)
    aver=0
    for v in inv_deg:
        aver+=v[1]
    aver=float(aver)/len(inv_deg)
    return aver

# calculate the inverse degree heterogeneity of an undirected graph
def heter_invdeg(g):
    heter=0
    inv_deg=inverse_degree(g)
    aver=aver_invdeg(g)
    for v in inv_deg:
        heter+=pow((v[1]-aver),2)
    heter=float(heter)/len(inv_deg)
    return heter

# calculate the convection of two unjiont subset in a graph
def convection(g,nbunch,r):
    g=to_weighted(g)
    conv1=0
    conv2=0
    comp=set(g.nodes())-set(nbunch)
    for d1 in nbunch:
        for d2 in set.intersection(comp,set(g.neighbors(d1))):
            conv1+=g[d1][d2]['weight']*r
            conv2+=g[d2][d1]['weight']
    conv3=float(conv1)/(conv1+conv2)
    conv4=float(conv2)/(conv1+conv2)
    return (conv3,conv4)

# calculate the convection sequence of a graph
def convection_seq(g,r):
    seq=[]
    nbunch=[]
    g1=to_weighted(g)
    temp=g1.in_degree(weight='weight')
    temp=temp.items()
    temp.sort(key=op.itemgetter(1),reverse=True)
    for i in range(len(temp)-1):
        nbunch.append(temp[i][0])
        conv=convection(g,nbunch,r)
        seq.append(conv[0])
    return seq

# calculate the sum of the inverse temprature
def inverse_temp(g):
    g=to_weighted(g)
    temp=g.in_degree(weight='weight')
    inv={}
    invsum=0
    for d in g.nodes():
        inv[d]=1.0/temp[d]
        invsum+=1.0/temp[d]
    return (inv,invsum)

# caculate the sum of the inverse (tempreture times degree)
def inv_tempdeg(g,r):
    g=to_weighted(g)
    temp=g.in_degree(weight='weight')
    deg=g.degree()
    invsum=0
    for d in g.nodes():
        invsum+=1.0/deg[d]/(r+temp[d])
    return invsum

# calculate the degree heterogeneity of a undirect and connected graph
def deg_heter(g):
    deg=g.degree()
    edges=nx.number_of_edges(g)
    nodes=nx.number_of_nodes(g)
    aver_deg=2*edges/float(nodes)
    degsum=0
    for d in g.nodes():
        degsum+=(deg[d]-aver_deg)**2
    degsum=degsum/float(nodes)
    return degsum


       
    
    

    
    
    
    
    



    
        
    
            
    
        
        
    
