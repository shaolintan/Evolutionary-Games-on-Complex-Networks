import networkx as nx
import random 
import GraphMeasure as gm
from operator import itemgetter
import math
import numpy as np

# graph used here are undirected unweighted
# initiate the nodes in nbunch as mutants
def add_dynamic_attributes(g,nbunch):
    a={}
    nodes=nx.nodes(g)
    b=set(nodes)-set(nbunch)
    for d1 in nbunch:
        a[d1]=1
    for d2 in b:
        a[d2]=0
    nx.set_node_attributes(g,'type',a)        
    return g

#initiate the first time of each node
def add_first_time(g,nbunch,N):
    a={}
    nodes=nx.nodes(g)
    b=set(nodes)-set(nbunch)
    for d1 in nbunch:
        a[d1]=1
    for d2 in b:
        a[d2]=N
    nx.set_node_attributes(g,'first_time',a)
    return g

#replace the typre of the dead node with the birth node.
def update_type(g,death,birth):
    g.node[death]['type']=g.node[birth]['type']
    return g

# replace the type of the dead node with the birth node with mutation rate u
def mutated_update_type(g,death,birth,u=0):
    p=random.random()
    if p>=u:
        g.node[death]['type']=g.node[birth]['type']
    else:
        q=random.random()
        if q>=0.5: g.node[death]['type']=0
        else: g.node[death]['type']=1
    return g
    
# test whether all the nodes have the same type
def test_nodes_type(g):
    result=0
    for d1 in g.nodes_iter():
        result+=g.node[d1]['type']
    result=float(result)/g.order()
    return result



#--------------------------here for 'death=birth' process--------------------
# random choose a individual to die as in the "death-birth" process
def DB_random_death(g):
    nodes=nx.nodes(g)
    death=random.sample(nodes,1)  # death is a list
    return death

# selecte an neighbor of individual d with probability proportional to their fitness
def DB_selected_birth(g,d,r):
    neighbors=g.neighbors(d)
    totalfit=0
    for d1 in neighbors:
        totalfit=totalfit+1+(r-1)*g.node[d1]['type']
    rand=random.random()
    start=0
    end=(g.node[neighbors[0]]['type']*(r-1)+1)/float(totalfit)
    for i in range(len(neighbors)):
        if (rand>=start) and (rand<end):
            birth=neighbors[i]    # birth is node
            break
        start=end
        end=end+(g.node[neighbors[i+1]]['type']*(r-1)+1)/float(totalfit)
    return birth

# simulate the "death-birth" process with initial mutant set as nbunch
def death_birth_simulation(g,nbunch,r):
    g=add_dynamic_attributes(g,nbunch)
    while True:
        death=DB_random_death(g)
        birth=DB_selected_birth(g,death[0],r)
        g=update_type(g,death[0],birth)
        test=test_nodes_type(g)
        if test==0 or test==1: break
    return test

# do the simulation for N times to approximately compute the fixation probability
def death_birth_fixation(g,nbunch,r,N):
    fix_prob=0
    i=0
    while i<N:
        result=death_birth_simulation(g,nbunch,r)
        fix_prob+=result
        i+=1
    fix_prob=float(fix_prob)/N
    return fix_prob

# find the fixation probability for different kinds of fitness in (0,10)
def DB_fixation_vs_fitness(g,nbunch,N):
    fix_prob=[]
    r=range(4,100,4)
    for i in range(len(r)):
        r[i]=r[i]/10.0
        fix=death_birth_fixation(g,nbunch,r[i],N)
        fix_prob.append(fix)
    return (r,fix_prob)

# find the fixation probability for different initial set of mutants
def DB_fixation_vs_mutantset(g,r,N):
    fix_prob=[]
    nodes=[u'0',u'1',u'2',u'3',u'4',u'5',u'6',u'7',u'8',u'9',u'10',u'11',u'12',\
       u'13',u'14',u'15',u'16',u'17',u'18',u'19']
    for i in range(len(nodes)):
        nbunch=nodes[:i+1]
        fix=death_birth_fixation(g,nbunch,r,N)
        fix_prob.append(fix)
    return (nodes,fix_prob)

# find the node which have the maximal marginal effect on the fixaiton probability
def DB_max_margin(g,nbunch,r,N):
    nodes=nx.nodes(g)
    fix_seq=[]
    b=list(set(nodes)-set(nbunch))
    for i in range(len(b)):
        newset=nbunch+[b[i]]
        fix=death_birth_fixation(g,newset,r,N)
        fix_seq.append((newset,fix))
    index=max(fix_seq, key=itemgetter(1))
    result=index[0]
    return result

# Iterate N times to approximate the fixaiton probability.
def approximate_death_birth_simulation(g,nbunch,r):
    g=add_dynamic_attributes(g,nbunch)
    fix_times=0
    while True:
        death=DB_random_death(g)
        birth=DB_selected_birth(g,death[0],r)
        g=update_type(g,death[0],birth)
        test=test_nodes_type(g)
        fix_times+=1
        if g.order()*test>=2./math.log10(r):
            test=1
            break
        elif test==0: break
    return (test,fix_times)

# do the simulation for M times to approximately compute the fixation probability
def approximate_death_birth_fixation(g,nbunch,r,M):
    fix_prob=0
    fix_times=0
    for i in range(M):
        result=approximate_death_birth_simulation(g,nbunch,r)
        fix_prob+=result[0]
        fix_times+=result[1]
        i+=1
    fix_prob=float(fix_prob)/M
    fix_times=fix_times/float(M)
    return (fix_prob,fix_times)

# find the average fixation probability for one random mutants
def DB_average_fixation(g,r,N):
    aver=0
    for d in g.nodes_iter():
        fix=death_birth_fixation(g,[d],r,N)
        aver+=fix
    aver=float(aver)/g.order()
    return aver

# find the fixation probability for different graphs with fixed nodes.
def DB_fixation_vs_seqgraph(r,K,M,N):
    fix_prob=[]
    var=[]
    i=0
    gseq=[]
    while i<=M:
        seq=[]
        for j in range(K):
            a=random.randint(1,K-1)
            seq.append(a)
        try:
            g=nx.random_degree_sequence_graph(seq)
            if not(nx.is_connected(g)):
                continue
        except(nx.NetworkXUnfeasible, nx.NetworkXError):
            continue
        result=DB_average_fixation(g,r,N)
        temp=gm.Temp_var(g)
        fix_prob.append(result)
        var.append(temp)
        gseq.append(g)
        i=i+1
    return (var,fix_prob,gseq)

# compare the heat heterogeneity with the mixing assortativity.
def heat_asso(K,M):
    var=[]
    ass=[]
    i=0
    while i<=M:
        seq=[]
        for j in range(K):
            a=random.randint(1,K-1)
            seq.append(a)
        try:
            g=nx.random_degree_sequence_graph(seq)
            if not(nx.is_connected(g)):
                continue
        except(nx.NetworkXUnfeasible, nx.NetworkXError):
            continue
        temp=gm.Temp_var(g)
        var.append(temp)
        mid=nx.degree_assortativity_coefficient(g)
        ass.append(mid)
        i=i+1
    return (var,ass)

    
#------------------------------here for 'birth-death' process---------------------
# selecte the individual for reproducation with a probability proportional to their fitness
def BD_selected_birth(g,r):
    nodes=g.nodes()
    totalfit=0
    for d1 in nodes:
        totalfit=totalfit+1+(r-1)*g.node[d1]['type']
    rand=random.random()
    start=0
    end=(g.node[nodes[0]]['type']*(r-1)+1)/float(totalfit)
    for i in range(len(nodes)):
        if (rand>=start) and (rand<end):
            birth=nodes[i]    # birth is node
            break
        start=end
        end=end+(g.node[nodes[i+1]]['type']*(r-1)+1)/float(totalfit)
    return birth


# choose a neighbor of individual d for replacement randomly
def BD_random_death(g,d):
    neighbors=g.neighbors(d)
    death=random.sample(neighbors,1)  # death is a list
    return death


# simulate the "birth-death" process with initial mutant set as nbunch
def birth_death_simulation(g,nbunch,r):
    g=add_dynamic_attributes(g,nbunch)
    while True:
        birth=BD_selected_birth(g,r)
        death=BD_random_death(g,birth)
        g=update_type(g,death[0],birth)
        test=test_nodes_type(g)
        if test==0 or test==1: break
    return test

# do the simulation for N times to approximately compute the fixation probability
def birth_death_fixation(g,nbunch,r,N):
    fix_prob=0
    for i in range(N):
        result=birth_death_simulation(g,nbunch,r)
        fix_prob+=result
        i+=1
    fix_prob=float(fix_prob)/N
    return fix_prob


# find the average fixation probability for one random mutants
def BD_average_fixation(g,r,N):
    aver=0
    for d in g.nodes_iter():
        fix=birth_death_fixation(g,[d],r,N)
        aver+=fix
    aver=float(aver)/g.order()
    return aver


# find the fixation probability for random graphs with fixed nodes.
def BD_fixation_vs_randomgraph(r,M,N):
    fix_prob=[]
    var=[]
    for i in range(M):
        g=nx.erdos_renyi_graph(10,0.4)
        if not(nx.is_connected(g)): continue
        result=BD_average_fixation(g,r,N)
        g1=gm.to_weighted(g)
        temp=gm.Temp_var(g1)
        fix_prob.append(result)
        var.append(temp)
    return (var,fix_prob)


# find the fixation probability for different graphs with fixed nodes.
def BD_fixation_vs_seqgraph(r,K,M,N):
    fix_prob=[]
    var=[]
    inv_temp=[]
    i=0
    gseq=[]
    while i<=M:
        seq=[]
        for j in range(K):
            a=random.randint(1,K-1)
            seq.append(a)
        try:
            g=nx.random_degree_sequence_graph(seq)
            if not(nx.is_connected(g)):
                continue
        except(nx.NetworkXUnfeasible, nx.NetworkXError):
            continue
        result=BD_average_fixation(g,r,N)
        temp=gm.Temp_var(g)
        inv=gm.inverse_temp(g)
        fix_prob.append(result)
        var.append(temp)
        inv_temp.append(inv)
        gseq.append(g)
        i=i+1
    return (var,inv_temp,fix_prob,gseq)

# find the fixation probability for random geometric graph with 12 nodes
def BD_fixation_vs_geograph(r,M,N):
    fix_prob=[]
    var=[]
    inv_deg=[]
    heterdeg=[]
    i=0
    while i<=M:
        g=nx.random_geometric_graph(12,0.4)
        if not(nx.is_connected(g)): continue
        result=BD_average_fixation(g,r,N)
        g1=gm.to_weighted(g)
        temp=gm.Temp_var(g1)
        mid=gm.heter_invdeg(g1)
        deg=gm.Heter_graph(g1)
        fix_prob.append(result)
        var.append(temp)
        inv_deg.append(mid)
        heterdeg.append(deg)
        i=i+1
    return (var,inv_deg,heterdeg,fix_prob)


#-----------here it is a fast algorithm for 'birth-death' process---------------

# find the individuals in the boundaries of mutants and residents
def state_bounderies(g):
    bounderies=[]
    for d1 in g.nodes_iter():
        for d2 in g.neighbors_iter(d1):
            if g.node[d1]['type']!=g.node[d2]['type']:
                bounderies.append(d1)
                break
    return bounderies

# fast BD birth selection algorithm
def fast_BD_selected_birth(g,r,bound):
    nodes=g.nodes()
    totalfit=0
    boundfit=0
    for d1 in nodes:
        totalfit=totalfit+1+(r-1)*g.node[d1]['type']
    for d2 in bound:
        boundfit=boundfit+1+(r-1)*g.node[d2]['type']
    while True:
        rand=random.random()
        if (rand>=0) and (rand<boundfit/float(totalfit)):
            start=0
            end=(g.node[bound[0]]['type']*(r-1)+1)/float(totalfit)
            for i in range(len(bound)):
                if (rand>=start) and (rand<end):
                     birth=bound[i]    # birth is node
                     break
                start=end
                end=end+(g.node[bound[i+1]]['type']*(r-1)+1)/float(totalfit)
            break
    return birth

# fast_simulate the "birth-death" process with initial mutant set as nbunch
def fast_birth_death_simulation(g,nbunch,r):
    g=add_dynamic_attributes(g,nbunch)
    fix_time=0
    while True:
        bound=state_bounderies(g)
        birth=fast_BD_selected_birth(g,r,bound)
        death=BD_random_death(g,birth)
        g=update_type(g,death[0],birth)
        test=test_nodes_type(g)
        fix_time+=1
        if test==1 or g.order()*test>=2./math.log10(r):
            test=1
            break
        elif test==0: break
    return (test,fix_time)

# do the simulation for N times to approximately compute the fixation probability
def fast_birth_death_fixation(g,nbunch,r,N):
    fix_prob=0
    fix_time=0
    for i in range(N):
        result=fast_birth_death_simulation(g,nbunch,r)
        fix_prob+=result[0]
        fix_time+=result[1]
        i+=1
    fix_prob=float(fix_prob)/N
    fix_time=float(fix_time)/N
    return (fix_prob,fix_time)


# find the average fixation probability for one random mutants
def fast_BD_average_fixation(g,r,N):
    aver=0
    for d in g.nodes_iter():
        fix=fast_birth_death_fixation(g,[d],r,N)
        aver+=fix
    aver=float(aver)/g.order()
    return aver

# find the fixation probability for different graphs with fixed nodes.
def fast_BD_fixation_vs_seqgraph(r,K,M,N):
    fix_prob=[]
    hetertemp=[]
    i=0
    gseq=[]
    while i<=M:
        seq=[]
        for j in range(K):
            a=random.randint(1,K-1)
            seq.append(a)
        try:
            g=nx.random_degree_sequence_graph(seq)
            if not(nx.is_connected(g)):
                continue
        except(nx.NetworkXUnfeasible, nx.NetworkXError):
            continue
        result=BD_average_fixation(g,r,N)
        g1=gm.to_weighted(g)
        temp=gm.Temp_var(g1)
        fix_prob.append(result)
        hetertemp.append(temp)
        gseq.append(g)
        i=i+1
    return (hetertemp,fix_prob,gseq)

# do approximate simulation for birth-death process
def approximate_birth_death_simulation(g,nbunch,r):
    g=add_dynamic_attributes(g,nbunch)
    fix_time=0
    while True:
        birth=BD_selected_birth(g,r)
        death=BD_random_death(g,birth)
        g=update_type(g,death[0],birth)
        test=test_nodes_type(g)
        fix_time+=1
        if test==1 or g.order()*test>=2./math.log10(r):
            test=1
            break
        elif test==0: break
    return (test,fix_time)

# find approximate fixation probability for birth-death process
def approximate_birth_death_fixation(g,nbunch,r,N):
    fix_prob=0
    fix_time=0
    for i in range(N):
        result=approximate_birth_death_simulation(g,nbunch,r)
        fix_prob+=result[0]
        fix_time+=result[1]
        i+=1
    fix_prob=float(fix_prob)/N
    fix_time=float(fix_time)/N
    return (fix_prob,fix_time)
    
    
#-----------------------here for evolutionary process with mutation--------------
# birth-death process with mutation
def mutated_birth_death_simulation(g,r,T,u):
    nbunch=random.sample(g.nodes(),int(g.order()/200))
    g=add_dynamic_attributes(g,nbunch)
    mutant_rate=[]
    i=0
    for i in range(T):
        birth=BD_selected_birth(g,r)
        death=BD_random_death(g,birth)
        g=mutated_update_type(g,death[0],birth,u)
        test=test_nodes_type(g)
        mutant_rate.append(test)
        i+=1
    return mutant_rate

# death-birth process with mutation
def mutated_death_birth_simulation(g,r,T,u):
    g=add_dynamic_attributes(g,[])
    mutants=[]
    i=0
    for i in range(T):
        death=DB_random_death(g)
        birth=DB_selected_birth(g,death[0],r)
        g=mutated_update_type(g,death[0],birth,u)
        test=test_nodes_type(g)
        mutants.append(test)
        i+=1
    return mutants

#derive the average increasment of the mutants' number
def increase_mutant(N,r,T,u):
    x=[0]
    for i in range(T):
        a=(N-x[i])*x[i]*(1-u)*(r-1)/float(N*(r*x[i]+N-x[i]))+u*(N-2*x[i])/float(2*N)
        x.append(x[i]+a)
    return x

    
#-------------------------here for birth-death game dynamics--------------------
# calculate the fitness of each individual
def calculate_fitness(g,pm,w):
    payoff={}
    for d1 in g.nodes_iter():
        payoff[d1]=0
        for d2 in g.neighbors_iter(d1):
            mid1=np.dot([g.node[d1]['type'],1-g.node[d1]['type']],pm)
            payoff[d1]+=np.dot(mid1,[g.node[d2]['type'],1-g.node[d2]['type']])
        payoff[d1]=1-w+w*payoff[d1]
    nx.set_node_attributes(g,'fitness',payoff)
    return g

# selecte the individual for reproducation with a probability proportional to their fitness
def BD_selected_birth_game(g):
    nodes=g.nodes()
    totalfit=0
    for d1 in nodes:
        totalfit=totalfit+g.node[d1]['fitness']
    rand=random.random()
    start=0
    end=g.node[nodes[0]]['fitness']/float(totalfit)
    for i in range(len(nodes)):
        if (rand>=start) and (rand<end):
            birth=nodes[i]    # birth is node
            break
        start=end
        end=end+g.node[nodes[i+1]]['fitness']/float(totalfit)
    return birth

# update the population states
def update_type_game(g,game,death,birth,w):
    former=g.node[death]['type']
    g.node[death]['type']=g.node[birth]['type']
    for d1 in g.neighbors_iter(death):
        pheno=g.node[d1]['type']
        g.node[d1]['fitness']=g.node[d1]['fitness']+\
          w*(game[pheno][1]-game[pheno][0])*(former-g.node[birth]['type'])
        g.node[death]['fitness']=g.node[death]['fitness']+\
          w*(game[1][pheno]-game[0][pheno])*(former-g.node[birth]['type'])
    return g  
    
# simulation the evolutionary game
def BD_game_simulation(g,game,w,nbunch):
    g=calculate_fitness(g,game,w)
    while True:
        birth=BD_selected_birth_game(g)
        death=BD_random_death(g,birth)
        g=update_type_game(g,game,death[0],birth,w)
        test=test_nodes_type(g)
        if test==0 or test==1: break
    return test

# simulate to get the fixation probability in the evolutionary game
def BD_game_fixation(g,game,w,nbunch,N):
    fix_prob=0
    for i in range(N):
        result=BD_game_simulation(g,game,w,nbunch)
        fix_prob+=result
        i+=1
    fix_prob=float(fix_prob)/N
    return fix_prob
        
# find the average fixation probability in evolutionary game
def BD_game_average_fixation(g,game,w,N):
    aver=0
    for d in g.nodes_iter():
        fix=BD_game_fixation(g,game,w,[d],N)
        aver+=fix
    aver=float(aver)/g.order()
    return aver

# find the fixation probability for different graphs with fixed nodes in evolutioanry games.
def BD_game_fixation_vs_seqgraph(game,w,K,M,N):
    fix_prob=[]
    var=[]
    inv_temp=[]
    i=0
    gseq=[]
    while i<=M:
        seq=[]
        for j in range(K):
            a=random.randint(1,K-1)
            seq.append(a)
        try:
            g=nx.random_degree_sequence_graph(seq)
            if not(nx.is_connected(g)):
                continue
        except(nx.NetworkXUnfeasible, nx.NetworkXError):
            continue
        result=BD_game_average_fixation(g,game,w,N)
        temp=gm.Temp_var(g)
        inv=gm.inverse_temp(g)
        fix_prob.append(result)
        var.append(temp)
        inv_temp.append(inv)
        gseq.append(g)
        i=i+1
    return (var,inv_temp,fix_prob,gseq)
    
    
#---------------------------record the evolving process------------------------
# simulate the "birth-death" process with initial mutant set as nbunch
def record_BD_simulation(g,nbunch,r):
    g=add_dynamic_attributes(g,nbunch)
    record={}
    for d in g.nodes_iter():
        record[d]=[]
        t=0
    while True:
        birth=BD_selected_birth(g,r)
        death=BD_random_death(g,birth)
        g=update_type(g,death[0],birth)
        for d in g.nodes_iter():
            record[d].append(g.node[d]['type'])
        test=test_nodes_type(g)
        t+=1
        if test==0 or test==1: break
    return (test,record,t)

# record the spreading sequence of mutants in nbunch
def record_BD_spreading(g,nbunch,r,N):
    frequence={}
    for d in g.nodes_iter():
        frequence[d]=np.zeros(10000)
    tmin=10000
    for i in range(N):
        result=record_BD_simulation(g,nbunch,r)
        if result[0]==1:
            i=i+1
            tmin=min(tmin,result[2])
            for d in g.nodes_iter():
                frequence[d]=np.add(result[1][d][:tmin],frequence[d][:tmin])
    for d in g.nodes_iter():
        frequence[d]=frequence[d]/float(N)
    return frequence


# record the spreading sequence of a random mutant
def record_BD_random_spreading(g,r,N):
    frequence={}
    for d in g.nodes_iter():
        frequence[d]=np.zeros(10000)
    tmin=10000
    for d1 in g.nodes_iter():
        result=record_BD_spreading(g,[d1],r,N)
        tmin=min(tmin,len(result[d1]))
        for d2 in g.nodes_iter():
            frequence[d2]=np.add(result[d2][:tmin],frequence[d2][:tmin])
    return frequence


# ---------------------------------Analytical results of birth-death process on small-order graphs--------------------------------------------------------
# translate a dec into bin
def bin(x,N):
	result=[]
	x=int(x)
	while x>0:
            mod=x%2
	    x/=2
	    result.append(mod)
	if len(result)<N:
            for i in range(N-len(result)):
                result.append(0)
	return result

# given a graph and it dead individual, find the translation probability from state i to j
def BD_trans_prob(g,r,k,i,j):
    result=0
    N=g.order()
    bin_i=bin(i,N)
    bin_j=bin(j,N)
    fit_i=np.zeros(N)
    for m in range(N):
        fit_i[m]=(r-1)*bin_i[m]+1
    for d in g.neighbors(k):
        p=float(fit_i[d])/fit_i.sum()/float(g.degree()[d])*abs(bin_i[d]-bin_i[k])
        result+=p
    return result
        
# given a graph, find the translation matrix
def BD_trans_mat(g,r):
    N=g.order()
    T=np.zeros((2**N,2**N))
    for i in range(2**N):
        bin_i=bin(i,N)
        for j in range(2**N):
            bin_j=bin(j,N)
            bin_c=np.bitwise_xor(bin_i,bin_j)
            diff=bin_c.sum()
            if diff==1:
                k=np.nonzero(bin_c)
                p=BD_trans_prob(g,r,k[0][0],i,j)
                T[i][j]=p
    for i in range(2**N):
        T[i][i]=1-T.sum(1)[i]
    return T

# compute the fixation probability.
def BD_fix_prob(g,r):
    N=g.order()
    T=BD_trans_mat(g,r)
    Q=T[1:2**N-1,1:2**N-1]
    S=np.zeros((2**N-2,2))
    S[:,0]=T[1:2**N-1,0]
    S[:,1]=T[1:2**N-1,2**N-1]
    I=np.eye(2**N-2)
    Y=np.linalg.inv(I-Q)
    B=np.dot(Y,S)
    return B
    
#-------------------------some special statistics of the birth-death process-----    

# compute the number of 0-1,0-0,1-1 edges in the graph
def zero_one_edges(g):
    num=0
    num1=0
    num0=0
    for (a,b) in g.edges_iter():
        if g.node[a]['type']!=g.node[b]['type']:
            num+=1
        elif g.node[a]['type']==1 and g.node[b]['type']==1:
            num1+=1
        elif g.node[a]['type']==0 and g.node[b]['type']==0:
            num0+=1
    return (float(num)/g.size(),float(num1)/g.size(),float(num0)/g.size())

# count the number of mutant in a population
def num_of_mutants(g):
    num=0
    for d in g.nodes():
        num+=g.node[d]['type']
    return float(num)/g.order()

# record the change over time of the 0-1 edges and mutants in the evolving process
def BD_change_of_01edges(g,r,N):
    num01=[]
    num00=[]
    num11=[]
    num2=[]
    nodes=g.nodes()
    nbunch=random.sample(nodes,1)
    g=add_dynamic_attributes(g,nbunch)
    mid1=zero_one_edges(g)
    mid2=num_of_mutants(g)
    num01.append(mid1[0])
    num11.append(mid1[1])
    num00.append(mid1[2])
    num2.append(mid2)
    i=0
    while i<N:
        birth=BD_selected_birth(g,r)
        death=BD_random_death(g,birth)
        g=update_type(g,death[0],birth)
        mid1=zero_one_edges(g)
        num01.append(mid1[0])
        num11.append(mid1[1])
        num00.append(mid1[2])
        mid2=num_of_mutants(g)
        num2.append(mid2)
        i+=1
        test=test_nodes_type(g)
        if test==0 or test==1:
            nbunch=random.sample(nodes,1)
            g=add_dynamic_attributes(g,nbunch)       
    return (num01,num11,num00,num2)

# record the average change over time of the 0-1 edges and mutants in the evolving process
def BD_average_change_of_01edges(g,r,N,M):
    nodes=g.nodes()
    nbunch=random.sample(nodes,1)
    num=BD_change_of_01edges(g,r,nbunch,N)
    aver_num01=num[0]
    aver_num11=num[1]
    aver_num00=num[2]
    aver_num2=num[3]
    i=1
    while i<M:
        nbunch=random.sample(nodes,1)
        num=BD_change_of_01edges(g,r,nbunch,N)
        aver_num01=[x+y for x,y in zip(num[0],aver_num01)]
        aver_num11=[x+y for x,y in zip(num[1],aver_num11)]
        aver_num00=[x+y for x,y in zip(num[2],aver_num00)]
        aver_num2=[x+y for x,y in zip(num[3],aver_num2)]
        i+=1
    aver_num01=[float(x)/M for x in aver_num01]
    aver_num11=[float(x)/M for x in aver_num11]
    aver_num00=[float(x)/M for x in aver_num00]
    aver_num2=[float(x)/M for x in aver_num2]
    return (aver_num01,aver_num11,aver_num00,aver_num2)

# record the change of 01edges with the mutants in the evolving process
def BD_edges_vs_mutants(g,r,N,M):
    nodes=g.nodes()
    nbunch=random.sample(nodes,1)
    num=BD_change_of_01edges(g,r,nbunch,N)
    aver_num01=num[0]
    aver_num11=num[1]
    aver_num00=num[2]
    aver_num2=num[3]
    i=1
    while i<M:
        nbunch=random.sample(nodes,1)
        num=BD_change_of_01edges(g,r,nbunch,N)
        aver_num01=aver_num01+num[0]
        aver_num11=aver_num11+num[1]
        aver_num00=aver_num00+num[2]
        aver_num2=aver_num2+num[3]
        i+=1
    return (aver_num01,aver_num11,aver_num00,aver_num2)

# record the number of mutants,01edges,00edges and 11edges on the mutant-resident board
def board_variant(g):
    board_mu=[]
    board_re=[]
    mun_00=0
    num_11=0
    for d1 in g.nodes():
        for d2 in g.neighbors(d1):
            if g.node[d1]['type']!=g.node[d2]['type']:
                if g.node[d1]['type']==1:
                    board_mu.append(d1)
                elif g.node[d1]['type']==0:
                    board_re.append(d1)
    board_mu=set(board_mu)
    board_re=set(board_re)
    for d3 in board_mu:
        for d4 in g.neighbors(d3):
            if g.node[d3]['type']==g.node[d4]['type']:
                num_11+=1
    for d5 in board_re:
        for d6 in g.neighbors(d5):
            if g.node[d5]['type']==g.node[d6]['type']:
                num_00+=1
    num_mu=len(board_mu)
    num_re=len(board_re)
    return (float(num_mu)/g.order(),float(num_re)/g.order(),float(num_00)/g.size(),float(num_11)/g.size())

# record the change of the board variant in evolving process
def BD_board_variant(g,r,nbunch,N):
    num01=[]
    num00=[]
    num11=[]
    num_mu=[]
    num_re=[]
    total_mu=[]
    g=add_dynamic_attributes(g,nbunch)
    mid=board_variant(g)
    num_mu.append(mid[0])
    num_re.append(mid[1])
    num00.append(mid[2])
    num11.append(mid[4])
    mid2=num_of_mutants(g)
    mid3=zero_one_edges(g)
    num01.append(mid3[0])
    total_mu.append(mid2)
    i=0
    while i<N:
        birth=BD_selected_birth(g,r)
        death=BD_random_death(g,birth)
        g=update_type(g,death[0],birth)
        mid=board_variant(g)
        num_mu.append(mid[0])
        num_re.append(mid[1])
        num00.append(mid[2])
        num11.append(mid[4])
        mid2=num_of_mutants(g)
        total_mu.append(mid2)
        mid3=zero_one_edges(g)
        num01.append(mid3[0])
        i+=1
    return (num_mu,num_re,num_00,num_01,num_11,total_mu)

# record the average change of the board variant in the evolving process
def BD_average_board_variant(g,r,nbunch,N,M):
    nodes=g.nodes()
    nbunch=random.sample(nodes,1)
    num=BD_board_variant(g,r,nbunch,N)
    aver_num01=num[3]
    aver_num11=num[4]
    aver_num00=num[2]
    aver_num_mu=num[0]
    aver_num_re=num[1]
    aver_mu=num[5]
    i=1
    while i<M:
        nbunch=random.sample(nodes,1)
        num=BD_board_variant(g,r,nbunch,N)
        aver_num01=aver_num01+num[3]
        aver_num11=aver_num11+num[4]
        aver_num00=aver_num00+num[2]
        aver_num_mu=aver_num_mu+num[0]
        aver_num_re=aver_num+re+num[1]
        aver_mu=aver_mu+num[5]
        i+=1
    return (aver_num01,aver_num11,aver_num00,aver_num_mu.aver_num_re,aver_mu)

# change the array into a graph
def array_to_graph(array):
    g=nx.Graph()
    for i in range(12):
        key=array[i].keys()
        for j in key:
            g.add_edge(i,j)
    return g

# average the data with appointed index
def data_average(data,index):
    arg=np.argsort(index)
    index=index[arg]
    data=data[arg]
    newdata=[]
    newindex=[]
    N=len(data)
    ind=index[0]
    mid1=0
    mid2=0
    num=0
    i=0
    while i<N:
        if index[i]==ind:
            mid1+=index[i]
            mid2+=data[i]
            i+=1
            num+=1
        elif index[i] !=ind:
            ind=index[i]
            newdata.append(float(mid2)/num)
            newindex.append(float(mid1)/num)
            num=0
            mid1=0
            mid2=0
        if i==N:
            newdata.append(float(mid2)/num)
            newindex.append(float(mid1)/num)
    return (newdata,newindex)


# record the first time that node changes into mutants in the evolving process
def BD_first_time(g,r,nbunch,N):
    g=add_dynamic_attributes(g,nbunch)
    g=add_first_time(g,nbunch,N)
    i=0
    while i<N:
        i+=1
        birth=BD_selected_birth(g,r)
        death=BD_random_death(g,birth)
        g=update_type(g,death[0],birth)
        if g[birth]['type']==1:
            if g[death[0]]['first_time']!=N:
                g[death[0]]['first_time']=i
        test=test_nodes_type(g)
        if test==0 or test==1: break
    return (g,test)         


# average the first time for M simulations
def BD_average_first_time(g,r,nbunch,N,M):
    k=0
    a={}
    for d in g.nodes():
        a[d]=0
    nx.set_node_attributes(g,'average_first_time',a)
    while k<M:
        rst=BD_first_time(g,r,nbunch,N)
        g1=rst[0]
        if rst[1]==1:
            for d in g.nodes():
                g[d]['average_first_time']=g[d]['average_first_time']+g1[d]['first_time']
            k+=1
    for d in g.nodes():
        g[d]['average_first_time']=g[d]['average_first_time']/float(M)
    return g


# Compute the averaged fraction of 0-1edges, 0-0edges and 1-1edges by mean field method
def edges_vs_mutants(g,M):
    size=g.order()
    nodes=g.nodes()
    zo_edges=np.zeros(size)
    oo_edges=np.zeros(size)
    zz_edges=np.zeros(size)
    mutants=[]
    for k in range(size):
        i=k+1
        mutants.append(i)
        m=0
        while m<M:
            nbunch=random.sample(nodes,i)
            g=add_dynamic_attributes(g,nbunch)
            result=zero_one_edges(g)
            zo_edges[k]=zo_edges[k]+result[0]
            oo_edges[k]=oo_edges[k]+result[1]
            zz_edges[k]=zz_edges[k]+result[2]
            m+=1
    zo_edges=zo_edges/float(M)
    oo_edges=oo_edges/float(M)
    zz_edges=zz_edges/float(M)
    return (zo_edges,oo_edges,zz_edges,mutants)    
