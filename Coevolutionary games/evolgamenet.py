import networkx as nx
import numpy as np
import random as rd
import evolDynamic as ed

# calculate the final state
def evol_game(g,c,w,u,N):
    nodes=nx.nodes(g)
    a=g.size()/g.order()
    s=g.order()
    nbunch=rd.sample(nodes,s/2)
    g=ed.add_dynamic_attributes(g,nbunch)
    i=0
    result=[]
    while i<N:
            g=ed.calculate_fitness(g,[[1,0],[1+c,c]],w)
            nodes=nx.nodes(g)
            death=rd.sample(nodes,1)
            g.remove_node(death[0])
            birth=ed.BD_selected_birth_game(g)
            g.add_node(i)
            if g.degree()[birth]<=a:
                nbunch=g.neighbors(birth)
            else:
                nbunch=rd.sample(g.neighbors(birth),a)       
            for d in nbunch:
                g.add_edge(i,d)
            g.add_edge(i,birth)
            p=rd.random()
            if p>=u:
                g.node[i]['type']=g.node[birth]['type']
            else:
                q=rd.random()
                if q>=0.5: g.node[i]['type']=0
                else: g.node[i]['type']=1
            cooper=ed.test_nodes_type(g)
            result.append(cooper)
            i+=1
    eout=[result,g]
    return eout

# calculate the fixation probability of cooperators
def fix_evol_game(g,c,w,u,N,M):
    k=0
    aver=0
    while k<M:
        g1=g.copy()
        fin=evol_game(g1,c,w,u,N)
        aver=aver+fin
        k+=1
    aver=aver/float(M)
    return aver

# calculate the fixation probability for different c
def fix_evol_gamec(g,w,u,N,M):
    aver=[]
    for c in range(11):
        c1=c/float(10)
        a=fix_evol_game(g,c1,w,u,N,M)
        aver.append(a)
    return aver
    

# calculate the fixation probability for different w
def fix_evol_gamew(g,c,u,N,M):
    aver=[]
    for w in range(11):
        w1=w/float(10)
        a=fix_evol_game(g,c,w1,u,N,M)
        aver.append(a)
    return aver

# calculate the fixation probability for different u
def fix_evol_gameu(g,c,w,N,M):
    aver=[]
    u=[]
    u0=0
    while u0<0.2:
        a=fix_evol_game(g,c,w,u0,N,M)
        aver.append(a)
        u.append(u0)
        u0=u0+0.02
    return (aver,u)


# calculate the final state of precious model
def pre_evol_game(g,c,w,u,N):
    nodes=nx.nodes(g)
    s=g.order()
    nbunch=rd.sample(nodes,s/2)
    g=ed.add_dynamic_attributes(g,nbunch)
    i=0
    result=[]
    while i<N:
            g=ed.calculate_fitness(g,[[1,0],[1+c,c]],w)
            nodes=nx.nodes(g)
            death=rd.sample(nodes,1)
            birth=ed.BD_selected_birth_game(g)
            p=rd.random()
            if p>=u:
                g.node[death[0]]['type']=g.node[birth]['type']
            else:
                q=rd.random()
                if q>=0.5: g.node[death[0]]['type']=0
                else: g.node[death[0]]['type']=1
            i+=1
            cooper=ed.test_nodes_type(g)
            result.append(cooper)
    return result

# calculate the fixation probability of cooperators
def pre_fix_evol_game(g,c,w,u,N,M):
    k=0
    aver=0
    while k<M:
        g1=g.copy()
        fin=pre_evol_game(g1,c,w,u,N)
        aver=aver+fin
        k+=1
    aver=aver/float(M)
    return aver

# calculate the fixation probability for different c
def pre_fix_evol_gamec(g,w,u,N,M):
    aver=[]
    for c in range(11):
        c1=c/float(10)
        a=pre_fix_evol_game(g,c1,w,u,N,M)
        aver.append(a)
    return aver
    

# calculate the fixation probability for different w
def pre_fix_evol_gamew(g,c,u,N,M):
    aver=[]
    for w in range(11):
        w1=w/float(10)
        a=pre_fix_evol_game(g,c,w1,u,N,M)
        aver.append(a)
    return aver

# calculate the fixation probability for different u
def pre_fix_evol_gameu(g,c,w,N,M):
    aver=[]
    u=[]
    u0=0
    while u0<0.2:
        a=pre_fix_evol_game(g,c,w,u0,N,M)
        aver.append(a)
        u.append(u0)
        u0=u0+0.02
    return (aver,u)

# calculate the average payoff of the population
def payoff_evol_game(g,c,w,u,N,M):
    k=0
    aver=evol_game(g,c,w,u,N)
    aver=np.array(aver)
    while k<M-1:
        g1=g.copy()
        fin=evol_game(g1,c,w,u,N)
        fin=np.array(fin)
        aver=aver+fin
        k+=1
    aver=aver/float(M)
    return aver

# calculate the average payoff of the population
def payoff_pre_evol_game(g,c,w,u,N,M):
    k=0
    aver=pre_evol_game(g,c,w,u,N)
    aver=np.array(aver)
    while k<M-1:
        g1=g.copy()
        fin=pre_evol_game(g1,c,w,u,N)
        fin=np.array(fin)
        aver=aver+fin
        k+=1
    aver=aver/float(M)
    return aver

# calculate the final state for the case of random link 
def evol_game_sec(g,c,w,u,p,N):
    nodes=nx.nodes(g)
    s=g.order()
    nbunch=rd.sample(nodes,s/2)
    g=ed.add_dynamic_attributes(g,nbunch)
    i=0
    result=[]
    while i<N:
            g=ed.calculate_fitness(g,[[1,0],[1+c,c]],w)
            nodes=nx.nodes(g)
            death=rd.sample(nodes,1)
            g.remove_node(death[0])
            birth=ed.BD_selected_birth_game(g)
            g.add_node(i)
            if g.degree()[birth]<=3:
                nbunch=g.neighbors(birth)
            else:
                nbunch=rd.sample(g.neighbors(birth),3)       
            for d in nbunch:
                rd1=rd.random()
                if rd1>p:
                    g.add_edge(i,d)
                else:
                    nodes=nx.nodes(g)
                    node=rd.sample(nodes,1)
                    g.add_edge(i,node[0])
            rd1=rd.random()
            if rd1>p:
                g.add_edge(i,birth)
            else:
                nodes=nx.nodes(g)
                node=rd.sample(nodes,1)
                g.add_edge(i,node[0])
            rd2=rd.random()
            if rd2>=u:
                g.node[i]['type']=g.node[birth]['type']
            else:
                q=rd.random()
                if q>=0.5: g.node[i]['type']=0
                else: g.node[i]['type']=1
            cooper=ed.test_nodes_type(g)
            result.append(cooper)
            i+=1
    eout=[result,g]
    return eout

# calculate the final state with death happens with probability inversely proportional to fitness
def evol_game_trd(g,c,w,u,N):
    nodes=nx.nodes(g)
    s=g.order()
    nbunch=rd.sample(nodes,s/2)
    g=ed.add_dynamic_attributes(g,nbunch)
    i=0
    result=[]
    while i<N:
            g=ed.calculate_fitness(g,[[1,0],[1+c,c]],w)
            nodes=nx.nodes(g)
            death=ed.BD_selected_death_game(g)
            g.remove_node(death)
            birth=ed.BD_selected_birth_game(g)
            g.add_node(i)
            if g.degree()[birth]<=3:
                nbunch=g.neighbors(birth)
            else:
                nbunch=rd.sample(g.neighbors(birth),3)       
            for d in nbunch:
                g.add_edge(i,d)
            g.add_edge(i,birth)
            p=rd.random()
            if p>=u:
                g.node[i]['type']=g.node[birth]['type']
            else:
                q=rd.random()
                if q>=0.5: g.node[i]['type']=0
                else: g.node[i]['type']=1
            cooper=ed.test_nodes_type(g)
            result.append(cooper)
            i+=1
    eout=[result,g]
    return eout

# calculate the final state with death happens with probability inversely proportional to fitness
def evol_game_fth(g,c,w,u,p,N):
    nodes=nx.nodes(g)
    s=g.order()
    nbunch=rd.sample(nodes,s/2)
    g=ed.add_dynamic_attributes(g,nbunch)
    i=0
    result=[]
    while i<N:
            g=ed.calculate_fitness(g,[[1,0],[1+c,c]],w)
            nodes=nx.nodes(g)
            death=ed.BD_selected_death_game(g)
            g.remove_node(death)
            birth=ed.BD_selected_birth_game(g)
            g.add_node(i)
            if g.degree()[birth]<=3:
                nbunch=g.neighbors(birth)
            else:
                nbunch=rd.sample(g.neighbors(birth),3)       
            for d in nbunch:
                rd1=rd.random()
                if rd1>p:
                    g.add_edge(i,d)
                else:
                    nodes=nx.nodes(g)
                    node=rd.sample(nodes,1)
                    g.add_edge(i,node[0])
            rd1=rd.random()
            if rd1>p:
                g.add_edge(i,birth)
            else:
                nodes=nx.nodes(g)
                node=rd.sample(nodes,1)
                g.add_edge(i,node[0])
            rd2=rd.random()
            if rd2>=u:
                g.node[i]['type']=g.node[birth]['type']
            else:
                q=rd.random()
                if q>=0.5: g.node[i]['type']=0
                else: g.node[i]['type']=1
            cooper=ed.test_nodes_type(g)
            result.append(cooper)
            i+=1
    eout=[result,g]
    return eout

# calculate the final state of random geometric graphs for public goods games
def evol_game_fiv(S,r,rwd,c,w,u,N):
    g=nx.random_geometric_graph(S,r)
    nodes=nx.nodes(g)
    nbunch=rd.sample(nodes,S/2)
    g=ed.add_dynamic_attributes(g,nbunch)
    i=0
    result=[]
    while i<N:
            g=ed.calculate_fitness_pub(g,rwd,c,w)
            nodes=nx.nodes(g)
            death=rd.sample(nodes,1)
            g.remove_node(death[0])
            birth=ed.BD_selected_birth_game(g)
            pos_sel=g.node[birth]['pos']
            exp_x=np.random.randn()
            exp_y=np.random.randn()
            inv_x=4*r /np.pi*np.arctan(exp_x)
            inv_y=4*r/np.pi*np.arctan(exp_y)
            pos_i=[inv_x+pos_sel[0],inv_y+pos_sel[1]]
            for k in range(2):
                if pos_i[k]>1:
                    pos_i[k]=1
                elif pos_i[k]<0:
                    pos_i[k]=0
            nodes=g.nodes(data=True)
            g.add_node(S+i,attr_dict={'pos':pos_i})
            while nodes:
                v,dv = nodes.pop()
                pv = dv['pos']
                d = sum(((a-b)**2 for a,b in zip(pv,pos_i)))
                if d <= r**2:
                    g.add_edge(v,S+i)
            rd2=rd.random()
            if rd2>=u:
                g.node[S+i]['type']=g.node[birth]['type']
            else:
                q=rd.random()
                if q>=0.5: g.node[S+i]['type']=0
                else: g.node[S+i]['type']=1
            cooper=ed.test_nodes_type(g)
            result.append(cooper)
            i+=1
    eout=[result,g]
    return eout

# calculate the final state for the case of random link with birth and death rate

# r is the birth-to-death rate ratio
def evol_game_six(g,c,w,u,p,r,N):
    nodes=nx.nodes(g)
    s=g.order()
    nbunch=rd.sample(nodes,s/2)
    g=ed.add_dynamic_attributes(g,nbunch)
    i=0
    result=[]
    while i<N:
            g=ed.calculate_fitness(g,[[1,0],[1+c,c]],w)
            nodes=nx.nodes(g)
            if i%r==0:
                death=rd.sample(nodes,1)
                g.remove_node(death[0])
            birth=ed.BD_selected_birth_game(g)
            g.add_node(i)
            if g.degree()[birth]<=3:
                nbunch=g.neighbors(birth)
            else:
                nbunch=rd.sample(g.neighbors(birth),3)       
            for d in nbunch:
                rd1=rd.random()
                if rd1>p:
                    g.add_edge(i,d)
                else:
                    nodes=nx.nodes(g)
                    node=rd.sample(nodes,1)
                    g.add_edge(i,node[0])
            rd1=rd.random()
            if rd1>p:
                g.add_edge(i,birth)
            else:
                nodes=nx.nodes(g)
                node=rd.sample(nodes,1)
                g.add_edge(i,node[0])
            rd2=rd.random()
            if rd2>=u:
                g.node[i]['type']=g.node[birth]['type']
            else:
                q=rd.random()
                if q>=0.5: g.node[i]['type']=0
                else: g.node[i]['type']=1
            cooper=ed.test_nodes_type(g)
            result.append(cooper)
            i+=1
    eout=[result,g]
    return eout



# calculate the final state for the case of random link 
def evol_game_sev(g,c,w,u,p,N):
    nodes=nx.nodes(g)
    s=g.order()
    nbunch=rd.sample(nodes,s/2)
    g=ed.add_dynamic_attributes(g,nbunch)
    i=0
    result=[]
    while i<N:
            g=ed.calculate_fitness(g,[[1,0],[1+c,c]],w)
            nodes=nx.nodes(g)
            death=rd.sample(nodes,1)
            g.remove_node(death[0])
            birth=ed.BD_selected_birth_game(g)
            g.add_node(i+s)
            nbunch=g.neighbors(birth)     
            for d in nbunch:
                rd1=rd.random()
                if rd1>p:
                    g.add_edge(i,d)
                else:
                    nodes=nx.nodes(g)
                    node=rd.sample(nodes,1)
                    g.add_edge(i,node[0])
            rd1=rd.random()
            if rd1>p:
                g.add_edge(i,birth)
            else:
                nodes=nx.nodes(g)
                node=rd.sample(nodes,1)
                g.add_edge(i,node[0])
            rd2=rd.random()
            if rd2>=u:
                g.node[i]['type']=g.node[birth]['type']
            else:
                q=rd.random()
                if q>=0.5: g.node[i]['type']=0
                else: g.node[i]['type']=1
            cooper=ed.test_nodes_type(g)
            result.append(cooper)
            i+=1
    eout=[result,g]
    return eout
