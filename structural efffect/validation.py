import networkx as nx
import bi_graph as bg
import evolDynamic as ed

#--validity of the numerical solution of the fixation probability for complete bipartite graphs

#--birth-death process

#--fixation probability of a random mutant with different fitness r
def BD_comp(M,N):
    fix_theo=[]
    fix_simu=[]
    fitness=[]
    r0=0.1
    g=nx.complete_bipartite_graph(M,N)
    for i in range(40):
        r=r0+0.1*i
        fix_theo_r=bg.BD_fix_prob(M,N,r)
        fix_simu_r=ed.BD_average_fixation(g,r,10000)
        fitness.append(r)
        fix_theo.append(fix_theo_r)
        fix_simu.append(fix_simu_r)
    return (fix_theo,fix_simu,fitness)
        
#--death-birth process

#--fixation probability of a random mutant with different fitness r
def DB_comp(M,N):
    fix_theo=[]
    fix_simu=[]
    fitness=[]
    r0=0.1
    g=nx.complete_bipartite_graph(M,N)
    for i in range(40):
        r=r0+0.1*i
        fix_theo_r=bg.DB_fix_prob(M,N,r)
        fix_simu_r=ed.DB_average_fixation(g,r,10000)
        fitness.append(r)
        fix_theo.append(fix_theo_r)
        fix_simu.append(fix_simu_r)
    return (fix_theo,fix_simu,fitness)
        
