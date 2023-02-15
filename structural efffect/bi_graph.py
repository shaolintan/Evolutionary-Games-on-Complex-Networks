import numpy as np
import networkx as nx

#--numerical solution of the fixation probability for complete bipartite graph

#--birth-death process

#--calculate the translation matrix
def BD_tran_matrix(M,N,r):
    S=(M+1)*(N+1)
    P=np.zeros([S,S])
    P[0][0]=1
    P[S-1][S-1]=1 # boundary conditions
    for k in range(M):
        i=k+1
        j=0
        W=r*(i+j)+M-i+N-j
        s1=(N+1)*i+j
        s2=(N+1)*(i-1)+j
        s3=(N+1)*i+j+1
        P[s1][s2]=float(N-j)/W*float(i)/M
        P[s1][s3]=float(r*i)/W*float(N-j)/N
        P[s1][s1]=1-P[s1][s2]-P[s1][s3]
    for k in range(M-1):
        i=k+1
        j=N
        W=r*(i+j)+M-i+N-j
        s1=(N+1)*i+j
        s2=(N+1)*(i+1)+j
        s3=(N+1)*i+j-1
        P[s1][s2]=float(r*j)/W*float(M-i)/M
        P[s1][s3]=float(M-i)/W*float(j)/N
        P[s1][s1]=1-P[s1][s2]-P[s1][s3]
    for k in range(N):
        j=k+1
        i=0
        W=r*(i+j)+M-i+N-j
        s1=(N+1)*i+j
        s2=(N+1)*(i+1)+j
        s3=(N+1)*i+j-1
        P[s1][s2]=float(r*j)/W*float(M-i)/M
        P[s1][s3]=float(M-i)/W*float(j)/N
        P[s1][s1]=1-P[s1][s2]-P[s1][s3]
    for k in range(N-1):
        j=k+1
        i=M
        W=r*(i+j)+M-i+N-j
        s1=(N+1)*i+j
        s2=(N+1)*(i-1)+j
        s3=(N+1)*i+j+1
        P[s1][s2]=float(N-j)/W*float(i)/M
        P[s1][s3]=float(r*i)/W*float(N-j)/N
        P[s1][s1]=1-P[s1][s2]-P[s1][s3]
    for k1 in range(M-1):
        for k2 in range(N-1):
            i=k1+1
            j=k2+1
            W=r*(i+j)+M-i+N-j
            s1=(N+1)*i+j
            s2=(N+1)*(i-1)+j
            s3=(N+1)*i+j+1
            s4=(N+1)*(i+1)+j
            s5=(N+1)*i+j-1
            P[s1][s2]=float(N-j)/W*float(i)/M
            P[s1][s3]=float(r*i)/W*float(N-j)/N
            P[s1][s4]=float(r*j)/W*float(M-i)/M
            P[s1][s5]=float(M-i)/W*float(j)/N
            P[s1][s1]=1-P[s1][s2]-P[s1][s3]-P[s1][s4]-P[s1][s5]
    return P

# calculate the fixation probability of a random mutant
def BD_fix_prob(M,N,r):
    S=(M+1)*(N+1)
    P=BD_tran_matrix(M,N,r)
    A=P[1:S-1,(0,S-1)]
    Q=P[1:S-1,1:S-1]
    I=np.eye(S-2,S-2)
    B=np.linalg.inv(I-Q)
    C=np.dot(B,A)
    fix=(N*C[0][1]+M*C[N][1])/float(M+N)
    return fix
    
# output the fixation probability and heat heterogeneity for a set of M
def BD_output(N):
    rho0_9=[]
    rho1_1=[]
    heat_heter=[]
    deg_heter=[]
    for i in range(N):
        m1=i+1
        m2=2*N-m1
        rho0_9.append(BD_fix_prob(m1,m2,0.9))
        rho1_1.append(BD_fix_prob(m1,m2,1.1))
        heat_heter.append(float((m1-m2)**2)/(m1*m2))
        deg_heter.append(m1*m2*(m1-m2)**2/float((m1+m2)**2))
    return (rho0_9,rho1_1,heat_heter,deg_heter)
        
#-----------------------------------------------------------------------------     

#--death-birth process

#--calculate the translation matrix
def DB_tran_matrix(M,N,r):
    S=(M+1)*(N+1)
    P=np.zeros([S,S])
    P[0][0]=1
    P[S-1][S-1]=1 # boundary conditions
    for k in range(M):
        i=k+1
        j=0
        W=M+N
        s1=(N+1)*i+j
        s2=(N+1)*(i-1)+j
        s3=(N+1)*i+j+1
        P[s1][s2]=float(i)/W*float(N-j)/(N-j+r*j)
        P[s1][s3]=float(N-j)/W*float(r*i)/(M-i+r*i)
        P[s1][s1]=1-P[s1][s2]-P[s1][s3]
    for k in range(M-1):
        i=k+1
        j=N
        W=M+N
        s1=(N+1)*i+j
        s2=(N+1)*(i+1)+j
        s3=(N+1)*i+j-1
        P[s1][s2]=float(M-i)/W*float(r*j)/(N-j+r*j)
        P[s1][s3]=float(j)/W*float(M-i)/(M-i+r*i)
        P[s1][s1]=1-P[s1][s2]-P[s1][s3]
    for k in range(N):
        j=k+1
        i=0
        W=M+N
        s1=(N+1)*i+j
        s2=(N+1)*(i+1)+j
        s3=(N+1)*i+j-1
        P[s1][s2]=float(M-i)/W*float(r*j)/(N-j+r*j)
        P[s1][s3]=float(j)/W*float(M-i)/(M-i+r*i)
        P[s1][s1]=1-P[s1][s2]-P[s1][s3]
    for k in range(N-1):
        j=k+1
        i=M
        W=M+N
        s1=(N+1)*i+j
        s2=(N+1)*(i-1)+j
        s3=(N+1)*i+j+1
        P[s1][s2]=float(i)/W*float(N-j)/(N-j+r*j)
        P[s1][s3]=float(N-j)/W*float(r*i)/(M-i+r*i)
        P[s1][s1]=1-P[s1][s2]-P[s1][s3]
    for k1 in range(M-1):
        for k2 in range(N-1):
            i=k1+1
            j=k2+1
            W=r*(i+j)+M-i+N-j
            s1=(N+1)*i+j
            s2=(N+1)*(i-1)+j
            s3=(N+1)*i+j+1
            s4=(N+1)*(i+1)+j
            s5=(N+1)*i+j-1
            P[s1][s2]=float(i)/W*float(N-j)/(N-j+r*j)
            P[s1][s3]=float(N-j)/W*float(r*i)/(M-i+r*i)
            P[s1][s4]=float(M-i)/W*float(r*j)/(N-j+r*j)
            P[s1][s5]=float(j)/W*float(M-i)/(M-i+r*i)
            P[s1][s1]=1-P[s1][s2]-P[s1][s3]-P[s1][s4]-P[s1][s5]
    return P

# calculate the fixation probability of a random mutant
def DB_fix_prob(M,N,r):
    S=(M+1)*(N+1)
    P=DB_tran_matrix(M,N,r)
    A=P[1:S-1,(0,S-1)]
    Q=P[1:S-1,1:S-1]
    I=np.eye(S-2,S-2)
    B=np.linalg.inv(I-Q)
    C=np.dot(B,A)
    fix=(N*C[0][1]+M*C[N][1])/float(M+N)
    return fix

# output the fixation probability and heat heterogeneity for a set of M
def DB_output(N):
    rho0_9=[]
    rho1_1=[]
    heat_heter=[]
    deg_heter=[]
    for i in range(N):
        m1=i+1
        m2=2*N-m1
        rho0_9.append(DB_fix_prob(m1,m2,0.9))
        rho1_1.append(DB_fix_prob(m1,m2,1.1))
        heat_heter.append(float((m1-m2)**2)/(m1*m2))
        deg_heter.append(m1*m2*(m1-m2)**2/float((m1+m2)**2))
    return (rho0_9,rho1_1,heat_heter,deg_heter)



# output the fixation probability and heat hetergeneithy for a set of M for both
# BD and DB processes.
def output(N):
    BD_rho0_9=[]
    BD_rho1_1=[]
    DB_rho0_9=[]
    DB_rho1_1=[]
    heat_heter=[]
    deg_heter=[]
    for i in range(N):
        m1=i+1
        m2=2*N-m1
        BD_rho0_9.append(BD_fix_prob(m1,m2,0.9))
        BD_rho1_1.append(BD_fix_prob(m1,m2,1.1))
        DB_rho0_9.append(DB_fix_prob(m1,m2,0.9))
        DB_rho1_1.append(DB_fix_prob(m1,m2,1.1))
        heat_heter.append(float((m1-m2)**2)/(m1*m2))
        deg_heter.append(m1*m2*(m1-m2)**2/float((m1+m2)**2))
    return (heat_heter,BD_rho0_9,BD_rho1_1,DB_rho0_9,DB_rho1_1,deg_heter)




    

    
