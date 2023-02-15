import matplotlib.pyplot as plt
import numpy as np

DB_heat=np.load('heat_1.npy')
DB_deg=np.load('deg_1.npy')
DB_07=np.load('fix_prob0_7.npy')
DB_11=np.load('fix_prob1_1.npy')
DB_15=np.load('fix_prob1_5.npy')
DB_19=np.load('fix_prob1_9.npy')
agr=np.argsort(DB_heat)
DB_heat=DB_heat[agr]
DB_deg=DB_deg[agr]
DB_07=DB_07[agr]
DB_11=DB_11[agr]
DB_15=DB_15[agr]
DB_19=DB_19[agr]

plt.figure(0)

ax1=plt.subplot(311)
plt.plot(DB_heat,'ms')
plt.ylabel(r'$H_t(G)$',fontweight='heavy',fontsize=13)
plt.text(4,0.6,'(a)',fontweight='heavy',fontsize=13)

ax2=plt.subplot(312)
plt.plot(DB_19,'ro')
plt.ylabel(r'$\rho_1(1.9)$',fontweight='heavy',fontsize=13)
plt.text(4,0.5,'(b)',fontweight='heavy',fontsize=13)

ax3=plt.subplot(313)
plt.plot(DB_15,'b>')
plt.ylabel(r'$\rho_1(1.5)$',fontweight='heavy',fontsize=13)
plt.text(4,0.365,'(c)',fontweight='heavy',fontsize=13)

plt.xlabel(r'$G$',fontweight='heavy',fontsize=13)

plt.figure(1)

ax4=plt.subplot(311)
plt.plot(DB_11,'g^')
plt.ylabel(r'$\rho_1(1.1)$',fontweight='heavy',fontsize=13)
plt.text(4,0.144,'(d)',fontweight='heavy',fontsize=13)

ax5=plt.subplot(312)
plt.plot(DB_07,'c<')
plt.ylabel(r'$\rho_1(0.7)$',fontweight='heavy',fontsize=13)
plt.text(4,0.0065,'(e)',fontweight='heavy',fontsize=13)

ax6=plt.subplot(313)
plt.plot(DB_deg,'ko')
plt.ylabel(r'$H_d(G)$',fontweight='heavy',fontsize=13)
plt.text(4,7,'(f)',fontweight='heavy',fontsize=13)

plt.xlabel(r'$G$',fontweight='heavy',fontsize=13)

plt.show()
