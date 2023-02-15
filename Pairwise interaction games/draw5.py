import matplotlib.pyplot as plt
import numpy as np

BD_heat=np.load('heat0.npy')
BD_deg=np.load('deg0.npy')
BD_07=np.load('fix_prob07.npy')
BD_11=np.load('fix_prob11.npy')
BD_15=np.load('fix_prob15.npy')
BD_19=np.load('fix_prob19.npy')
agr=np.argsort(BD_heat)
BD_heat=BD_heat[agr]
BD_deg=BD_deg[agr]
BD_07=BD_07[agr]
BD_11=BD_11[agr]
BD_15=BD_15[agr]
BD_19=BD_19[agr]

plt.figure(0)

ax1=plt.subplot(311)
plt.plot(BD_heat,'ms')
plt.ylabel(r'$H_t(G)$',fontweight='heavy',fontsize=13)
plt.text(4,0.6,'(a)',fontweight='heavy',fontsize=13)

ax2=plt.subplot(312)
plt.plot(BD_19,'ro')
plt.ylabel(r'$\rho_1(1.9)$',fontweight='heavy',fontsize=13)
plt.text(4,0.5,'(b)',fontweight='heavy',fontsize=13)

ax3=plt.subplot(313)
plt.plot(BD_15,'b>')
plt.ylabel(r'$\rho_1(1.5)$',fontweight='heavy',fontsize=13)
plt.text(4,0.365,'(c)',fontweight='heavy',fontsize=13)

plt.xlabel(r'$G$',fontweight='heavy',fontsize=13)

plt.figure(1)

ax4=plt.subplot(311)
plt.plot(BD_11,'g^')
plt.ylabel(r'$\rho_1(1.1)$',fontweight='heavy',fontsize=13)
plt.text(4,0.144,'(d)',fontweight='heavy',fontsize=13)

ax5=plt.subplot(312)
plt.plot(BD_07,'c<')
plt.ylabel(r'$\rho_1(0.7)$',fontweight='heavy',fontsize=13)
plt.text(4,0.0065,'(e)',fontweight='heavy',fontsize=13)

ax6=plt.subplot(313)
plt.plot(BD_deg,'ko')
plt.ylabel(r'$H_d(G)$',fontweight='heavy',fontsize=13)
plt.text(4,7,'(f)',fontweight='heavy',fontsize=13)

plt.xlabel(r'$G$',fontweight='heavy',fontsize=13)

plt.show()
