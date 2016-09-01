import numpy as np
from scipy.interpolate import spline,interp1d
from os import listdir
import matplotlib.pyplot as plt

dir = '/home/angeliki/git/pragmatics/model/logs/'
files = '/home/angeliki/git/pragmatics/model/logs/index.txt'

dataset = 'v1'
N = 3500
x = range(N)
k = 100
color = ['r','g','b','r','g','b','r','g','b']
ls = ['-','-','-','--','--','--',':',':',':']

xnew = np.linspace(min(x), max(x), k)

labels = []

all_accs = {}
s = 0
for fi in sorted(listdir(dir)):
	if not fi.endswith('.txt'):
		continue
	with open(dir+fi,'r') as f:
		print fi
		accs = []
		l = 0
		while l<N:
			line = f.readline()
			line = line.strip()
			if ' @ ' in line:
				accs.append(float(line.split('@')[2].rstrip().strip()))
				l = l+1
		#smoothing values		
		print fi
		all_accs[fi] =  interp1d(x,accs)(xnew)


#One plot per attribute size (100)
to_plot =[dataset+'_300_100.txt', dataset+'_300_100_h2.txt',dataset+'_300_100_d0.txt']
labels = []
for fi in to_plot:
	fi
	a,  = plt.plot(xnew, all_accs[fi], label=fi)
	labels.append(a)

plt.legend(handles=labels,bbox_to_anchor=(1, 0.5))
plt.savefig(dataset+'_300_100.png')
plt.clf()

#One plot per attribute size (2)
to_plot =[dataset+'_300_2.txt', dataset+'_300_2_h2.txt',dataset+'_300_2_d0.txt']  
labels = []
for fi in to_plot:
        fi
        a,  = plt.plot(xnew, all_accs[fi], label=fi)
        labels.append(a)

plt.legend(handles=labels,bbox_to_anchor=(1, 0.5))
plt.savefig(dataset+'_300_2.png')
plt.clf()

#One plot per attribute size (500)
to_plot =[dataset+'_300_500.txt', dataset+'_300_500_h2.txt',dataset+'_300_500_d0.txt']  
labels = []
for fi in to_plot:
        fi
        a,  = plt.plot(xnew, all_accs[fi], label=fi)
        labels.append(a)

plt.legend(handles=labels,bbox_to_anchor=(1, 0.5))
plt.savefig(dataset+'_300_500.png')
plt.clf()

print "here"
#One plot per model type (plain)
to_plot =[2,40,50,60,70,90,100,110,120]
labels = []
for x in to_plot:
        fi = dataset+'_300_'+str(x)+'.txt'
	print fi
        a,  = plt.plot(xnew, all_accs[fi], label=fi)
        labels.append(a)

plt.legend(handles=labels,bbox_to_anchor=(1, 0.5))
plt.savefig(dataset+'_300_default.png')
plt.clf()

print "After"

#One plot per model type (h2)
to_plot =[dataset+'_300_2_h2.txt', dataset+'_300_100_h2.txt',dataset+'_300_500_h2.txt']
labels = []
for fi in to_plot:
        fi
        a,  = plt.plot(xnew, all_accs[fi], label=fi)
        labels.append(a)

plt.legend(handles=labels,bbox_to_anchor=(1, 0.5))
plt.savefig(dataset+'_300_h2.png')
plt.clf()

#One plot per model type (d0)
to_plot =[dataset+'_300_2_d0.txt', dataset+'_300_100_d0.txt',dataset+'_300_500_d0.txt']
labels = []
for fi in to_plot:
        fi
        a,  = plt.plot(xnew, all_accs[fi], label=fi)
        labels.append(a)

plt.legend(handles=labels,bbox_to_anchor=(1, 0.5))
plt.savefig(dataset+'_300_d0.png')
plt.clf()


