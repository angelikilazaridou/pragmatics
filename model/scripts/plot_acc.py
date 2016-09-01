import numpy as np
from scipy.interpolate import spline,interp1d
from os import listdir
import matplotlib.pyplot as plt

dir = '/home/angeliki/git/pragmatics/model/logs/'
files = '/home/angeliki/git/pragmatics/model/scripts/index.txt'
dataset_names = {'v1':'ReferIt','v2':'Objects','v3':'Shapes'}

dataset = 'v2'
feat_size = '300'
N = 3500

x = range(N)
k = 100

#color = ['r','g','b','r','g','b','r','g','b']
#ls = ['-','-','-','--','--','--',':',':',':']

xnew = np.linspace(min(x), max(x), k)

labels = []
to_plot = []

all_accs = {}
s = 0
with open(files,'r') as f1:
	for line in f1:
		fi = line.strip()
		with open(dir+fi,'r') as f:
			print fi

			attr = fi.split('_')[2]
			to_plot.append((fi,attr))

			accs = []
			l = 0
			while l<N:
				line = f.readline()
				line = line.strip()
				if ' @ ' in line:
					accs.append(float(line.split('@')[2].rstrip().strip()))
					l = l+1
			#smoothing values		
			all_accs[fi] =  interp1d(x,accs)(xnew)



#One plot per model type (h2)
for fi,attr in to_plot:
        fi
        a,  = plt.plot(xnew, all_accs[fi], label=attr, linewidth=2)
        labels.append(a)

plt.suptitle(dataset_names[dataset], fontsize=23)
plt.xlabel('# iterations', fontsize=20)
plt.ylabel('communication success', fontsize=20)
plt.legend(handles=labels,bbox_to_anchor=(1, 0.5))
plt.savefig(dataset+'_ALL.eps')
plt.clf()


