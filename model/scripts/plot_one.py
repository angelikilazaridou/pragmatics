import numpy as np
from scipy.interpolate import spline,interp1d
from os import listdir
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys

f  = sys.argv[1]
dataset = sys.argv[2]
N = int(sys.argv[3])
pos = int(sys.argv[4]) 

x = range(N-1)
k = 100

xnew = np.linspace(min(x), max(x), k)

all_accs = []

with open(f,'r') as f:
	accs = []
	l = 0
	while l<N:
		line = f.readline()
		line = line.strip()
		if ' @ ' in line:
			if l != 0:
				accs.append(float(line.split('@')[pos].rstrip().strip()))
			l = l+1
	#smoothing values		
	all_accs =  interp1d(x,accs)(xnew)


print(len(all_accs))
#One plot per model type (h2)
a,  = plt.plot(xnew, all_accs,  linewidth=2)
plt.xlabel('# iterations', fontsize=20)
plt.ylabel('communication success', fontsize=20)
plt.savefig(dataset+'_ALL.pdf')
plt.clf()


