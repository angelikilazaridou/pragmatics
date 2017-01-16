import numpy as np
from scipy.interpolate import spline,interp1d
from os import listdir
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys

logs = "to_run.txt"

result_dir = "prob_plots/"
logs_dir = "../logs_fb/grounding/RL/"

N = 100
pos = 1

k = 10


	

def plot_file(logs_dir, txt, smooth=False):
	x = []
	all_accs = []

	with open(logs_dir+txt,'r') as f:
		accs = []
		l = 0
		line = "blabla"
		while line:
			line = f.readline()
			line = line.strip()
			if ' @ ' in line:
				if l != 0:
					if len(x) > 0 and float(line.split('@')[0].rstrip().strip()) < x[-1]:		
						continue
					accs.append(float(line.split('@')[pos].rstrip().strip()))
					x.append(int(line.split('@')[0].strip()))
			
				l = l+1
	if smooth:
		xnew = np.linspace(min(x), max(x), k)
		#smoothing values		
		all_accs =  interp1d(x,accs)(xnew)
	else:
		xnew = x
		all_accs = accs
	
	if len(xnew)>5:
		#One plot per model type (h2)
		a,  = plt.plot(xnew, all_accs,  linewidth=2)
		plt.xlabel('# iterations', fontsize=20)
		plt.ylabel('communication success', fontsize=20)
		plt.savefig(result_dir+txt+'_ALL.png')
		plt.clf()
	return xnew, accs
with open(logs) as l:
	for line in l:
		line = line.strip()
		print "Working with file ",line
		x,acc = plot_file(logs_dir,line)
