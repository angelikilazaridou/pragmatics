import os
import json

tune_dir = '/home/angeliki/git/pragmatics/model/tune_new_single/'
best_f = ''
best_acc = -1
best_iter = 0
for fname in os.listdir(tune_dir):
	if not fname.endswith(".json"):
		continue
	with open(tune_dir+fname,'r') as f:
		data = json.load(f)
		acc = sorted(data['val_acc_history'].items(), key=lambda x:x[1], reverse=True)
		print fname, acc[0]
		if acc[0][1]>best_acc:
			best_acc = acc[0][1]
			best_f = fname
			best_iter = acc[0]
print best_iter, best_acc, best_f
		
		

