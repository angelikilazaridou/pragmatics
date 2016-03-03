import random
import shutil


import re

ansi_escape = re.compile(r'\x1b[^m]*m')
results_file = '../model_predictions_s2.txt'
res_dir = 'google_dir_s2/'
items = {}
to_keep = 200

with open(results_file,'r') as f:
	image_line = -1
	for line in f.readlines():
		#grab attribute
		line = ansi_escape.sub('', line)
		line = line.strip()
		if image_line==0:
			attr = line
			if not el[2] in items:
				items[el[2]] = []
			items[el[2]].append((el[0],el[1],attr,el[2]))
			image_line = -1
			continue
		if line.startswith("#"):
			image_line = 1
			continue
		if image_line ==1:
			image_line = 0
			el = line.split(' ')
			continue
all_attrs = items.keys()
all_items = []
for k in items:
	all_items.extend(items[k])
print(len(all_items))
random.shuffle(all_items)
print(all_items[0])
for i in range(to_keep):
	print "Processing ",i
	with open(res_dir+'image1_'+str(i)+'.txt','w') as f:
		print len(all_items)
		f.write(all_items[i][0])
		shutil.copyfile("/home/angeliki/bla/"+all_items[i][0]+".jpg","google_dir/"+all_items[i][0]+".jpg")
	with open(res_dir+'image2_'+str(i)+'.txt','w') as f:
                f.write(all_items[i][1])
		shutil.copyfile("/home/angeliki/bla/"+all_items[i][1]+".jpg","google_dir/"+all_items[i][1]+".jpg")
	#with open(res_dir+'attrs_'+str(i)+'.txt','w') as f:
	#	att = random.randint(0,len(all_attrs)-1)	
	#	if random.randint(0,2) == 1:
        #        	f.write(all_items[i][2]+"\n"+all_attrs[att])
	#	else:
	#		f.write(all_attrs[att]+"\n"+all_items[i][2])
	with open(res_dir+'g_attrs_'+str(i)+'.txt','w') as f:
                f.write(all_items[i][2]+"\n")


