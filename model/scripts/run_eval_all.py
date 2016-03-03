import os
import json
from subprocess import call

tune_dir = '/home/angeliki/git/pragmatics/model/tune_new_single/'
gpu = "0"

for fname in os.listdir(tune_dir):
	if not fname.endswith(".t7"):
		continue
	to_test = tune_dir+fname
	print "############################# "+to_test+" ################################"
	print "@@@@@@@   REFERIT STRATEGY 1   @@@@@@@@@@@"
	command = "th eval_referit.lua -model "+to_test+" -gpuid "+gpu+" -strategy "+"1"+" -exclude "+"0"
	os.system(command)
	print "@@@@@@@   REFERIT STRATEGY 3   @@@@@@@@@@@"
        command = "th eval_referit.lua -model "+to_test+" -gpuid "+gpu+" -strategy "+"3"+" -exclude "+"1"
        os.system(command)
	print "@@@@@@@   CARINA    @@@@@@@@@@@"
        command = "th eval_test.lua -model "+to_test+" -gpuid "+gpu
        os.system(command)

