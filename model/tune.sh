#!/bin/bash
echo "hello"

hiddensize=( 20 30 40 60 ) 
model=( model1_fast_NB model1_fast)
weightdecay=( 0 0.1 0.05 0.01 0.008 0.005 0.003 0.001)
learningrate=( 0.001 0.004 0.006 0.008 0.009 0.0001)
optim=( rmsprop)

for w in ${weightdecay[@]}; do
	for h in ${hiddensize[@]}; do
			for m in ${model[@]}; do 			
				for l in ${learningrate[@]}; do
					for o in ${optim[@]}; do
	                			th train.lua -model $m -weight_decay $w -learning_rate $l -hidden_size $h -optim $o -batch_size 32 >log.txt 
					done
				done
			done
	done
done



