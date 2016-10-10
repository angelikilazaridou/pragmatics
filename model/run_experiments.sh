#!/bin/bash

game_size=( 2 )
layers=( "probs" "fc" )
grounding=( 0 )
viewpoints=( 1 0 )
vocab_size=( 10 30 100 450 )
sender=( "sender_simple" "sender_no_embeddings" "sender_conv")
embedding_size=( 50 100 300 )
noise=( 0 1 )

for g in ${game_size[@]}; do
  for l in ${layers[@]}; do
    for gr in ${grounding[@]}; do
      for v in ${viewpoints[@]}; do
	for w in ${vocab_size[@]}; do
	    for s in ${sender[@]}; do
		for n in ${noise[@]}; do
			for e in ${embedding_size[@]}; do
		        	th train.lua  -batch_size 32 -comm_game v2 -comm_feat_size 300 -vocab_size ${w} -property_size 100 -comm_game_size ${g} -hidden_size 20 -id exp1 -gpuid -1 -temperature 10 -max_iters 50000 -gr_task v2 -gr_feat_size 300 -embedding_size_S ${e}  -embedding_size_R ${e} -grounding ${gr} -comm_sender ${s} -save_checkpoint_every 1000 -comm_viewpoints ${v} -comm_layer ${l} -comm_noise ${n} > logs_fb/grounding/RL/log_sender@${s}_embS@${e}_embR${e}_V@${w}_@${g}_l@${l}_gr@${gr}_v@${v}_n@${n}.txt &  
			done
		done
	    done
	done
      done
    done
  done
done

