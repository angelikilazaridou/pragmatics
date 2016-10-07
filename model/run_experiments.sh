#!/bin/bash

game_size=( 2 4 )
layers=( "fc" )
grounding=( 0 0.5 )
viewpoints=( 1 )

for g in ${game_size[@]}; do
  for l in ${layers[@]}; do
    for gr in ${grounding[@]}; do
      for v in ${viewpoints[@]}; do
        th train.lua  -batch_size 32 -comm_game v2 -comm_feat_size 300 -vocab_size 100 -property_size 100 -comm_game_size ${g} -hidden_size 20 -id exp1 -gpuid -1 -temperature 10 -max_iters 50000 -gr_task v2 -gr_feat_size 300 -embedding_size_S 100 -embedding_size_R 100 -grounding ${gr} -save_checkpoint_every 1000 -comm_viewpoints ${v} -comm_layer ${l} > logs/log_g@${g}_l@${l}_gr@${gr}_v@${v}.txt &  
      done
    done
  done
done

