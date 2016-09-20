#!/bin/bash

attributes=( 2 40 60 80 100 120 )
games=( v1 )

for a in ${attributes[@]}; do
    for g in ${games[@]}; do
        th eval.lua -model conll/cp_id_g@${g}_h@20_d@0_f@300_a@${a}.cp.t7 -feat_size 300 -vocab_size ${a} -game_session ${g} -split test -val_images_use 1000 > log_g@${g}_h@20_d@0_f@300_a@${a}.txt 
    done
done

