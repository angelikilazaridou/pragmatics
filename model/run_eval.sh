#!/bin/bash

for f in `cat to_run_gr.txt` 
do
	th eval2.lua -batch_size 128 -game v2 -val_images_use 10000 -noise 0 -temperature 10 -model grounding/gr/$f -split test_hard
	python scripts/txt2html.py html/$f.txt html/$f"_hard.txt"
done

#th eval2.lua -batch_size 128 -game v2 -model grounding/cp_id_sender@sender_no_embeddings_g@v2_gs@2_t@10_v@1_l@fc_g@0_vocab@10_property@100_embR@150_hidden@50.cp.t7  -val_images_use 10000 -noise 0 -temperature 10 -split test_hard
