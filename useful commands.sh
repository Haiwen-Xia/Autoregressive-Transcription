quota -s -u  yrb

# merely reboot gpus

sudo lsof /dev/nvidia* | awk '{print $2}' | tail -n +2 | xargs -r kill -9

[1] 1055843
[1]+  Exit 127                nohup train.py --config-name slakh_all2all > slakhall2all.log 2>&1


python train.py --config-name slakh_all2all run_name=all2all_pred_instru_no_info midi_include_program=True train.device=cuda:0 

nohu python train.py --config-name slakh_all2all run_name=all2all_pred_instru_info midi_include_program=True  train.device=cuda:1 > logs/all2all_pred_instru_info.log 2>&1 &

python train.py --config-name slakh_n2one run_name=n2one

nohup python train.py --config-name n2sev run_name=n2rand train.device=cuda:4 > logs/n2sevfull.log 2>&1 &

nohup python train.py --config-name n2sev run_name=randmix_all train.device=cuda:3 train_datasets.Slakh2100.mode=rand_mix test_datasets.Slakh2100.mode=all > logs/randmix.log 2>&1 & #randmix 2 all 不是conditional source separation

nohup python train.py --config-name slakh_all2all run_name=MuQ_frozen_w_large_bs_4 train.device=cuda:1 train.gradient_accumulation=1 > logs/all2all4.log 2>&1 & #* 这是一个相对fair play, 调了llm 和 encoder

nohup python train.py --config-name slakh_all2all run_name=largelm_bs_4 train.device=cuda:2 train.gradient_accumulation=1 audio_encoder.name=PianoTranscriptionCRnn audio_encoder.trainable=false > logs/all2all_largelm.log 2>&1 & #* 这是一个相对fair play, 调了llm 和 encoder

nuhup python train.py --config-name masestro run_name=MuQ_frozen train.device=cuda:6 audio_encoder.name=MuQ trainable=false > logs/maestro_ablate.log 2>&1 &



nohup python auto_launch.py  3,4 -- --config-name all2all run_name=add_drumpitch tokenizer.drum_pitch=True > logs/all2all_add_drumpitch.log 2>&1 &
python debug_train_eval.py --config-name all2all

nohup python train.py --config-name maestro run_name=reproduce > logs/maestro_reproduce.log 2>&1 &

nohup python train_slakh.py --config /data/yrb/musicarena/Haiwen/piano_transcription/configs/conformer2d_nopool.yaml > logs/conformer2d_nopool_w_decoder.log 2>&1 &