quota -s -u  yrb

# merely reboot gpus

sudo lsof /dev/nvidia* | awk '{print $2}' | tail -n +2 | xargs -r kill -9

[1] 1055843
[1]+  Exit 127                nohup train.py --config-name slakh_all2all > slakhall2all.log 2>&1


python train.py --config-name slakh_all2all run_name=all2all_pred_instru_no_info midi_include_program=True train.device=cuda:0 

nohup > python train.py --config-name slakh_all2all run_name=all2all_pred_instru_info midi_include_program=True  train.device=cuda:1 > logs/all2all_pred_instru_info.log 2>&1 &

python train.py --config-name slakh_n2one run_name=n2one