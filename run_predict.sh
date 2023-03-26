python run_predict.py \
    --batch_size 4 \
    --save_pdb \
    --ipc_file ~/tmp.ipc \
    --data_dir ./sabdab_release0504/clean \
    --prefix ./results/2499 \
    --model ./model.pth \
    --ipc_file ~/test.ipc \
    --model_features ./examples/model_features_abrep.json  \
    --name_idx ./sabdab_release0504/test_0.99.idx
    
#--gpu_list 0 \
