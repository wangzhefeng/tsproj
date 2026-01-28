nvidia-smi

# export CUDA_VISIBLE_DEVICES=0
export LOG_NAME=itransformer-AIDC_A_dataset-larger

model_name=iTransformer

# 训练、验证、测试
python -u run_dl.py \
    --task_name long_term_forecast \
    --des 'Exp iTransformer_288_144_288_A_larger' \
    --is_training 1 \
    --is_testing 1 \
    --testing_step 288 \
    --is_forecasting 0 \
    --model_id iTransformer_288_144_288_A_larger \
    --model $model_name \
    --root_path ./dataset/electricity_work/ \
    --data_path A3_F2_201_A_dataset.csv \
    --data A3_F2_201_A_dataset \
    --features M \
    --target y \
    --time time \
    --checkpoints ./saved_results/pretrained_models/ \
    --test_results ./saved_results/test_results/ \
    --predict_results ./saved_results/predict_results/ \
    --freq 5min \
    --embed timeF \
    --seq_len 288 \
    --label_len 144 \
    --pred_len 288 \
    --train_ratio 0.7 \
    --test_ratio 0.2 \
    --moving_avg 25 \
    --embed_type 0 \
    --d_model 128 \
    --d_ff 512 \
    --enc_in 1312 \
    --dec_in 1312 \
    --c_out 1312 \
    --e_layers 8 \
    --d_layers 8 \
    --factor 3 \
    --n_heads 4 \
    --dropout 0.05 \
    --num_workers 4 \
    --itr 1 \
    --train_epochs 20 \
    --batch_size 64 \
    --loss MSE \
    --activation gelu \
    --use_dtw 0 \
    --learning_rate 3e-5 \
    --patience 7 \
    --lradj type1 \
    --scale 1 \
    --inverse 1 \
    --use_gpu 1 \
    --gpu_type 'cuda' \
    --use_multi_gpu 1 \
    --devices 0,1,2,3,4,5,6,7
