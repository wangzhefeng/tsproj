nvidia-smi

# export CUDA_VISIBLE_DEVICES=0
export LOG_NAME=timemixer-AIDC_B

model_name=TimeMixer

# 训练、验证、测试
python -u run_dl.py \
    --task_name long_term_forecast \
    --des 'Exp TimeMixer_288_0_288_B' \
    --is_training 1 \
    --is_testing 1 \
    --testing_step 288 \
    --is_forecasting 0 \
    --model_id TimeMixer_288_0_288_B \
    --model $model_name \
    --root_path ./dataset/electricity_work/ \
    --data_path AIDC_B_dataset.csv \
    --data AIDC_B_dataset \
    --features M \
    --target y_B \
    --time date \
    --checkpoints ./saved_results/pretrained_models/ \
    --test_results ./saved_results/test_results/ \
    --predict_results ./saved_results/predict_results/ \
    --freq 5min \
    --embed timeF \
    --seq_len 288 \
    --label_len 0 \
    --pred_len 288 \
    --train_ratio 0.7 \
    --test_ratio 0.2 \
    --moving_avg 25 \
    --embed_type 0 \
    --d_model 16 \
    --d_ff 32 \
    --enc_in 1777 \
    --dec_in 1777 \
    --c_out 1777 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --n_heads 4 \
    --dropout 0.05 \
    --top_k 5 \
    --down_sampling_layers 2 \
    --down_sampling_method avg \
    --down_sampling_window 2 \
    --begin_order 0 \
    --num_workers 4 \
    --itr 1 \
    --train_epochs 20 \
    --batch_size 8 \
    --loss MSE \
    --activation gelu \
    --use_dtw 0 \
    --learning_rate 3e-5 \
    --patience 1 \
    --lradj type1 \
    --scale 1 \
    --inverse 1 \
    --use_gpu 1 \
    --gpu_type 'cuda' \
    --use_multi_gpu 1 \
    --devices 0,1,2,3,4,5,6,7
