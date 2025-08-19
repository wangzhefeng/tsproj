# export CUDA_VISIBLE_DEVICES=0
export LOG_NAME=timemixer_A

model_name=TimeMixer

# 模型大小相关参数
# --d_model 512
# --d_ff 2048
# --n_heads 8
# --e_layers 2
# --d_layers 1

# 训练最终用于预测的模型
python -u run_dl.py \
    --task_name long_term_forecast \
    --des 'Exp TimeMixer_288_0_288_A' \
    --is_training 1 \
    --is_testing 0 \
    --is_forecasting 0 \
    --model_id aidc_a_288_0_288 \
    --model $model_name \
    --root_path ./dataset/electricity_work/ \
    --data_path AIDC_A_dataset.csv \
    --data AIDC_A_dataset \
    --features M \
    --target y_A \
    --time date \
    --checkpoints ./saved_results/pretrained_models/ \
    --test_results ./saved_results/test_results/ \
    --predict_results ./saved_results/predict_results/ \
    --freq 5min \
    --embed timeF \
    --seq_len 288 \
    --label_len 0 \
    --pred_len 288 \
    --train_ratio 0.8 \
    --test_ratio 0.1 \
    --moving_avg 25 \
    --embed_type 0 \
    --d_model 32 \
    --d_ff 64 \
    --enc_in 1233 \
    --dec_in 1233 \
    --c_out 1233 \
    --e_layers 2 \
    --d_layers 2 \
    --factor 3 \
    --n_heads 2 \
    --dropout 0.05 \
    --top_k 5 \
    --down_sampling_layers 2 \
    --down_sampling_method avg \
    --down_sampling_window 2 \
    --begin_order 0 \
    --num_workers 4 \
    --itr 1 \
    --train_epochs 5 \
    --batch_size 64 \
    --loss MSE \
    --activation gelu \
    --use_dtw 0 \
    --learning_rate 1e-4 \
    --patience 10 \
    --lradj type1 \
    --scale 1 \
    --inverse 1 \
    --use_gpu 1 \
    --gpu_type 'cuda' \
    --use_multi_gpu 1 \
    --devices 6,7
