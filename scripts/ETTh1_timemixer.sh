export CUDA_VISIBLE_DEVICES=1
export LOG_NAME=timemixer

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
    --des 'Exp TimeMixer_96' \
    --is_training 1 \
    --is_testing 1 \
    --is_forecasting 0 \
    --model_id weather_96_96 \
    --model $model_name \
    --root_path ./dataset/weather \
    --data_path weather.csv \
    --data weather \
    --features M \
    --target OT \
    --time date \
    --checkpoints ./saved_results/pretrained_models/ \
    --test_results ./saved_results/test_results/ \
    --predict_results ./saved_results/predict_results/ \
    --freq 15min \
    --embed timeF \
    --seq_len 96 \
    --label_len 0 \
    --pred_len 96 \
    --train_ratio 0.8 \
    --test_ratio 0.3 \
    --moving_avg 25 \
    --embed_type 0 \
    --d_model 16 \
    --d_ff 32 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --e_layers 3 \
    --d_layers 1 \
    --factor  3 \
    --n_heads 1 \
    --dropout 0.05 \
    --top_k 5 \
    --down_sampling_layers 3 \
    --down_sampling_window 2 \
    --begin_order 1 \
    --num_workers 0 \
    --itr 1 \
    --train_epochs 10 \
    --batch_size 32 \
    --loss MSE \
    --activation gelu \
    --use_dtw 0 \
    --learning_rate 5e-3 \
    --patience 10 \
    --lradj type1 \
    --scale 1 \
    --inverse 1 \
    --use_gpu 0 \
    --gpu_type 'cuda' \
    --use_multi_gpu 0 \
    --devices 0,1,2,3,4,5,6,7
