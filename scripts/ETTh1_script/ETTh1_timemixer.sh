export CUDA_VISIBLE_DEVICES=0
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
    --des 'Exp TimeMixer_96_0_96' \
    --is_training 1 \
    --is_testing 1 \
    --is_forecasting 0 \
    --model_id etth1_96_0_96 \
    --model $model_name \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --data ETTh1 \
    --features MS \
    --target OT \
    --time date \
    --checkpoints ./saved_results/pretrained_models/ \
    --test_results ./saved_results/test_results/ \
    --predict_results ./saved_results/predict_results/ \
    --freq 1h \
    --embed timeF \
    --seq_len 96 \
    --label_len 0 \
    --pred_len 96 \
    --train_ratio 0.7 \
    --test_ratio 0.2 \
    --moving_avg 25 \
    --embed_type 0 \
    --d_model 16 \
    --d_ff 32 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 1 \
    --e_layers 2 \
    --d_layers 1 \
    --factor  3 \
    --n_heads 1 \
    --dropout 0.05 \
    --top_k 5 \
    --down_sampling_layers 2 \
    --down_sampling_method avg \
    --down_sampling_window 2 \
    --begin_order 0 \
    --num_workers 0 \
    --itr 1 \
    --train_epochs 1 \
    --batch_size 128 \
    --loss MSE \
    --activation gelu \
    --use_dtw 0 \
    --learning_rate 0.01 \
    --patience 10 \
    --lradj type1 \
    --scale 1 \
    --inverse 1 \
    --use_gpu 1 \
    --gpu_type 'mps' \
    --use_multi_gpu 0 \
    --devices 0,1,2,3,4,5,6,7
