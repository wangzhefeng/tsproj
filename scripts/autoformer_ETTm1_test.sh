export CUDA_VISIBLE_DEVICES=1
export LOG_NAME=asc

model_name=Autoformer

# 模型大小相关参数
# --d_model 512
# --d_ff 2048

# 训练、验证、测试
python -u run_dl.py \
    --task_name long_term_forecast \
    --des 'Exp' \
    --is_training 0 \
    --is_testing 1 \
    --is_forecasting 1 \
    --model_id ETTm1_96_96 \
    --model $model_name \
    --root_path ./dataset/long_term_forecast/ETT-small \
    --data_path ETTm1.csv \
    --data ETTm1 \
    --features MS \
    --target OT \
    --checkpoints ./saved_results/pretrained_models/ \
    --test_results ./saved_results/test_results/ \
    --predict_results ./saved_results/predict_results/ \
    --freq 15min \
    --embed timeF \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --train_ratio 0.7 \
    --test_ratio 0.2 \
    --moving_avg 25 \
    --embed_type 0 \
    --d_model 512 \
    --d_ff 2048 \
    --enc_in 7 \
    --dec_in 7 \
    --e_layers 2 \
    --d_layers 1 \
    --factor  3 \
    --n_heads 1 \
    --c_out 1 \
    --dropout 0.05 \
    --output_attention 0 \
    --num_workers 0 \
    --iters 1 \
    --train_epochs 1 \
    --batch_size 1 \
    --loss MSE \
    --activation gelu \
    --use_dtw 0 \
    --learning_rate 1e-4 \
    --patience 7 \
    --lradj type1 \
    --scale 1 \
    --inverse 1 \
    --use_amp 0 \
    --use_gpu 0 \
    --gpu_type 'cuda' \
    --use_multi_gpu 0 \
    --devices 0,1,2,3,4,5,6,7
