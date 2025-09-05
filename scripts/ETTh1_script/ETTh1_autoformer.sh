export CUDA_VISIBLE_DEVICES=0
export LOG_NAME=autoformer-etth1

model_name=Autoformer

# 训练、验证、测试
python -u run_dl.py \
    --task_name long_term_forecast \
    --des 'Exp Autoformer_24_12_24' \
    --is_training 1 \
    --is_testing 1 \
    --testing_step 24 \
    --is_forecasting 0 \
    --model_id etth1_24_12_24 \
    --model $model_name \
    --root_path ./dataset/ETT-small \
    --data_path ETTh1.csv \
    --data ETTh1 \
    --features M \
    --target OT \
    --time date \
    --checkpoints ./saved_results/pretrained_models/ \
    --test_results ./saved_results/test_results/ \
    --predict_results ./saved_results/predict_results/ \
    --freq 1h \
    --embed timeF \
    --seq_len 24 \
    --label_len 12 \
    --pred_len 24 \
    --train_ratio 0.7 \
    --test_ratio 0.2 \
    --moving_avg 25 \
    --embed_type 0 \
    --d_model 64 \
    --d_ff 2048 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --n_heads 2 \
    --dropout 0.05 \
    --num_workers 0 \
    --itr 1 \
    --train_epochs 5 \
    --batch_size 64 \
    --loss MSE \
    --activation gelu \
    --use_dtw 0 \
    --learning_rate 1e-5 \
    --patience 7 \
    --lradj type1 \
    --scale 1 \
    --inverse 1 \
    --use_gpu 1 \
    --gpu_type 'mps' \
    --use_multi_gpu 0 \
    --devices 0,1,2,3,4,5,6,7
