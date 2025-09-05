export CUDA_VISIBLE_DEVICES=0
export LOG_NAME=itransformer-AIDC_A

model_name=iTransformer

# 训练、验证、测试
python -u run_dl.py \
    --task_name long_term_forecast \
    --des 'Exp iTransformer_288_0_288_A' \
    --is_training 1 \
    --is_testing 1 \
    --testing_step 288 \
    --is_forecasting 0 \
    --model_id iTransformer_288_0_288_A \
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
    --label_len 144 \
    --pred_len 288 \
    --train_ratio 0.7 \
    --test_ratio 0.2 \
    --moving_avg 25 \
    --embed_type 0 \
    --d_model 16 \
    --d_ff 32 \
    --enc_in 1233 \
    --dec_in 1233 \
    --c_out 1233 \
    --e_layers 2 \
    --d_layers 2 \
    --factor 3 \
    --n_heads 1 \
    --dropout 0.05 \
    --down_sampling_layers 2 \
    --down_sampling_window 2 \
    --begin_order 0 \
    --num_workers 0 \
    --itr 1 \
    --train_epochs 5 \
    --batch_size 8 \
    --loss MSE \
    --activation gelu \
    --use_dtw 0 \
    --learning_rate 1e-4 \
    --patience 10 \
    --lradj type1 \
    --scale 1 \
    --inverse 1 \
    --use_gpu 1 \
    --gpu_type 'mps' \
    --use_multi_gpu 0 \
    --devices 0,1,2,3,4,5,6,7
