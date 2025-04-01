export CUDA_VISIBLE_DEVICES=0
export LOG_NAME=ETTm1

model_name=LSTM_Univariate_SingleOutput


python -u run_dl.py \
    --task_name long_term_forecast \
    --des 'Exp' \
    --is_training 1 \
    --is_testing 0 \
    --is_forecasting 0 \
    --model_id wind_96_96 \
    --model $model_name \
    --root_path ./dataset \
    --data_path wind_dataset.csv \
    --data wind_dataset \
    --features MS \
    --target OT \
    --checkpoints ./saved_results/pretrained_models/ \
    --test_results ./saved_results/test_results/ \
    --predict_results ./saved_results/predict_results/ \
    --freq 15min \
    --embed timeF \
    --seq_len 1 \
    --label_len 48 \
    --pred_len 96 \
    --target_index 0 \
    --split_ratio 0.8 \
    --train_ratio 0.7 \
    --test_ratio 0.2 \
    --feature_size 1 \
    -- num_layers 2 \
    --hidden_size 256 \
    --output_size 1 \
    --num_workers 0 \
    --iters 1 \
    --train_epochs 10 \
    --batch_size 32 \
    --loss MSE \
    --best_loss 0 \
    --activation gelu \
    --use_dtw 0 \
    --learning_rate 3e-4 \
    --patience 7 \
    --lradj type1 \
    --scale 1 \
    --inverse 1 \
    --use_gpu 0 \
    --gpu_type 'cuda' \
    --use_multi_gpu 0 \
    --devices 0,1,2,3,4,5,6,7
