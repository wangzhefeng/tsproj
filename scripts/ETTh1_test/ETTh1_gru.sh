export CUDA_VISIBLE_DEVICES=0
export LOG_NAME=ETTh1_gru

model_name=GRU

python -u run_dl.py \
    --task_name long_term_forecast \
    --des 'Exp' \
    --is_training 1 \
    --is_testing 0 \
    --testing_step 24 \
    --is_forecasting 0 \
    --model_id ETTh1_gru \
    --model $model_name \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --data ETTh1 \
    --features MS \
    --target OT \
    --pred_method recursive_multi_step \
    --checkpoints ./saved_results/pretrained_models/ \
    --test_results ./saved_results/test_results/ \
    --predict_results ./saved_results/predict_results/ \
    --freq d \
    --embed timeF \
    --window_len 6 \
    --pred_len 2 \
    --step_size 1 \
    --feature_size 1 \
    --output_size 1 \
    --output 1 \
    --hidden_size 32 \
    --num_layers 2 \
    --train_ratio 0.8 \
    --test_ratio 0.2 \
    --iters 1 \
    --train_epochs 10 \
    --learning_rate 5e-3 \
    --batch_size 1 \
    --loss MSE \
    --activation gelu \
    --use_dtw 0 \
    --patience 7 \
    --lradj type1 \
    --scale 1 \
    --inverse 1 \
    --num_workers 0 \
    --use_gpu 0 \
    --gpu_type 'cuda' \
    --use_multi_gpu 0 \
    --devices 0,1,2,3,4,5,6,7
