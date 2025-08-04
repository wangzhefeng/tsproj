export CUDA_VISIBLE_DEVICES=0
export LOG_NAME=wind_lstm_ms_seq2seq_multi_step

model_name=LSTM


python -u run_dl.py \
    --task_name long_term_forecast \
    --des 'Exp' \
    --is_training 1 \
    --is_testing 0 \
    --is_forecasting 0 \
    --model_id wind_lstm_ms_seq2seq_multi_step \
    --model $model_name \
    --root_path ./dataset \
    --data_path wind_dataset.csv \
    --data wind_dataset \
    --features MS \
    --target WIND \
    --pred_method recursive_multi_step \
    --checkpoints ./saved_results/pretrained_models/ \
    --test_results ./saved_results/test_results/ \
    --predict_results ./saved_results/predict_results/ \
    --freq d \
    --embed timeF \
    --seq_len 20 \
    --feature_size 8 \
    --hidden_size 256 \
    --output_size 2 \
    --num_layers 2 \
    --train_ratio 0.8 \
    --test_ratio 0.2 \
    --iters 1 \
    --train_epochs 10 \
    --learning_rate 3e-4 \
    --num_workers 0 \
    --batch_size 32 \
    --loss MSE \
    --activation gelu \
    --use_dtw 0 \
    --patience 7 \
    --lradj type1 \
    --scale 0 \
    --inverse 0 \
    --use_gpu 1 \
    --gpu_type 'cuda' \
    --use_multi_gpu 0 \
    --devices 0,1,2,3,4,5,6,7

