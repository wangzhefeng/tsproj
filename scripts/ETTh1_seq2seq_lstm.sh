export CUDA_VISIBLE_DEVICES=0

# log vars
model_id=ETTh1_lstm2lstm
export LOG_NAME=$model_id
# model vars
model_name=LSTM2LSTM

python -u run_dl.py \
    --task_name long_term_forecast \
    --des 'Exp' \
    --is_training 1 \
    --is_testing 1 \
    --testing_step 24 \
    --is_forecasting 0 \
    --model_id $model_id \
    --model $model_name \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --data ETTh1 \
    --features M \
    --target OT \
    --pred_method recursive_multi_step \
    --inspect_fit 1 \
    --rolling_predict 1 \
    --rolling_data_path ETTh1-Test.csv \
    --checkpoints ./saved_results/pretrained_models/ \
    --test_results ./saved_results/test_results/ \
    --predict_results ./saved_results/predict_results/ \
    --freq h \
    --embed timeF \
    --seq_len 64 \
    --pred_len 24 \
    --step_size 1 \
    --num_layers 2 \
    --feature_size 7 \
    --hidden_size 128 \
    --kernel_size 3 \
    --target_size 7 \
    --teacher_forcing 0.3 \
    --train_ratio 0.6 \
    --test_ratio 0.2 \
    --iters 1 \
    --train_epochs 20 \
    --batch_size 32 \
    --learning_rate 1e-3 \
    --lr_scheduler 1 \
    --lradj type1 \
    --dropout 0.05 \
    --loss MSE \
    --optimizer adam \
    --activation gelu \
    --use_dtw 0 \
    --patience 14 \
    --scale 1 \
    --inverse 1 \
    --num_workers 0 \
    --use_gpu 1 \
    --gpu_type 'cuda' \
    --use_multi_gpu 0 \
    --devices 0,1,2,3,4,5,6,7
