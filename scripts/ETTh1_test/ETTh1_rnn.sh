export CUDA_VISIBLE_DEVICES=0

# log vars
model_id=ETTh1_rnn
export LOG_NAME=$model_id

# model vars
model_name=RNN

python -u run_dl.py \
    --task_name long_term_forecast \
    --des 'Exp' \
    --is_training 1 \
    --is_testing 1 \
    --testing_step 24 \
    --is_forecasting 0 \
    --model_id $model_id \
    --model $model_name \
    --root_path ./dataset/ETT-small \
    --data_path ETTh1.csv \
    --data ETTh1 \
    --target OT \
    --train_ratio 0.6 \
    --test_ratio 0.2 \
    --pred_method recursive_multi_step \
    --inspect_fit 1 \
    --rolling_predict 1 \
    --rolling_data_path ETTh1-Test.csv \
    --checkpoints ./saved_results/pretrained_models/ \
    --test_results ./saved_results/test_results/ \
    --predict_results ./saved_results/predict_results/ \
    --freq h \
    --embed timeF \
    --seq_len 126 \
    --pred_len 24 \
    --step_size 1 \
    --features M \
    --feature_size 7 \
    --target_size 7 \
    --hidden_size 64 \
    --kernel_size 3 \
    --num_layers 2 \
    --dropout 0.05 \
    --activation gelu \
    --teacher_forcing 0.3 \
    --iters 1 \
    --train_epochs 20 \
    --batch_size 32 \
    --learning_rate 5e-3 \
    --lr_scheduler 1 \
    --lradj type1 \
    --loss MSE \
    --optimizer adam \
    --use_dtw 0 \
    --patience 14 \
    --scale 1 \
    --inverse 1 \
    --num_workers 0 \
    --use_gpu 1 \
    --gpu_type 'cuda' \
    --use_multi_gpu 0 \
    --devices 0,1,2,3,4,5,6,7
