export CUDA_VISIBLE_DEVICES=0
export LOG_NAME=line_B

model_name=PatchTST

python -u run_dl.py \
    --task_name long_term_forecast \
    --des 'Exp' \
    --is_training 1 \
    --is_testing 1 \
    --testing_step 24 \
    --is_forecasting 0 \
    --model_id df_load_72_24 \
    --model $model_name \
    --root_path ./dataset/electricity/AIDC/line_B \
    --data_path df_load.csv \
    --data df_load \
    --features MS \
    --target load \
    --time time \
    --checkpoints ./saved_results/pretrained_models/ \
    --test_results ./saved_results/test_results/ \
    --predict_results ./saved_results/predict_results/ \
    --freq 5min \
    --embed timeF \
    --seq_len 864 \
    --label_len 144 \
    --pred_len 288 \
    --train_ratio 0.6 \
    --test_ratio 0.3 \
    --moving_avg 25 \
    --embed_type 0 \
    --d_model 64 \
    --d_ff 2048 \
    --enc_in 10792 \
    --dec_in 10792 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --n_heads 1 \
    --c_out 1 \
    --dropout 0.05 \
    --rev 1 \
    --padding 0 \
    --num_workers 0 \
    --itr 1 \
    --train_epochs 30 \
    --batch_size 1 \
    --loss MSE \
    --activation gelu \
    --use_dtw 0 \
    --learning_rate 1e-5 \
    --patience 14 \
    --lradj type1 \
    --scale 1 \
    --inverse 1 \
    --use_gpu 1 \
    --gpu_type 'mps' \
    --use_multi_gpu 0 \
    --devices 0,1,2,3,4,5,6,7
