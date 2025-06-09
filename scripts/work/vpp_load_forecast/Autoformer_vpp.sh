export CUDA_VISIBLE_DEVICES=1
export LOG_NAME=asc

model_name=Autoformer

python -u run_autoformer.py \
    --task_name long_term_forecast \
    --des 'Exp' \
    --is_training 1 \
    --do_forecasting 0 \
    --model_id asc1_96_48_96 \
    --model $model_name \
    --root_path ./dataset/ashichuang_dev_20250206_hist30days_pred1days/asc1/pred/ \
    --data_path df_history.csv \
    --data df_history \
    --features MS \
    --target load \
    --checkpoints ./saved_results/pretrained_models/ \
    --test_results ./saved_results/test_results/ \
    --predict_results ./saved_results/predict_results/ \
    --freq 15min \
    --embed timeF \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --moving_avg 4 \
    --embed_type 0 \
    --d_model 512 \
    --enc_in 9 \
    --dec_in 9 \
    --e_layers 2 \
    --d_layers 1 \
    --n_heads 1 \
    --d_ff 2048 \
    --c_out 1 \
    --dropout 0.05 \
    --rev 1 \
    --output_attention 0 \
    --padding 0 \
    --num_workers 0 \
    --iters 1 \
    --train_epochs 5 \
    --batch_size 1 \
    --loss MSE \
    --activation gelu \
    --use_dtw 0 \
    --learning_rate 1e-4 \
    --patience 7 \
    --lradj type1 \
    --scale 1 \
    --use_gpu 0 \
    --gpu_type 'cuda' \
    --use_multi_gpu 0 \
    --devices 0,1,2,3,4,5,6,7


# TODO
# --use_amp 0 \
# --inverse 1 \
