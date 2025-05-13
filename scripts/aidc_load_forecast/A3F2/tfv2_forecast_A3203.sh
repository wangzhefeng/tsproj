export CUDA_VISIBLE_DEVICES=0
export LOG_NAME=A3203

model_name=Transformer_v2

python -u run.py \
    --task_name long_term_forecast \
    --des 'Exp' \
    --is_training 1 \
    --is_testing 1 \
    --is_forecasting 0 \
    --model_id all_df_72_24 \
    --model $model_name \
    --root_path ./dataset/electricity/A3F2/tf_data \
    --data_path all_df.csv \
    --data all_df \
    --features MS \
    --target 203_load \
    --checkpoints ./saved_results/pretrained_models/ \
    --test_results ./saved_results/test_results/ \
    --predict_results ./saved_results/predict_results/ \
    --freq h \
    --embed timeF \
    --seq_len 72 \
    --label_len 12 \
    --pred_len 24 \
    --train_ratio 0.6 \
    --test_ratio 0.3 \
    --moving_avg 25 \
    --embed_type 0 \
    --d_model 64 \
    --d_ff 2048 \
    --enc_in 10792 \
    --dec_in 10792 \
    --e_layers 12 \
    --d_layers 12 \
    --factor 3 \
    --n_heads 1 \
    --c_out 1 \
    --dropout 0.05 \
    --rev 1 \
    --padding 0 \
    --output_attention 0 \
    --num_workers 0 \
    --iters 1 \
    --train_epochs 30 \
    --batch_size 1 \
    --loss MSE \
    --activation gelu \
    --use_dtw 0 \
    --learning_rate 1e-5 \
    --patience 7 \
    --lradj type1 \
    --scale 1 \
    --inverse 1 \
    --use_amp 0 \
    --use_gpu 1 \
    --gpu_type 'cuda' \
    --use_multi_gpu 0 \
    --devices 0,1,2,3,4,5,6,7
