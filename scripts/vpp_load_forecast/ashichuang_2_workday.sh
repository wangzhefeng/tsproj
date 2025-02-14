export CUDA_VISIBLE_DEVICES=0
export LOG_NAME=asc2

model_name=Transformer_original

# small model: ashichuang asc2 data-epoch=10
# python -u tf_power_forecasting.py \
#     --is_training 1 \
#     --is_predicting 1 \
#     --root_path ./dataset/ashichuang_dev_20250206_hist30days_pred1days/asc2/pred/ \
#     --data_path df_history_workday.csv \
#     --rolling_data_path ETTh1-Test.csv \
#     --target load \
#     --freq h \
#     --embed timeF \
#     --seq_len 72 \
#     --label_len 12 \
#     --pred_len 24 \
#     --model $model_name \
#     --rollingforecast 1 \
#     --embed_type 0 \
#     --d_model 64 \
#     --enc_in 21 \
#     --dec_in 21 \
#     --e_layers 4 \
#     --d_layers 4 \
#     --n_heads 1 \
#     --d_ff 2048 \
#     --activation gelu \
#     --c_out 1 \
#     --dropout 0.05 \
#     --rev 1 \
#     --output_attention 0 \
#     --use_dtw 0 \
#     --padding 0 \
#     --loss MSE \
#     --features MS \
#     --iters 10 \
#     --train_epochs 10 \
#     --batch_size 1 \
#     --learning_rate 1e-5 \
#     --patience 7 \
#     --lradj type1 \
#     --checkpoints ./saved_results/pretrained_models/ \
#     --test_results ./saved_results/test_results/ \
#     --predict_results ./saved_results/predict_results/ \
#     --show_results 1 \
#     --inverse 1 \
#     --scale 1 \
#     --gpu 0 \
#     --use_multi_gpu 0 \
#     --devices 0,1,2,3,4,5,6,7 \
#     --num_workers 0


# small model: ashichuang asc2 data-epoch=20
python -u tf_power_forecasting.py \
    --is_training 0 \
    --is_predicting 1 \
    --root_path ./dataset/ashichuang_dev_20250206_hist30days_pred1days/asc2/pred/ \
    --data_path df_history_workday.csv \
    --rolling_data_path ETTh1-Test.csv \
    --target load \
    --freq h \
    --embed timeF \
    --seq_len 72 \
    --label_len 12 \
    --pred_len 24 \
    --model $model_name \
    --rollingforecast 1 \
    --embed_type 0 \
    --d_model 64 \
    --enc_in 21 \
    --dec_in 21 \
    --e_layers 4 \
    --d_layers 4 \
    --n_heads 1 \
    --d_ff 2048 \
    --activation gelu \
    --c_out 1 \
    --dropout 0.05 \
    --rev 1 \
    --output_attention 0 \
    --use_dtw 0 \
    --padding 0 \
    --loss MSE \
    --features MS \
    --iters 10 \
    --train_epochs 20 \
    --batch_size 1 \
    --learning_rate 1e-5 \
    --patience 7 \
    --lradj type1 \
    --checkpoints ./saved_results/pretrained_models/ \
    --test_results ./saved_results/test_results/ \
    --predict_results ./saved_results/predict_results/ \
    --show_results 1 \
    --inverse 1 \
    --scale 1 \
    --gpu 0 \
    --use_multi_gpu 0 \
    --devices 0,1,2,3,4,5,6,7 \
    --num_workers 0
