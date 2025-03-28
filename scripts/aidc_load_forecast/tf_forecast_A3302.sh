export CUDA_VISIBLE_DEVICES=5
export LOG_NAME=A3021

model_name=Transformer

python -u tf_power_forecasting.py \
    --is_training 0 \
    --is_predicting 1 \
    --root_path ./dataset/electricity/ \
    --data_path all_df_A3_301_302.csv \
    --rolling_data_path ETTh1-Test.csv \
    --target 302_load \
    --freq h \
    --embed timeF \
    --seq_len 72 \
    --label_len 12 \
    --pred_len 24 \
    --model $model_name \
    --rollingforecast 1 \
    --embed_type 0 \
    --d_model 64 \
    --enc_in 6946 \
    --dec_in 6946 \
    --e_layers 100 \
    --d_layers 100 \
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
    --iters 100 \
    --train_epochs 1000 \
    --batch_size 1 \
    --learning_rate 1e-5 \
    --patience 7 \
    --lradj type1 \
    --checkpoints ./saved_results/pretrained_models/ \
    --test_results ./saved_results/test_results/ \
    --predict_results ./saved_results/predict_results/ \
    --inverse 1 \
    --scale 1 \
    --gpu 5 \
    --use_multi_gpu 0 \
    --devices 0,1,2,3,4,5,6,7 \
    --num_workers 0 \
