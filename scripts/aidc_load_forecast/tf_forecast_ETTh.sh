export CUDA_VISIBLE_DEVICES=4

model_name=Transformer

python -u tf_power_forecasting.py \
    --is_training 1 \
    --is_predicting 1 \
    --root_path ./dataset/electricity/ \
    --data_path ETTh1.csv \
    --rolling_data_path ETTh1-Test.csv \
    --target OT \
    --freq h \
    --embed timeF \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --model $model_name \
    --rollingforecast 1 \
    --embed_type 0 \
    --d_model 64 \
    --enc_in 10793 \
    --dec_in 10793 \
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
    --train_epochs 10 \
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
    --use_gpu 1 \
    --gpu 0 \
    --use_multi_gpu 0 \
    --devices 0,1,2,3 \
    --num_workers 0 \
