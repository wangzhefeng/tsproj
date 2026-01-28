export LOG_NAME=itransformer-AIDC_A_dataset-large

model_name=LightGBM

# 训练、验证、测试
python -u run_ml.py \
    --task_name machine_learning_long_term_forecast \
    --is_training 1 \
    --is_testing 1 \
    --testing_step 288 \
    --is_forecasting 0 \
    --model_id LightGBM_288_144_288_A_large \
    --model $model_name \
    --data_dir ./dataset/electricity_work/ \
    --data_path ETTh1.csv \
    --data ETTh1 \
    --freq 5min \
    --n_per_day 288 \
    --target_ts_feat time \
    --target_series_numeric_features 0 \
    --target_series_categorical_features 0 \
    --target y \
    --date_history_path date.csv \
    --date_future_path date_future.csv \
    --date_ts_feat date \
    --weather_history_path weather.csv \
    --weather_future_path weather_future.csv \
    --weather_ts_feat ts \
    --now_time 2025-05-19 \
    --scale 1 \
    --inverse 1 \
    --target_transform 0 \
    --target_transform_predict 0 \
    --date_type 0 \
    --lags 1,2,3 \
    --pred_method univariate-multip-step-recursive \
    --features MS \
    --history_days 30 \
    --predict_days 1 \
    --window_days 15 \
    --train_ratio 0.7 \
    --test_ratio 0.2 \
    --loss MSE \
    --learning_rate 3e-5 \
    --patience 7 \
    --checkpoints ./saved_results/pretrained_models/ \
    --test_results ./saved_results/test_results/ \
    --pred_results ./saved_results/predict_results/

    # --num_workers 4 \
    # --use_gpu 1 \
    # --gpu_type 'cuda' \
    # --use_multi_gpu 1 \
    # --devices 0,1,2,3,4,5,6,7
