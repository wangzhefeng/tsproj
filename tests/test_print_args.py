# -*- coding: utf-8 -*-

import argparse
from print_args import print_args

def create_mock_args():
    # 创建模拟的 args 对象
    parser = argparse.ArgumentParser()
    
    # Basic Config
    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--model_id', type=str, default='test_model')
    parser.add_argument('--model', type=str, default='Transformer')
    
    # Data Loader
    parser.add_argument('--data', type=str, default='ETTh1')
    parser.add_argument('--root_path', type=str, default='./data/ETT/')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    
    # Forecasting Task
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly')
    parser.add_argument('--inverse', type=int, default=0)
    
    # Model Parameters
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--num_kernels', type=int, default=6)
    parser.add_argument('--enc_in', type=int, default=7)
    parser.add_argument('--dec_in', type=int, default=7)
    parser.add_argument('--c_out', type=int, default=7)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--moving_avg', type=int, default=25)
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--distil', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--output_attention', type=int, default=0)
    
    # Run Parameters
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--des', type=str, default='test')
    parser.add_argument('--loss', type=str, default='mse')
    parser.add_argument('--lradj', type=str, default='type1')
    parser.add_argument('--use_amp', type=int, default=0)
    
    # GPU
    parser.add_argument('--use_gpu', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_multi_gpu', type=int, default=0)
    parser.add_argument('--devices', type=str, default='0')
    
    # De-stationary Projector Params
    parser.add_argument('--p_hidden_dims', type=list, default=[128, 128])
    parser.add_argument('--p_hidden_layers', type=int, default=2)
    
    return parser.parse_args([])

if __name__ == "__main__":
    args = create_mock_args()
    print_args(args)
