# -*- coding: utf-8 -*-

# ***************************************************
# * File        : run.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-01-08
# * Version     : 1.0.010821
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import argparse

from exp.exp_forecasting_tf import Exp_Forecast_TF
from exp.exp_forecasting_dl import Exp_Forecast
from utils.print_args import print_args
from utils.device import torch_gc
from utils.random_seed import set_seed
from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def args_parse():
    parser = argparse.ArgumentParser(description='Transformer Multivariate Time Series Forecasting')
    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecasting', 
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--is_training', type=int, required=True, default=0, help='Whether to conduct training')
    parser.add_argument('--is_testing', type=int, required=True, default=0, help='Whether to conduct testing')
    parser.add_argument('--is_forecasting', type=int, required=True, default=0, help='Whether to conduct forecasting')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Transformer', help='model name, options: [Autoformer, Transformer, TimesNet]')
    # data loader
    parser.add_argument('--root_path', type=str, required=True, default='./dataset/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, required=True, default='ETTh1.csv', help='data file')
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--freq', type=str, required=True, default='h', 
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], \
                              you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--features', type=str, default='MS', 
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, required=True, default='OT', help='target feature in S or MS task')    
    parser.add_argument('--checkpoints', type=str, default='./saved_results/pretrained_models/', help='location of model models')
    parser.add_argument('--test_results', type=str, default='./saved_results/test_results/', help='location of model models')
    parser.add_argument('--predict_results', type=str, default='./saved_results/predict_results/', help='location of model models') 
    # forecasting task
    parser.add_argument('--seq_len', type=int, required=True, default=72, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=12, help='start token length')
    parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
    parser.add_argument('--train_ratio', type=float, required=True, default=0.7, help='train dataset ratio')
    parser.add_argument('--test_ratio', type=float, required=True, default=0.2, help='test dataset ratio')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--scale', type=int, default=1, help = 'data transform')
    parser.add_argument('--inverse', type=int, default=1, help='inverse output data')
    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')
    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%%)')
    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=1, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false', default=True,
                        help='whether to use distilling in encoder, using this argument means not using distilling')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1, help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg', help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None, help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=96, help='the length of segmen-wise iteration of SegRNN') 
    parser.add_argument('--embed_type', type=int, default=0, help='0: value embedding + temporal embedding + positional embedding 1: value embedding + positional embedding')
    # TODO parser.add_argument('--rev', type=int, default=1, help='whether to apply RevIN')
    parser.add_argument('--output_attention', type=int, default=0, help='whether to output attention in ecoder')
    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--iters', type=int, default=10, help='train iters')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size of train input data')
    parser.add_argument('--patience', type = int, default=3, help = 'early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', type=int, default=0, help='use automatic mixed precision training') 
    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128], help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')
    # metrics (dtw)
    parser.add_argument('--use_dtw', type=int, default=0, help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')
    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true", help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true", help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true", help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true", help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")
    # TimeXer
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    # GPU
    parser.add_argument('--use_gpu', type=int, default=1, help='use gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')
    parser.add_argument('--use_multi_gpu', type=int, default=0, help = 'use multiple gpus')
    parser.add_argument('--devices', type=str, default="0,1,2,3,4,5,6,7,8", help='device ids of multile gpus')
    # TODO FreDF
    # parser.add_argument('--add_fredf', type=int, default=0, help='weather add fredf loss')
    # parser.add_argument('--rec_lambda', type=float, default=0., help='weight of reconstruction function')
    # parser.add_argument('--auxi_lambda', type=float, default=1, help='weight of auxilary function')
    # parser.add_argument('--auxi_mode', type=str, default='fft', help='auxi loss mode, options: [fft, rfft]')
    # parser.add_argument('--auxi_type', type=str, default='complex', help='auxi loss type, options: [complex, mag, phase, mag-phase]')
    # parser.add_argument('--leg_degree', type=int, default=2, help='degree of legendre polynomial')
    # parser.add_argument('--auxi_loss', type=str, default='MAE', help='loss function')
    # parser.add_argument('--module_first', type=int, default=1, help='calculate module first then mean ')
    # parser.add_argument('--add_noise', type=int, default=1, help='add noise')
    # parser.add_argument('--noise_amp', type=float, default=1, help='noise ampitude')
    # parser.add_argument('--noise_freq_percentage', type=float, default=0.05, help='noise frequency percentage')
    # TODO others
    parser.add_argument("--target_index", type=int, default=0, help="Target index for forecasting")
    parser.add_argument("--pred_method", type=str, default="recursive_multi_step", help="Prediction method: recursive_multi_step | direct_multi_step_output | direct_recursive_mix")
    parser.add_argument("--feature_size", type=int, default=1, help="feature size")
    parser.add_argument("--hidden_size", type=int, default=256, help="hidden size")
    parser.add_argument("--num_layers", type=int, default=2, help="number of layers")
    parser.add_argument("--output_size", type=int, default=1, help="output size")
    # 命令行参数解析
    args = parser.parse_args()

    return args


def run(args):
    # setting record of experiments
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.expand,
        args.d_conv,
        args.factor,
        args.embed,
        args.distil,
        args.des,
        # TODO args.add_fredf,
    )
    # 模型训练
    if args.is_training: 
        for ii in range(args.iters):
            # setting record of experiments
            training_setting = setting + str(ii)
            # 模型训练
            logger.info(f">>>>>>>>> start training: iter-{ii}: {training_setting}>>>>>>>>>>")
            exp = Exp_Forecast(args)
            train_results = exp.train(training_setting)
            # TODO 模型测试
            # logger.info(f">>>>>>>>> start testing: iter-{ii}: {training_setting}>>>>>>>>>>")
            # exp.test(training_setting, load = True)
    
    # 模型测试
    if args.is_testing:
        ii = 0
        # setting record of experiments
        test_setting = setting + str(ii)
        # 模型测试
        logger.info(f">>>>>>>>> start testing: iter-{ii}: {test_setting}>>>>>>>>>>")
        exp = Exp_Forecast(args)
        exp.test(test_setting, load = True)
    
    # 模型最终训练
    if not args.is_training and not args.is_testing and not args.is_forecasting:
        ii = "final"
        # setting record of experiments
        final_training_setting = setting + str(ii)
        # 模型在所有训练数据上训练
        logger.info(f">>>>>>>>> start training: iter-{ii}: {final_training_setting}>>>>>>>>>>")
        exp = Exp_Forecast(args)
        train_results = exp.train(final_training_setting)
        logger.info(f"train_results: {train_results}")
    
    # 模型预测
    if args.is_forecasting: 
        ii = 0
        # setting record of experiments
        forecasting_setting = setting + str(ii)
        # 模型预测
        logger.info(f">>>>>>>>> start forecasting: {forecasting_setting}>>>>>>>>>>")
        exp = Exp_Forecast(args)
        exp.forecast(forecasting_setting, load = True)
        
    # empty cache
    logger.info(f">>>>>>>>>>>> Empty cuda cache and memory pecices...")
    torch_gc(gpu_type=args.gpu_type, cuda_device=exp.device)





# 测试代码 main 函数
def main():
    # 设置随机数
    set_seed(seed = 2023)
    # 参数解析
    args = args_parse()
    print_args(args)
    # 参数使用
    run(args)

if __name__ == "__main__":
    main()
