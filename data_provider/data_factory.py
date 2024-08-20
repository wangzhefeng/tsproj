from torch.utils.data import DataLoader

from data_provider.data_loader import (Dataset_Custom, Dataset_ETT_hour,
                                       Dataset_ETT_minute, Dataset_M4,
                                       MSLSegLoader, PSMSegLoader,
                                       SMAPSegLoader, SMDSegLoader,
                                       SWATSegLoader, UEAloader)
from data_provider.uea import collate_fn


data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader
}


def data_provider(args, flag):
    """
    数据集准备

    Args:
        args (_type_): 参数集
        flag (_type_): 任务标签

    Returns:
        _type_: data_set, data_loader
    """
    # 数据集类
    Data = data_dict[args.data]
    # TODO
    timeenc = 0 if args.embed != 'timeF' else 1
    # 区别在 test 和 train/valid 任务下是否进行 shuffle 数据
    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    # TODO
    drop_last = False
    # batch size
    batch_size = args.batch_size
    # 数据频率
    freq = args.freq
    # 构建 Dataset 和 DataLoader
    if args.task_name == 'anomaly_detection':
        drop_last = False
        # Dataset
        data_set = Data(
            args = args,
            root_path = args.root_path,
            win_size = args.seq_len,
            flag = flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size = batch_size,
            shuffle = shuffle_flag,
            num_workers = args.num_workers,
            drop_last = drop_last
        )
        return data_set, data_loader
    elif args.task_name == 'classification':
        drop_last = False
        # Dataset
        data_set = Data(
            args = args,
            root_path = args.root_path,
            flag = flag,
        )
        # DataLoader
        data_loader = DataLoader(
            data_set,
            batch_size = batch_size,
            shuffle = shuffle_flag,
            num_workers = args.num_workers,
            drop_last = drop_last,
            collate_fn = lambda x: collate_fn(x, max_len = args.seq_len)
        )
        return data_set, data_loader
    else:
        # TODO
        if args.data == 'm4':
            drop_last = False
        # Dataset
        data_set = Data(
            args = args,
            root_path = args.root_path,
            data_path = args.data_path,
            flag = flag,
            size = [args.seq_len, args.label_len, args.pred_len],
            features = args.features,
            target = args.target,
            timeenc = timeenc,
            freq = freq,
            seasonal_patterns = args.seasonal_patterns
        )
        print(flag, len(data_set))
        # DataLoader
        data_loader = DataLoader(
            data_set,
            batch_size = batch_size,
            shuffle = shuffle_flag,
            num_workers = args.num_workers,
            drop_last = drop_last
        )
        return data_set, data_loader
