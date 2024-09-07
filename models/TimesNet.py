import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

from layers.Conv_Blocks import Inception_Block_V1
from layers.Embed import DataEmbedding


def FFT_for_Period(x, k = 2):
    """
    快速傅里叶变换

    Args:
        x (_type_): _description_
        k (int, optional): _description_. Defaults to 2.

    Returns:
        _type_: _description_
    """
    # xf shape [B, T, C], denoting the amplitude of frequency(T) given the datapiece at B,N.
    xf = torch.fft.rfft(x, dim = 1)
    
    # find period by amplitudes: here we assume that the periodic features are basically constant
    # in different batch and channel, so we mean out these two dimensions, 
    # getting a list frequency_list with shape[T] 
    # each element at pos t of frequency_list denotes the overall amplitude at frequency (t).
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    
    # by torch.topk(),we can get the biggest k elements of frequency_list, 
    # and its positions(i.e. the k-main frequencies in top_list).
    _, top_list = torch.topk(frequency_list, k)
    
    # Returns a new Tensor 'top_list', detached from the current graph.
    # The result will never require gradient. Convert to a numpy instance.
    top_list = top_list.detach().cpu().numpy()
    
    # period: a list of shape [top_k], recording the periods of mean frequencies respectively.
    period = x.shape[1] // top_list
    
    # Here, the 2nd item returned has a shape of [B, top_k], 
    # representing the biggest top_k amplitudes 
    # for each piece of data, with N features being averaged.
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len  # sequence length
        self.pred_len = configs.pred_len  # prediction length
        self.k = configs.top_k  # k denotes how many top frequencies are taken into consideration
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(
                configs.d_model, 
                configs.d_ff,
                num_kernels = configs.num_kernels
            ),
            nn.GELU(),
            Inception_Block_V1(
                configs.d_ff, 
                configs.d_model,
                num_kernels = configs.num_kernels
            )
        )

    def forward(self, x):
        # ------------------------------
        # 
        # ------------------------------
        # B: batch size, T: length of time series, N: Number of features
        B, T, N = x.size()
        # ------------------------------
        # 对每个周期进行快速傅里叶变换
        # ------------------------------
        # period_list([top_k]): the top-k-significant period
        # period_weight([B, top_k]): its weight(amplitude)
        period_list, period_weight = FFT_for_Period(x, self.k)
        # ------------------------------
        # 
        # ------------------------------
        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            # paddign: to form a 2D map, we need total length of the sequence, plus the part
            # to be predicted, to be divisible by the period, so padding is needed
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([
                    x.shape[0], 
                    (length - (self.seq_len + self.pred_len)), 
                    x.shape[2]
                ]).to(x.device)
                out = torch.cat([x, padding], dim = 1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            
            # reshape
            # reshape: we need each channel of a single piece of data to be a 2D variable,
            # Also, in order to implement the 2D conv later on, we need to adjust the 2 dimensions 
            # to be convolutioned to the last 2 dimensions, by calling the permute() func.
            # Whereafter, to make the tensor contiguous in memory, call contiguous()
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()

            # 2D conv: from 1d Variation to 2d Variation
            # 2D convolution to grap the intra- and inter- period information
            out = self.conv(out)

            # reshape back
            # reshape back, similar to reshape
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            # truncating down the padded part of the output and put it to result
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim = -1)

        # adaptive aggregation
        # First, use softmax to get the normalized weight from amplitudes --> 2D [B,top_k]
        period_weight = F.softmax(period_weight, dim = 1)
        # after two unsqueeze(1),shape -> [B,1,1,top_k],so repeat the weight to fit the shape of res
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        # add by weight the top_k periods' result, getting the result of this TimesBlock
        res = torch.sum(res * period_weight, -1)

        # residual connection
        res = res + x

        return res


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        # params init
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        # stack TimesBlock for e_layers times to form the main part of TimesNet, named model
        self.model = nn.ModuleList([
            TimesBlock(configs) for _ in range(configs.e_layers)
        ])

        # embedding & normalization
        # * enc_in is the encoder input size, the number of features for a piece of data
        # * d_model is the dimension of embedding
        self.enc_embedding = DataEmbedding(
            configs.enc_in, 
            configs.d_model, 
            configs.embed, 
            configs.freq,
            configs.dropout
        )
        self.layer = configs.e_layers  # num of encoder layers
        self.layer_norm = nn.LayerNorm(configs.d_model)

        # define the some layers for different tasks
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias = True)
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias = True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim = True).detach()
        x_enc = x_enc - means
        
        stdev = torch.sqrt(torch.var(x_enc, dim = 1, keepdim = True, unbiased = False) + 1e-5)
        
        x_enc /= stdev
        
        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)  # align temporal dimension
        
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        
        # porject back
        dec_out = self.projection(enc_out)
        
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev
        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask = None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
