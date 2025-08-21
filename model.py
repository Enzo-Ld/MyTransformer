from torch import nn
import torch
import matplotlib.pyplot as plt

#inpinp shape：[batch,seq_len,d_model]
class PositionEncoding(nn.Module):
    def __init__(self,d_model,max_seq_len=512):
        super().__init__()
        #shape:[max_sql_len,1]
        position=torch.arange(0,max_seq_len).unsqueeze(1)
        
        item=1/10000**(torch.arange(0,d_model,2)/d_model)

        tmp_pos=position*item

        pe=torch.zeros(max_seq_len,d_model)

        pe[:,0::2]=torch.sin(tmp_pos)
        pe[:,1::2]=torch.cos(tmp_pos)

        # plt.matshow(pe)
        # plt.show()

        pe=pe.unsqueeze(0)
        #增加一个维度变为（batch_size,max_seq_len,d_model)
        self.register_buffer("pe",pe,False)

    def forward(self,x):
        batch,seq_len,_=x.shape
        pe=self.pe
        return x+pe[:,:seq_len,:]
    
def attention(query,key,value,mask=None):
    d_model=key.shape[-1]
    #query,key,value shape:[batch,head,seq_len,d_k]

    att=torch.matmul(query,key.transpose(-2,-1))/torch.tensor(d_model).sqrt()


    if mask is not None:
        att=att.masked_fill(mask,-1e9)
        #分为KeyPaddingMask与SeqMask
        #keyPaddingMask：序列位置为padding的位置的attention score设置为-1e9
        #batch的序列长度需要统一（为了方便GPU运算），句子序列长度不足的句子需要padding填充
        #SeqMask：是decoder中self-attention的mask
        # ：其主要作用是屏蔽未来，防止模型在训练的时候偷看未来信息

    att_score=torch.softmax(att,dim=-1)
    # att_score的shape为[batch_size,num_heads,max_seq_len_q,max_seq_len_k]
    out=torch.matmul(att_score,value)
    #value的shape为[batch_size,num_heads,max_seq_len_v,d_k]
    #out的shape为[batch_size,num_heads,max_seq_len_q,d_k]
    return out

class MultiHeadAttention(nn.Module):
    def __init__(self,heads,d_model,dropout):
        super().__init__()
        assert d_model%heads==0
        self.heads=heads
        self.dropout=nn.Dropout(dropout)
        self.d_k=d_model//heads
        self.d_model=d_model
        self.q_linear=nn.Linear(d_model,d_model,bias=False)
        self.k_linear=nn.Linear(d_model,d_model,bias=False)
        self.v_linear=nn.Linear(d_model,d_model,bias=False)
        self.out_linear=nn.Linear(d_model,d_model,bias=False)

    def forward(self,q,k,v,mask=None):
        #[n,seq_len,d_model] -> [n,heads,seq_len,d_k]
        q=self.q_linear(q).view(q.shape[0],-1,self.heads,self.d_k).transpose(1,2)
        k=self.k_linear(k).view(k.shape[0],-1,self.heads,self.d_k).transpose(1,2)
        v=self.v_linear(v).view(v.shape[0],-1,self.heads,self.d_k).transpose(1,2)
        out=attention(q,k,v,mask).transpose(1,2).contiguous().view(q.shape[0],-1,self.d_model)
        #attention(q,k,v,mask)传入x经WkWqWv映射后的q,k,v，与keypaddingmask，并返回attention后的结果
        #transpose(1,2).contiguous().view(q.shape[0],-1,self.d_model)
        #先换维度位置为([batch_size,seq_len,heads,d_k])再转成([batch_size,seq_len,d_model])合并多头
        #拼接后的 𝐻只是把不同 head 的信息“堆在一起”，还没有真正融合
        #每个 head 的信息在拼接后是固定顺序的，模型没法自由混合
        #经过加上 𝑊𝑂的线性映射，模型可以学到 复杂的跨头交互，提升表达能力
        out=self.out_linear(out)#Wo
        #Wo就是学习如何加权组合各个 head的输出
        return self.dropout(out)

class Feedforward(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.1):
        super().__init__()
        self.ffn=nn.Sequential(
            #传入的参数为mutiheadattention的输出.c
            nn.Linear(d_model,d_ff,bias=False),
            #shape[batch_size,max_seq_len,d_model]->shape[batch_size,max_seq_len,d_ff]
            nn.ReLU(),
            nn.Linear(d_ff,d_model,bias=False),
            #shape[batch_size,max_seq_len,d_ff]->shape[batch_size,max_seq_len,d_model]
            nn.Dropout(dropout)
        )

    def forward(self,x):
        return self.ffn(x)
class EncoderLayer(nn.Module):
    def __init__(self,heads,d_model,d_ff,dropout=0.1):
        super().__init__()
        self.multi_heads_att=MultiHeadAttention(heads,d_model,dropout)
        self.ffn=Feedforward(d_model,d_ff,dropout)
        self.norms=nn.ModuleList([nn.LayerNorm(d_model)for i in range(2)])
        self.dropout=nn.Dropout(dropout)

    def forward(self,x,mask=None):
        multi_heads_att_out=self.multi_heads_att(x,x,x,mask)
        #计算多头注意力：
        multi_heads_att_out=self.norms[0](x+multi_heads_att_out)#残差+层归一化
        ffn_out=self.ffn(multi_heads_att_out)#feedforwardNeuralNetwork前馈神经网络
        ffn_out=self.norms[1](multi_heads_att_out+ffn_out)
        #LayerNorm归一化是对每一个词token进行Norm（0均值，1方差）
        out=self.dropout(ffn_out)
        return out

class Encoder(nn.Module):
    def __init__(self,vocab_size,d_model,pad_idx,heads,d_ff,n_layers,dropout=0.1,max_seq_len=512):

        super().__init__()
        self.embedding=nn.Embedding(vocab_size,d_model,pad_idx)
        #词嵌入Embedding:在表中寻找句子中出现的词token索引
        #词嵌入 (Embedding): 将句子中的每个 token 索引映射为 d_model 维的向量表示。
        self.position_encode=PositionEncoding(d_model,max_seq_len)
        #位置编码PositionEncoding:为句子中的每个词添加一个位置编码向量，
        # 使得模型能够理解单词的相对位置关系。pos为句子中的位置索引，i为d_model的index。
        self.encoder_layers=nn.ModuleList([EncoderLayer(heads,d_model,d_ff,dropout)for i in range(n_layers)])
        #定义多个EncoderLayer

    def forward(self,x,src_mask=None):
        embed_X=self.embedding(x)
        pos_embed_x=self.position_encode(embed_X)#添加位置编码
        #添加完位置编码的pos_embed是原始输入x
        for layer in self.encoder_layers:
            pos_embed_x=layer(pos_embed_x,src_mask)
            #将上一层输出传入下一层 EncoderLayer
            # 每个 EncoderLayer 内部包含:
            # 1. Multi-Head Attention (带残差和 LayerNorm)
            # 2. 前馈网络 Feedforward (带残差和 LayerNorm)
        return pos_embed_x
        # 输出整个 Encoder 的表示，形状 (batch_size, seq_len, d_model)
        # 每个 token 都有一个 d_model 维的向量表示，包含了上下文信息


class DecoderLayer(nn.Module):
    def __init__(self, heads,d_model,d_ff,dropout=0.1):
        super().__init__()
        self.masked_att=MultiHeadAttention(heads,d_model,dropout)#带掩码的注意力
        self.att=MultiHeadAttention(heads,d_model,dropout)
        #cross attention（q：由decoder的上一层的masked attention输入）
        #k，v：以encoder的输出作为输入
        self.ffn=Feedforward(d_model,d_ff,dropout)
        self.norms=nn.ModuleList([nn.LayerNorm(d_model)for i in range(3)])
        self.dropout=nn.Dropout(dropout)

    def forward(self,x,dst_mask,src_dst_mask,encode_kv):
        mask_att_out=self.masked_att(x,x,x,dst_mask)
        #decoder的第一个att：Multi-Head masked attention
        mask_att_out=self.norms[0](x+mask_att_out)
        src_dst_att_out=self.att(mask_att_out,encode_kv,encode_kv,src_dst_mask)
        #decoder的第二个att：Multi-Head crossed attention
        src_dst_att_out=self.norms[1](mask_att_out+src_dst_att_out)
        ffn_out=self.ffn(src_dst_att_out)
        ffn_out=self.norms[2](src_dst_att_out+ffn_out)
        out=self.dropout(ffn_out)
        return out


class Decoder(nn.Module):
    def __init__(self,vocab_size,d_model,pad_idx,heads,d_ff,n_layers,dropout=0.1,max_seq_len=512):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,d_model,pad_idx)
        self.position_encode=PositionEncoding(d_model,max_seq_len)
        self.decoder_layers=nn.ModuleList([DecoderLayer(heads,d_model,d_ff,dropout)for i in range(n_layers)])
    
    def forward(self,x,encode_kv,dst_mask=None,src_dst_mask=None):
        #decoder的原始输入x
        dst_embedding=self.embedding(x)
        #对x进行Embedding
        pos_embed_dst=self.position_encode(dst_embedding)
        #加上pos位置信息
        for layer in self.decoder_layers:
            pos_embed_dst=layer(pos_embed_dst,dst_mask,src_dst_mask,encode_kv)
        return pos_embed_dst


class Transformer(nn.Module):
    def __init__(self,enc_vocab_size,dec_vocab_size,pad_idx,n_layers,heads,d_model,d_ff,dropout=0.1,max_seq_len=512):
        super().__init__()
        self.encoder=Encoder(enc_vocab_size,d_model,pad_idx,heads,d_ff,n_layers,dropout,max_seq_len)
        self.decoder=Decoder(dec_vocab_size,d_model,pad_idx,heads,d_ff,n_layers,dropout,max_seq_len)
        self.out_linear=nn.Linear(d_model,dec_vocab_size)
        self.pad_idx=pad_idx
    
    def generate_mask(self,query,key,is_triu_mask=False):
        #attention.shape=[batch,heads,seq_q,seq_k]
        #mask是作用于attention上的，所以其张量目标为：[batch, 1, seq_len_q, seq_len_k]
        #query shape:[batch,seq_len_q,d_model]

        device=query.device
        batch,seq_q=query.shape[:2]
        _,seq_k=key.shape[:2]
        #生成keypadding mask
        mask=(key==self.pad_idx).unsqueeze(1).unsqueeze(2)
        #虽然key.shape=[batch,seq_k,d_model]但是判断key==self.pad_idx时，
        #比较的时词索引序列，不比较d_model，即 [batch, seq_k]
        #所以mask生成的是[batch,seq_k]的true/false张量
        #经.unsqueeze(1).unsqueeze(2)扩展成[batch,1,1,seq_k]
        mask=mask.expand(batch,1,seq_q,seq_k).to(device)
        #.expand()复制为[batch,1,seq_q,seq_k]
        if is_triu_mask:#future mask：作用于decoder训练时，保证当前位置之后的词不会被观测到
            dst_triu_mask=torch.triu(torch.ones(seq_q,seq_k,dtype=torch.bool),diagonal=1)
            #torch.ones(seq_q,seq_k,dtype=torch.bool)创建一个bool型矩阵的全为true的[seq_q,seq_k]矩阵
            #torch.triu(,diagonal=1) ; torch.tril(input, diagonal=0)  # 返回下三角部分
            # torch.triu()返回输入矩阵的 ​​上三角部分​,其余位置置零
            # diagonal参数决定 ​​从哪条对角线开始保留值(0:主对角线；-1：主对角线下一条；1：主对角线上一条)
            dst_triu_mask=dst_triu_mask.unsqueeze(0).unsqueeze(1).expand(batch,1,seq_q,seq_k).to(device)
            return mask|dst_triu_mask
            #decoder合并keyPadding mask和future mask
        return mask

    def forward(self,src,dst):
        src_mask=self.generate_mask(src,src)
        encoder_out=self.encoder(src,src_mask)
        dst_mask=self.generate_mask(dst,dst,is_triu_mask=True)
        src_dst_mask=self.generate_mask(dst,src)
        #cross attention中的q来自encoder，k来自decoder所以其seq_len不一定一致
        decoder_out=self.decoder(dst,encoder_out,dst_mask,src_dst_mask)
        out =self.out_linear(decoder_out)
        return out


    

if __name__=="__main__":
    #PositionEncoding(512,100)
    att = Transformer(100,120,0,6,8,512,1024)
    x=torch.randint(0,100,(5,64))
    y=torch.randint(0,120,(5,64))
    out=att(x,y)
    print(out.shape)
