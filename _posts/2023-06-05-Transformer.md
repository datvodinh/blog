---
title: Transformer
date: 2023-06-05 23:00:00 +0700
categories: [Machine Learning, AI]
tags: [ml,ai,llm]     # TAG names should always be lowercase
---

# Transformer
We will explore the power of the Transformer algorithm, the driving force behind the remarkable success of Large Language Models. Additionally, I will take you on a journey of building this algorithm from the ground up, providing you with a comprehensive understanding of its inner workings.

![Alt Text](https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_1.png)

--- 

## Attention

```python
class MultiHeadAttention(nn.Module):
    def __init__(self,embed_size,heads,bias=False):
        super(MultiHeadAttention,self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.heads_dim = int(embed_size / heads)
        self.keys = nn.Linear(embed_size,embed_size,bias=bias)
        self.queries = nn.Linear(embed_size,embed_size,bias=bias)
        self.values = nn.Linear(embed_size,embed_size,bias=bias)
        self.fc = nn.Linear(embed_size,embed_size,bias=bias)

    def forward(self,key,query,value,mask=None):

        # key shape: (batch_size,key_len,embed_size)
        # query shape: (batch_size,query_len,embed_size)
        # value shape: (batch_size,value_len,embed_size)
        # key and query and value all have the same shape
 

        keys = self.keys(key).reshape(key.shape[0],key.shape[1],self.heads,self.heads_dim)
        queries = self.queries(query).reshape(query.shape[0],query.shape[1],self.heads,self.heads_dim)
        values = self.values(value).reshape(value.shape[0],value.shape[1],self.heads,self.heads_dim)

        # keys shape: (batch_size,key_len,heads,head_dim)
        # queries shape: (batch_size,query_len,heads,head_dim)
        # values shape: (batch_size,value_len,heads,head_dim)
        keys = keys / (self.embed_size)**(1/4)
        queries = queries / (self.embed_size)**(1/4)
        
        dot_product = torch.einsum('bkhd,bqhd->bhqk',keys,queries)
        
        # dot_product shape: (batch_size,heads,query_len,key_len)
        if mask is not None:
            dot_product = dot_product.masked_fill(mask==0,float('-inf'))

        scaled_product = torch.softmax(dot_product ,dim=3)

        alpha = torch.einsum("bhqk,bvhd->bqhd",scaled_product,values)
        out = self.fc(alpha.reshape(key.shape[0],key.shape[1],self.embed_size))

        return out
```

## Encoder

```python
class EncoderBlock(nn.Module):
    def __init__(self,embed_size,heads,bias=False):
        super(EncoderBlock,self).__init__()
        self.attention    = MultiHeadAttention(embed_size,heads,bias)
        self.layer_norm1  = nn.LayerNorm(embed_size)
        self.layer_norm2  = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size,4*embed_size),
            nn.GELU(),
            nn.Linear(4*embed_size,embed_size)
        )

    def forward(self,key,query,value,mask=None):
        attention = self.attention(key,query,value,mask)
        out = self.layer_norm1(key + attention)
        out_ffw = self.feed_forward(out)
        out = self.layer_norm2(out + out_ffw)

        return out

class Encoder(nn.Module):
    def __init__(self,vocab_size,embed_size,heads,num_layers,max_len,dropout,bias=False):
        super(Encoder,self).__init__()
        self.embed = nn.Embedding(vocab_size,embed_size)
        self.position_embed = PositionalEncoding(embed_size,max_len=max_len,dropout=dropout)
        self.encoder_layers = nn.ModuleList(
            [
                EncoderBlock(embed_size,heads,bias)
                for _ in range(num_layers)
            ]

        )

        self.dropout = nn.Dropout(dropout)

    def forward(self,x,mask):
        x_embed = self.embed(x)
        x_embed = self.position_embed(x_embed)
        out = self.dropout(x_embed)
        for layer in self.encoder_layers:
            out = layer(out,out,out,mask)
    
        return out
```

## Decoder

```python
class DecoderBlock(nn.Module):
    def __init__(self,embed_size,heads,bias=False):
        super(DecoderBlock,self).__init__()
        self.encoder_block = EncoderBlock(embed_size,heads,bias)
        self.attention = MultiHeadAttention(embed_size,heads,bias)
        self.layer_norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout()

    def forward(self,x,enc_value,enc_key,src_mask,target_mask):
        out = self.layer_norm(x + self.attention(x,x,x,src_mask))
        out = self.dropout(out)
        out = self.encoder_block(key=enc_key,value=enc_value,query=out,mask=target_mask)

        return out


class Decoder(nn.Module):
    def __init__(self,vocab_size,embed_size,heads,num_layers,max_len,dropout,bias=False):
        super(Decoder,self).__init__()
        self.embed = nn.Embedding(vocab_size,embed_size)
        self.position_embed = PositionalEncoding(embed_size,max_len=max_len,dropout=dropout)
        self.decoder_layer = nn.ModuleList(
            [
                DecoderBlock(embed_size,heads,bias)
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_size,vocab_size)

    def forward(self,x,encoder_out,src_mask,target_mask):
        x_embed = self.embed(x)
        x_embed = self.position_embed(x_embed)
        out = self.dropout(x_embed)
        for layer in self.decoder_layer:
            out = layer(out,encoder_out,encoder_out,src_mask,target_mask)
        out = self.fc(out)

        return out
```