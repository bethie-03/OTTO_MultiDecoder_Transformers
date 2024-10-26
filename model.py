import torch
import torch.nn as nn
import math

class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) 
        self.bias = nn.Parameter(torch.zeros(features)) 

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True) 
        std = x.std(dim = -1, keepdim = True) 
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) 

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class EncoderInputEmbeddings(nn.Module):
    
    def __init__(self, d_model: int, aid_size: int, event_size: int = 6) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = aid_size
        self.embedding_aid = nn.Embedding(aid_size, d_model)
        self.embedding_etype = nn.Embedding(event_size, d_model)

    def forward(self, aid, event_type):
        return (self.embedding_aid(aid) + self.embedding_etype(event_type)) * math.sqrt(self.d_model)
    
class DecoderInputEmbeddings(nn.Module):
    
    def __init__(self, d_model: int, aid_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = aid_size
        self.embedding_aid = nn.Embedding(aid_size, d_model)

    def forward(self, aid):
        mask = aid != -1  
        aid_masked = torch.where(mask, aid, torch.zeros_like(aid)).long()
        embeddings = self.embedding_aid(aid_masked) * math.sqrt(self.d_model)
        embeddings = embeddings * mask.unsqueeze(-1).float()  
        summed_embeddings = embeddings.sum(dim=-2)  
        
        return summed_embeddings
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term) 
        pe = pe.unsqueeze(0) 
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) 
        return self.dropout(x)

class ResidualConnection(nn.Module):
    
        def __init__(self, features: int, dropout: float) -> None:
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.norm = LayerNormalization(features)
    
        def forward(self, x, sublayer):
            return x + self.dropout(sublayer(self.norm(x)))

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model 
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h 
        self.w_q = nn.Linear(d_model, d_model, bias=False) 
        self.w_k = nn.Linear(d_model, d_model, bias=False) 
        self.w_v = nn.Linear(d_model, d_model, bias=False) 
        self.w_o = nn.Linear(d_model, d_model, bias=False) 
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask.to(torch.float32) == 0, -1e4)
        attention_scores = attention_scores.softmax(dim=-1) 
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) 
        key = self.w_k(k)
        value = self.w_v(v) 
        #mask = mask.squeeze(0)
        
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
  
        return self.w_o(x)

class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderSelfAttentionBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.residual_connections = ResidualConnection(features, dropout)

    def forward(self, x, tgt_mask):
        x = self.residual_connections(x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        return x
    
class DecoderCrossAttentionBlock(nn.Module):

    def __init__(self, features: int, cross_attention_block: MultiHeadAttentionBlock, dropout: float) -> None:
        super().__init__()
        self.cross_attention_block = cross_attention_block
        self.residual_connections = ResidualConnection(features, dropout) 

    def forward(self, x, encoder_output, src_mask):
        x = self.residual_connections(x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        return x
    
class DecoderFeedForwardBlock(nn.Module):

    def __init__(self, features: int, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.feed_forward_block = feed_forward_block
        self.residual_connections = ResidualConnection(features, dropout)

    def forward(self, x):
        x = self.residual_connections(x, self.feed_forward_block)
        return x
    
class DecoderShareAttentionBlock(nn.Module):

    def __init__(self, features: int, share_attention_block: MultiHeadAttentionBlock, dropout: float) -> None:
        super().__init__()
        self.share_attention_block = share_attention_block
        self.residual_connections = ResidualConnection(features, dropout) 

    def forward(self, x, decoder_output, tgt_mask):
        x = self.residual_connections(x, lambda x: self.share_attention_block(x, decoder_output, decoder_output, tgt_mask))
        return x
    
class DecoderBlock(nn.Module):

    def __init__(self, 
                 decoder_self_attention_block: DecoderSelfAttentionBlock, 
                 decoder_cross_attention_block: DecoderCrossAttentionBlock, 
                 decoder_feed_forward_block: DecoderFeedForwardBlock, 
                 decoder_share_attention_block: DecoderShareAttentionBlock) -> None:
        super().__init__()
        self.decoder_self_attention_block = decoder_self_attention_block
        self.decoder_cross_attention_block = decoder_cross_attention_block
        self.decoder_feed_forward_block = decoder_feed_forward_block
        self.decoder_share_attention_block = decoder_share_attention_block

    def forward(self, x_0, x_1, x_2, encoder_output, src_mask, tgt_mask):
        x_0 = self.decoder_self_attention_block(x_0, tgt_mask)
        x_1 = self.decoder_self_attention_block(x_1, tgt_mask)
        x_2 = self.decoder_self_attention_block(x_2, tgt_mask)
        
        x_1_s_1 = self.decoder_share_attention_block(x_1, x_0, tgt_mask)
        x_2_s_1 = self.decoder_share_attention_block(x_2, x_1, tgt_mask)
        
        x_0 = self.decoder_cross_attention_block(x_0, encoder_output, src_mask)
        x_1 = self.decoder_cross_attention_block(x_1_s_1, encoder_output, src_mask)
        x_2 = self.decoder_cross_attention_block(x_2_s_1, encoder_output, src_mask)
        
        x_1_s_2 = self.decoder_share_attention_block(x_1, x_0, tgt_mask)
        x_2_s_2 = self.decoder_share_attention_block(x_2, x_1, tgt_mask)
        
        x_0 = self.decoder_feed_forward_block(x_0)
        x_1 = self.decoder_feed_forward_block(x_1_s_2)
        x_2 = self.decoder_feed_forward_block(x_2_s_2)
        
        x_1_s_3 = self.decoder_share_attention_block(x_1, x_0, tgt_mask)
        x_2_s_3 = self.decoder_share_attention_block(x_2, x_1, tgt_mask)
        
        return x_0, x_1_s_3, x_2_s_3
    
class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x_0, x_1, x_2, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x_0, x_1, x_2 = layer(x_0, x_1, x_2, encoder_output, src_mask, tgt_mask)
        return self.norm(x_0), self.norm(x_1), self.norm(x_2)

class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        return self.proj(x)
    
class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: EncoderInputEmbeddings, tgt_embed: DecoderInputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src_aid, src_type, src_mask):
        src = self.src_embed(src_aid, src_type)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self,
               tgt_0: torch.Tensor,
               tgt_1: torch.Tensor,
               tgt_2: torch.Tensor,
               encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor):
        
        tgt_0 = self.tgt_embed(tgt_0)
        tgt_0 = self.tgt_pos(tgt_0)
        
        tgt_1 = self.tgt_embed(tgt_1)
        tgt_1 = self.tgt_pos(tgt_1)
        
        tgt_2 = self.tgt_embed(tgt_2)
        tgt_2 = self.tgt_pos(tgt_2)
        return self.decoder(tgt_0, tgt_1, tgt_2 , encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int, N: int=3, h: int=8, dropout: float=0.5, d_ff: int=1024) -> Transformer:
    src_embed = EncoderInputEmbeddings(d_model, src_vocab_size)
    tgt_embed = DecoderInputEmbeddings(d_model, tgt_vocab_size)

    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    decoder_blocks = []
    for _ in range(N):
        self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        share_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        
        decoder_self_attention_block = DecoderSelfAttentionBlock(d_model, self_attention_block, dropout)
        decoder_cross_attention_block = DecoderCrossAttentionBlock(d_model, cross_attention_block, dropout)
        decoder_feed_forward_block = DecoderFeedForwardBlock(d_model, feed_forward_block, dropout)
        decoder_share_attention_block = DecoderShareAttentionBlock(d_model, share_attention_block, dropout)
        
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, decoder_feed_forward_block, decoder_share_attention_block)
        decoder_blocks.append(decoder_block)
    
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer
