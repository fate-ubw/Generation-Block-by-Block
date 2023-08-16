import torch
from fairseq.models.transformer import (
    TransformerDecoder,
)
import torch.nn.functional as F
from fairseq.modules import InsidechunkLearnedPositionalEmbedding
from fairseq.modules import InterchunkLearnedPositionalEmbedding


import pdb
class TransformerSarDecoder(TransformerDecoder):

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens,no_encoder_attn) 
        # define chunk positional embedding
        self.inside_chunk_position_embedding = InsidechunkLearnedPositionalEmbedding(args.chunk_size + embed_tokens.padding_idx + 1, args.decoder_input_dim, embed_tokens.padding_idx)
        self.inter_chunk_position_embedding = InterchunkLearnedPositionalEmbedding(args.max_num_chunk, args.decoder_input_dim, embed_tokens.padding_idx)

        
    def extract_features(self, prev_output_tokens, encoder_out=None, incremental_state=None, return_attn=False, src_attn_mask=None, incremental_update=True, **unused):
        """
        Similar to *forward* but only return features. 
        prev_output_tokens: 3d tensor [batch_size, num_chunk, chunk_size]
        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        # Note:masked matrix of semi-autoregresive

        #Note: mask matrix is different in training and inference stage
        if incremental_state is None:
            self_attn_mask = self.buffered_future_mask(prev_output_tokens) 
            self_attn_mask = self_attn_mask.cuda()
        else:
            # mask matrix in parallel inference 
            self_attn_mask = torch.tensor([], dtype=torch.int64)
            for i in range(int(prev_output_tokens.size(0))):
                mask = self.buffered_inference_mask(prev_output_tokens[i,:,:].unsqueeze_(0))
                self_attn_mask = torch.cat((self_attn_mask, mask.repeat(self.num_heads,1,1)),dim = 0) 
            self_attn_mask = self_attn_mask.cuda()

        # Note: inside chunk position embedding and inter chunk position embedding
        inside_chunk_positions = self.inside_chunk_position_embedding(prev_output_tokens)
        inter_chunk_positions = self.inter_chunk_position_embedding(prev_output_tokens)

        #Note: inference stage need part of the input data and position embedding
        if incremental_state is not None: 
            prev_output_tokens = prev_output_tokens[:, -1,:] 
            prev_output_tokens = prev_output_tokens.unsqueeze(dim=1)

            if inside_chunk_positions is not None:
                inside_chunk_positions = inside_chunk_positions[:, -int(prev_output_tokens.size(2)):,:]
            if inter_chunk_positions is not None:
                inter_chunk_positions = inter_chunk_positions[:,-int(prev_output_tokens.size(2)):,:]

        #Note: reshape the data: B X num_chunk X chunk_size --> B X T 
        prev_output_tokens = prev_output_tokens.reshape(prev_output_tokens.size(0),-1)
        # embed positions
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        if self.project_in_dim is not None:
            x = self.project_in_dim(x)
        # add inter chunk position embedding & inside chunk position embedding
        x += inside_chunk_positions
        x += inter_chunk_positions

        # if positions is not None:
        #     x += positions
        x = F.dropout(x, p=self.dropout, training=self.training) 

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn_lists = []
        inner_states = [x]

        # decoder layers
        for layer in self.layers:

            x, attn = layer( 
                x,
                encoder_out['encoder_out'] if encoder_out is not None else None,
                encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                incremental_update=incremental_update,
            )
            inner_states.append(x)
            attn_lists.append(attn)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        if return_attn:
            return x, {'attn': attn_lists, 'inner_states': inner_states} #
        else:
            return x, {'attn': None, 'inner_states': inner_states}

    def buffered_future_mask(self, tensor): # make noncausal mask matrix
        num_chunk = tensor.size(1)
        chunk_size = tensor.size(2)
        dim = num_chunk * chunk_size
        self._future_mask = torch.zeros(dim,dim)
        for i in range(num_chunk-1):
            self._future_mask[chunk_size * (i): chunk_size * (i+1), chunk_size * (i+1):] = float('-inf')
            chunk = tensor[0,i]
            chunk_flag_vector = chunk.ne(self.embed_tokens.padding_idx).int()
            chunk_len = int(chunk_flag_vector.sum())
            self._future_mask[chunk_size * i:, chunk_size * i + chunk_len: chunk_size * (i+1)] = float('-inf')# 全部的padding 都给-inf

        return self._future_mask[:dim, :dim]

    def buffered_inference_mask(self,tensor):
        num_chunk = tensor.size(1)
        chunk_size = tensor.size(2)
        dim = num_chunk * chunk_size
        self._future_mask = torch.zeros(chunk_size, dim)
        for i in range(num_chunk): 
            chunk = tensor[0,i]
            chunk_flag_vector = chunk.ne(self.embed_tokens.padding_idx).int()
            chunk_len = int(chunk_flag_vector.sum())
            self._future_mask[:,chunk_size * i + chunk_len: chunk_size * (i+1)] = float('-inf')
        return self._future_mask.unsqueeze_(0)