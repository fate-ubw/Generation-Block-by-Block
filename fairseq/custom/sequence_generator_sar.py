import math

import torch
import torch.nn.functional as F
from datetime import datetime
import numpy as np
from fairseq.utils import fill_with_neg_inf
from fairseq.custom.evaluate_utils_chunk_sar import count_valid_tokens, get_last_buffer
from fairseq.custom.sequence_generator import  SequenceGenerator
import pdb

class SarSequenceGenerator(SequenceGenerator):
    def __init__(self, tgt_dict, temperature=1.):
        super().__init__(tgt_dict,temperature)

    @torch.no_grad()
    def generate_completion(self, model, prefix_tokens, completion_length, topk, topp, chunk_size, valid_pad):
        """topk: <1 sampling, 1 greedy, >1 top-k sampling."""
        '''
        prefix_tokens: 2d tensor [batch_size, sample_length]
        '''
        model.eval()
        pred_toks = []
        context = prefix_tokens
        states = {}
        valid_generated_length = 0
        lack_num_valid_tokens = completion_length
        padding_idx = 1
        # First go over the context.
        for context_step in range(0, int(context.size(1)/chunk_size)-1):
            _context = context[:, :context_step * chunk_size + chunk_size]
        # interchunk need all data not a signle chunk so I have to input all _context into model
            _context = _context.reshape(_context.size(0),-1,chunk_size)
            _ = self._forward_one(model, _context,chunk_size, incremental_states=states, return_logits=True)

        while lack_num_valid_tokens > 0:
            context = context.reshape(context.size(0),-1 ,chunk_size)# 
            logits, attn_t = self._forward_one(model, context, chunk_size, incremental_states=states, return_logits=True)
            pred_tok = self._topk_decode(logits, topk, topp)

            # change surplus <chunk_e> into <pad> making generated text same as train data
            pred_chunk = pred_tok[0, :].tolist() 
            if 4 in pred_chunk:
                pred_chunk = pred_chunk[:pred_chunk.index(4) + 1]  # Include the <chunk_e> token
            # Padding the incomplete chunk to the desired chunk size
            padding_length = chunk_size - len(pred_chunk)
            pred_chunk += [padding_idx] * padding_length
            # Reshape the padded chunk and concatenate it to the context
            pred_chunk = torch.tensor(pred_chunk).cuda().unsqueeze(0)

            context = context.reshape(context.size(0),-1)
            context = torch.cat((context, pred_chunk), 1)

            context_length = count_valid_tokens(pred_chunk, valid_pad)
            lack_num_valid_tokens = completion_length - valid_generated_length
            if lack_num_valid_tokens > context_length:
                #situation1: need lots of tokens  
                pred_toks.append(pred_chunk) 
                valid_generated_length += context_length
            elif lack_num_valid_tokens < context_length:
                pred_chunk = get_last_buffer(pred_chunk[0,:], valid_pad, lack_num_valid_tokens)
                pred_toks.append(pred_chunk.unsqueeze(dim=0)) 
                break
            elif lack_num_valid_tokens == context_length:
                pred_toks.append(pred_chunk) 
                break
        pred_toks = torch.cat(pred_toks, 1) 
        return pred_toks

    def _forward_one(self, model, tokens, chunk_size, incremental_states=None, temperature=1., return_attn=False, return_logits=False, **decoder_kwargs):
        if incremental_states is not None:
            decoder_out = list(model.decoder(tokens, None, incremental_state=incremental_states, return_attn=return_attn, **decoder_kwargs))
        else:
            decoder_out = list(model.decoder(tokens, None, return_attn=return_attn, **decoder_kwargs)) 
        decoder_out[0] = decoder_out[0][:, -chunk_size:, :] 
        if temperature != 1.:
            decoder_out[0].div_(temperature) 
        attn = decoder_out[1]
        if type(attn) is dict:
            attn = attn['attn']
        # if attn is not None:
        #     if type(attn) is dict:
        #         attn = attn['attn']
        #     attn = attn[:, :, -1, :]  # B x L x t
        if return_logits:
            logits_t = decoder_out[0][:, -chunk_size:, :] 
            return logits_t, attn
        log_probs = model.get_normalized_probs(decoder_out, log_probs=True)
        log_probs = log_probs[:, -chunk_size:, :]
        return log_probs, attn

    def _topk_decode(self, logits, topk, topp, return_prob=False):
        """WARNING!!! This can modify the `self.pad` position of `logits`."""

        if topk == 1 and topp == 0:  # greedy
            # the data is 3d structure
            logits[:, :, self.pad] = -math.inf  
            pred_tok = logits.argmax(dim=2)
        else:
            batch_size, chunk_size, vocab_size = logits.shape  
            if topk > 1:
                logits[:,:, self.pad] = -1e10  # never select pad
                logits = self.top_k_logits(logits, topk)
                pred_tok = torch.softmax(logits, -1).multinomial(1)
            else:
                assert topp > 0.0
                logits = logits.view(batch_size * chunk_size, vocab_size)
                filtered_probs, bookkeep_idx = self._sample_topp(torch.softmax(logits, -1), sampling_topp=topp)
                selected = filtered_probs.multinomial(1)
                pred_tok = torch.gather(bookkeep_idx, index=selected, dim=-1)
                pred_tok = torch.reshape(pred_tok, (batch_size, chunk_size))
        if return_prob:
            return pred_tok, torch.gather(torch.softmax(logits, -1), index=pred_tok, dim=-1)
        return pred_tok
