# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from fairseq.models import (
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer_lm import transformer_lm_big
from fairseq.models.transformer_lm_sar import transformer_sar_lm_big,base_sar_lm_architecture


@register_model_architecture('transformer_lm', 'transformer_lm_ul')
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, 'decoder_layers', 16)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.0)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.1)
    args.no_decoder_final_norm = getattr(args, 'no_decoder_final_norm', False)
    args.tie_adaptive_proj = getattr(args, 'tie_adaptive_proj', True)
    transformer_lm_big(args)

@register_model_architecture('transformer_lm', 'transformer_lm_ul_base')
def transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, 'decoder_layers', 12)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 768)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 3072)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 12)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.0)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.1)
    args.no_decoder_final_norm = getattr(args, 'no_decoder_final_norm', False)
    args.tie_adaptive_proj = getattr(args, 'tie_adaptive_proj', True)
    transformer_lm_big(args)

#semi ar model
@register_model_architecture('transformer_sar_lm','transformer_sar_lm_ul')
def transformer_sar_lm_ul(args):
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.0)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.1)
    args.no_decoder_final_norm = getattr(args, 'no_decoder_final_norm', False)
    args.tie_adaptive_proj = getattr(args, 'tie_adaptive_proj', True)
    transformer_sar_lm_big(args)

@register_model_architecture('transformer_sar_lm', 'transformer_sar_lm_debug')
def transformer_sar_lm_debug(args):
    args.decoder_layers = getattr(args, 'decoder_layers', 3)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 128)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 128)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 2)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.0)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.1)
    args.no_decoder_final_norm = getattr(args, 'no_decoder_final_norm', False)
    args.tie_adaptive_proj = getattr(args, 'tie_adaptive_proj', True)
    base_sar_lm_architecture(args)#换一个小的模型和bathch size 
