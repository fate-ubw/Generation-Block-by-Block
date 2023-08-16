import torch
from fairseq import checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.meters import StopwatchMeter
from fairseq.custom.metrics import Metrics, TrainingMetrics
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from fairseq.utils import fill_with_neg_inf

import math
import pdb


def load(args, task=None, itr=None, generator=None, log=False, random_net_init=False, re_init_weight=None):
    """Returns task, model, generator, and dataset iterator for the given `args`."""
    assert args.path is not None, '--path required for generation!'
    import random
    random.seed(42)
    torch.manual_seed(42)
    utils.import_user_module(args)
    if log:
        print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    if task is None:
        task = tasks.setup_task(args)
        task.load_dataset(args.gen_subset)

    # Load ensemble
    if log:
        print('| loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(':'),
        arg_overrides=eval(args.model_overrides),
        task=task,
        random_net_init=random_net_init,
        re_init_weight=re_init_weight,
    )

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()
    model = models[0]

    if itr is None:
        # Load dataset (possibly sharded)
        itr = task.get_batch_iterator(
            dataset=task.dataset(args.gen_subset),
            max_tokens=args.max_tokens,
            max_sentences=args.max_sentences,
            max_positions=args.tokens_per_sample,
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            num_shards=args.num_shards,
            shard_id=args.shard_id,
            num_workers=args.num_workers,
        ).next_epoch_itr(shuffle=False)

    # Get model step
    step = torch.load(args.path)['optimizer_history'][-1]['num_updates']

    if generator is None:
        # Initialize generator
        generator = task.build_generator(args)
    return task, model, generator, itr, step

# Note: rewrite generate_completions 
def generate_completions_sar(model, generator, fairseq_generator, itr, eval_prefix_length, eval_completion_length, topk, topp, num_samples, beam_size, chunk_size, n_gram_block=0, include_prefix=True, two_step_gen=False):
    completions = []
    gt_completion = []
    completion_metrics = Metrics()
    actual_metrics = Metrics()
    prefix =[]
    aligned_target = []
    padding = [5,4,1]
    #Note: prepare all prefix data 
    for n, sample in enumerate(tqdm(itr)):
        input_sequence = sample['net_input']['src_tokens']
        assert input_sequence.dim() == 3
        chunk_size = input_sequence.size(2)
        prefix_batch, aligned_target_batch = make_sar_chunked_data_prefix_aligned_target(input_sequence.reshape(1,-1), eval_prefix_length, eval_completion_length, padding, chunk_size)
        #pdb.set_trace()
        prefix.extend(prefix_batch)
        aligned_target.extend(aligned_target_batch)
        input_of_actual_metrics = input_sequence.reshape(input_sequence.size(0),-1).tolist()[0]
        while 4 in input_of_actual_metrics:
            input_of_actual_metrics.remove(4)
        while 5 in input_of_actual_metrics:
            input_of_actual_metrics.remove(5)
        while 1 in input_of_actual_metrics:
            input_of_actual_metrics.remove(1)
        actual_metrics.update(torch.tensor(input_of_actual_metrics).unsqueeze(dim=0))
        if n == 1:
            break
    # generation code block 
    assert len(prefix) == len(aligned_target)
    for prefix_sample, target_sample in tqdm(zip(prefix, aligned_target), total=len(prefix)):
        # this is the place of calculate the generation of text
        # prefix_sample should be a tensor([1,sample_len])
        # restore data into complete chunk and reshape into [num_chunk, chunk_size]
        # Note: not every prefix is helf chunk, some prefix is completed tokens
        if len(prefix_sample) % chunk_size != 0:
            prefix_sample = padding_prefix_to_complete_chunk(prefix_sample, chunk_size)
        prefix_batch = prefix_sample.cuda().unsqueeze(dim=0)
        # make tenser.shape = [1,sample length]
        if beam_size > 1:
            assert topk == 1, 'with greedy topk must be 1'
            assert topp == 0.0, 'with greedy topp must be 0'
            sample['net_input']['src_tokens'] = prefix_batch
            res = fairseq_generator.generate([model], sample, prefix_batch, bos_token=0)  # prefix is there in preds!
            pred_completion = [res[i][0]['tokens'][eval_prefix_length:-1].cpu().tolist() for i in range(len(res))]
        elif beam_size == 1:
            if n_gram_block > 0:
                assert n_gram_block > 1, 'at least >= 2'
                pred_completion = generator.generate_completion_block(model, n_gram_block, prefix_batch, eval_completion_length, topk, topp)
                pred_completion = pred_completion.cpu().tolist()
            elif two_step_gen > 0:
                pred_completion = generator.generate_tsg_completion(model, prefix_batch, eval_completion_length, two_step_gen)
                pred_completion = pred_completion.cpu().tolist()
            else:
                #prefix shape is [1,sample length]
                pred_completion = generator.generate_completion(model, prefix_batch, eval_completion_length, topk, topp, chunk_size, padding)
                # pdb.set_trace()
                assert count_valid_tokens(pred_completion[0,:], padding) == eval_completion_length
                pred_completion = pred_completion.cpu().tolist()
        generate_list = []
        generate_texts = pred_completion
        for generate_text in generate_texts:
            while 4 in generate_text:
                generate_text.remove(4)
            while 5 in generate_text:
                generate_text.remove(5)
            while 1 in generate_text:
                generate_text.remove(1)
            generate_list.append(generate_text)
        completion_metrics.update(generate_list) 
        # actual_metrics calculate in the prepare looping block
        if include_prefix:
            completions.append(prefix_sample.tolist() + pred_completion[0])
            gt_completion.append(prefix_sample.tolist() + target_sample.tolist())
    completion_metrics = completion_metrics.report('generated')
    actual_metrics = actual_metrics.report('actual')
    return completions, gt_completion, completion_metrics, actual_metrics

def count_valid_tokens(input_tensor, valid_pad):
    '''
    input_tensor: 1d or 2d tensor 
    '''
    valid_num = 0
    size = input_tensor.dim()
    if size == 1:
        return sum([(token not in valid_pad) for token in input_tensor])
    elif size == 2:
        for value in input_tensor:
            num = sum([(token not in valid_pad) for token in value])
            valid_num += num
        return valid_num

def make_sar_chunked_data_prefix_aligned_target(input_sequence, prefix_length, eval_completion_length, valid_pad, chunk_size):
    # Warning: The function can only be used for batchsize = 1 sentence.
    """"
    get prefix and aligned target for sar model generation evaluation

    Parameters:
    param1:input_sequence - 2d tensor [1, sample_length] this function only operate batch_size==1 input
    param2 - this is a second param
    
    Returns:
    param1:prefix - (list of tensor with 1 dimension) each prefix is begin with a completed chunk but end with half chunk 
    param2:target_tokens - (list of tensor with 1 dimension) each target is begin with the next token of prefix

    Raises:
    assert dimension mismatch
    """
    assert input_sequence.dim() == 2
    seq_len = len(input_sequence[0, :])
    prefix = []
    target_tokens = []
    last_prefix_index = 0 

    while count_valid_tokens(input_sequence[0, last_prefix_index:], valid_pad) > prefix_length + eval_completion_length: 
        signle_prefix = input_sequence[0, last_prefix_index: last_prefix_index + prefix_length]
        if count_valid_tokens(signle_prefix,valid_pad) == prefix_length:
            # In this situation the valid tokens of signle_prefix is directly equal to prefix_length
            last_prefix_index += len(signle_prefix)
            prefix.append(signle_prefix)
        else :
            buffer_start_idx = last_prefix_index + prefix_length
            for i in range(buffer_start_idx, seq_len, prefix_length):
                buffer = input_sequence[0, i: i + prefix_length] 
                # calculate the length of buffer and current prefix 
                num_valid_signle_prefix = count_valid_tokens(signle_prefix, valid_pad)
                lack_valid_token_length = prefix_length - num_valid_signle_prefix
                num_valid_buffer = count_valid_tokens(buffer, valid_pad)
                if lack_valid_token_length > num_valid_buffer:
                    # this situation indicate that this turn is not the last turn of the cating buffer. So directly cat the buffer after current prefix
                    signle_prefix = torch.cat((signle_prefix, buffer),dim=0)
                    num_valid_signle_prefix = num_valid_signle_prefix + num_valid_buffer
                elif lack_valid_token_length < num_valid_buffer:
                    # In this situation buffer is the last buffer which means we have to get part of the bufffer cating after prefix making correct valid length 
                    #Note: here is different to ar chunk, because code need complete chunk not helf chunk
                    # ar chunk model do not have this question
                    last_buffer = get_last_buffer(buffer, valid_pad, lack_valid_token_length)
                    signle_prefix = torch.cat((signle_prefix, last_buffer))    
                    break
                elif lack_valid_token_length == num_valid_buffer:
                    #in this rare situation directly cat prefix and buffer and then you will get the right result
                    signle_prefix = torch.cat((signle_prefix, buffer))
                    break
            last_prefix_index += len(signle_prefix)
            prefix.append(signle_prefix)
        # get the target of the prefix
        # the logic of getting each gt com is the same as prefix

        start_taregt_idx = last_prefix_index # get target_idx 
        last_prefix_index = find_next_multiple(last_prefix_index, chunk_size) # get next prefix start index
        signle_target = input_sequence[0, start_taregt_idx: start_taregt_idx + eval_completion_length]
        # if the number of valid tokens in signle_target equal to eval_completion_length
        # then jump into the next loop
        if count_valid_tokens(signle_target, valid_pad) == eval_completion_length:
            target_tokens.append(signle_target)
            continue
        target_buffer_idx = start_taregt_idx + eval_completion_length 
        for i in range(target_buffer_idx, seq_len, eval_completion_length):
            target_buffer = input_sequence[0, i: i + eval_completion_length]
            num_valid_signle_target = count_valid_tokens(signle_target, valid_pad)
            lack_valid_target_length = eval_completion_length - num_valid_signle_target 
            num_valid_target_buffer  = count_valid_tokens(target_buffer, valid_pad)
            if lack_valid_target_length > num_valid_target_buffer:
                signle_target = torch.cat((signle_target,target_buffer), dim=0)
                num_valid_signle_target += num_valid_target_buffer # update the number of valid tokens in signle target
            elif lack_valid_target_length < num_valid_target_buffer:
                last_target_buffer = get_last_buffer(target_buffer, valid_pad, lack_valid_target_length)
                signle_target = torch.cat((signle_target, last_target_buffer))
                break
            elif lack_valid_target_length == num_valid_target_buffer:
                signle_target = torch.cat((signle_target, target_buffer))
                break
        target_tokens.append(signle_target)
    return prefix, target_tokens

def get_last_buffer(input_tensor, valid_pad, length):
    """   
    get number of length from input_tensor 
    
    Parameters:
    input_tensor - tensor with 1 dimension example: tensor([1,2,3,4,5])
    valid_pad - this is the mask tokens in input (list example: [4,5])
    length - (int) target valid length of output

    Returns:
    return a tensor with 1 dimension
    
    """
    mask = torch.Tensor([(token not in valid_pad) for token in input_tensor])
    cumsum = torch.cumsum(mask, dim=0)
    last_valid_token_idx = torch.where(cumsum == length)[0][0] if 1 in cumsum else 0
    return input_tensor[:last_valid_token_idx+1]

def find_next_multiple(n, div_number):
    """
    the target is to calculate the next start index of prefix
    
    Parameters:
    n - end index of last prefix (int)
    div_number - chunk_size (int)
    
    Returns:
    start index of next perfix
    """
    
    remainder = n % div_number
    if remainder == 0:
        return n
    else:
        return n + (div_number - remainder)

def get_last_buffer_in_entire_chunk(input_tensor, valid_pad, target_length):
    """ 
    this function is designed for sar model

    Parameters:
    input_tensor - 2d tensor let code get complete chunk easily
    valid_pad - mask tokens (list)
    target_length - chunk_size
    
    Returns:
    return the first complete chunk from input(tensor)
    """

    current_valid_len = 0
    last_chunk = torch.tensor([ ], dtype= torch.int64)
    for chunk in input_tensor:
        if current_valid_len < target_length:
            last_chunk = torch.cat((last_chunk, chunk)) # 
            current_valid_len += count_valid_tokens(chunk, valid_pad)
        else:
            break
    return last_chunk

def padding_prefix_to_complete_chunk(input_data, padding_length):
    """
    pad the last chunk to a complete chunk in input data (example: tensor([5,6])->tensor([5,6,7,4,1]))
    
    Parameters:
    input_data - one prefix data with target valid length (tensor with 1 dimension:tensor[len])
    padding_length - chunk size (int)
    
    Returns:
    return the the padded prefix whit complete last chunk (tensor with 1 dimension)
    """
    chunk_e_idx = torch.tensor([4])
    pad_idx = torch.tensor([1])
    make_up_data = torch.cat((input_data, chunk_e_idx)) # pad <chunk_e> for prefix first
    seq_len = len(make_up_data)
    remain_num = seq_len % padding_length # check the current length 
    if remain_num == 0:
        return make_up_data
    elif remain_num > 0:
        make_up_data = torch.cat((make_up_data, pad_idx.repeat(padding_length - remain_num)))
        return make_up_data
