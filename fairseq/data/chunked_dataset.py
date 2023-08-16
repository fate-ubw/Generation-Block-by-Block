import torch
import pdb
from . import MonolingualDataset
from fairseq.data import data_utils

def collate(samples, pad_idx, eos_idx,chunklength):

    if len(samples) == 0:
        return {}

    def merge(key, is_list=False): #key :source & target
        if is_list: # If is_list is True, merge will merge a list of tokens instead of a single token.
            res = []
            for i in range(len(samples[0][key])):
                res.append(data_utils.collate_tokens(
                    [s[key][i].reshape(-1) for s in samples], pad_idx, eos_idx, left_pad=False,
                ))
            return res
        else:
            return data_utils.collate_tokens(
                [s[key].reshape(-1) for s in samples], pad_idx, eos_idx, left_pad=False,
            )
    src_tokens = merge('source')
    src_tokens = src_tokens.reshape(len(samples),-1,chunklength)
    src_input_tokens = src_tokens[:,0:-1,:]
    target = src_tokens[:,1:,:]

    return {
        'id': torch.LongTensor([s['id'] for s in samples]),
        'nsentences': len(samples),
        'ntokens': sum(len(s['source'].reshape(-1)) for s in samples),
        'net_input': {
            'src_tokens': src_input_tokens,
            'src_lengths': torch.LongTensor([
                s['source'].numel() for s in samples
            ]),
        },
        'target': target,
    }

class ChunkedDataset(MonolingualDataset):

    def __init__(self,dataset, sizes, src_vocab, tgt_vocab, max_chunklength, add_eos_for_other_targets, shuffle, targets=None, add_bos_token=False ,restore_way = 'Stay_half_chunk',chunk_option= 'AR_insingleword'):
        # pdb.set_trace()
        super().__init__(dataset, sizes, src_vocab, tgt_vocab, add_eos_for_other_targets, shuffle, targets=None, add_bos_token=False)
        self.fixed_pad_length = None
        self.pad_to_bsz = None
        self.max_chunklength = max_chunklength
        self.restore_way = restore_way
        self.chunk_option = chunk_option

    def __getitem__(self, index):

        if self.targets is not None:
            source, future_target, past_target = self.dataset[index]
            source, target = self._make_source_target(source, future_target, past_target)
        else:  
            source = self.dataset[index]
            target = None
        source, target = self._maybe_add_bos(source, target)  

        source = source[1] 
        RestoredData = self._recover_chunk(source, self.restore_way) 
        map_index = self._map_tokens(RestoredData, self.chunk_option) 
        chunked_data = self._get_datainchunk(RestoredData, map_index)
        chunked_source = self._padding_chunked_data(chunked_data, self.max_chunklength)
        return {'id': index, 'source': chunked_source[0:-1], 'target': chunked_source[1:]}

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:
                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the right.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the right.
        """
        return collate(samples, self.vocab.pad(), self.vocab.eos(),self.max_chunklength)

    def _recover_chunk(self, sample, restore_way):
        if restore_way not in ['Restore_half_chunk', 'Stay_half_chunk']:
            restore_way = 'Stay_half_chunk'

        last_index_s = -1
        last_index_e = -1
        first_s = None
        final_e = None
        helf_chunk_s = None
        helf_chunk_e = None
        cur_index_e = None
        cur_index_s = None

        try:
            cur_index_s = sample.tolist().index(5) # Todo：index number depend on parsing feature(<chunk_s> <chunk_e>)
        except ValueError as e:
            pass
        try:
            cur_index_e = sample.tolist().index(4)
        except ValueError as e:
            pass

        if cur_index_e == None and cur_index_s == None:
            #class 1
            return sample
        elif cur_index_e == None and cur_index_s != None:
            #class 2
            if restore_way == 'Stay_half_chunk':
                sample = torch.cat((sample,torch.tensor([4])))
                return sample
            elif restore_way == 'Restore_half_chunk':
                return sample
        elif cur_index_e != None and cur_index_s == None:
            #class 5
            if restore_way == 'Stay_half_chunk':
                sample = torch.cat((torch.tensor([5]),sample))
                return sample
            elif restore_way == 'Restore_half_chunk': 
                return sample

        try:
            cur_index_s = sample.tolist().index(5, last_index_e + 1)
        except ValueError as e:
            pass
        try:
            cur_index_e = sample.tolist().index(4, cur_index_s)
        except ValueError as e:
            pass
        first_s = cur_index_s

        while cur_index_e < len(sample.tolist()):
            try:
                last_index_e = cur_index_e
                cur_index_e = sample.tolist().index(4, last_index_e + 1)
            except ValueError as e:
                break
        final_e = last_index_e
        try:
            helf_chunk_e = sample.tolist().index(4,0,first_s)
        except ValueError as error:
            pass
        try:
            helf_chunk_s = sample.tolist().index(5,final_e+1) 
        except ValueError as error:
            pass

        if helf_chunk_e == None and first_s != None and final_e != None and helf_chunk_s == None:
            #class 3
            return sample
        elif helf_chunk_e == None and first_s != None and final_e != None and helf_chunk_s != None:
            #class 4
            if restore_way == 'Stay_half_chunk':
                sample = torch.cat((sample,torch.tensor([4])))
                return sample
            elif restore_way == 'Restore_half_chunk':
                return sample
        elif helf_chunk_e != None and first_s != None and final_e != None and helf_chunk_s == None:
            # class 7
            if restore_way == 'Stay_half_chunk':
                sample = torch.cat((torch.tensor([5]),sample))
                return sample
            elif restore_way == 'Restore_half_chunk':
                return sample
        elif helf_chunk_e != None and first_s != None and final_e != None and helf_chunk_s != None:
            # class 8
            if restore_way == 'Stay_half_chunk':
                sample = torch.cat((torch.tensor([5]),sample,torch.tensor([4])))
                return sample
            elif restore_way == 'Restore_half_chunk':
                return sample

    def _map_tokens(self, mapped_data, chunk_option='AR_insingleword'):  #

        if chunk_option not in ['AR_insingleword', 'NAR_insingleword']:
            chunk_option = 'NAR_insingleword'
        last_index_s = -1
        last_index_e = -1
        chunk_data_index = []
        while chunk_data_index == [] or chunk_data_index[-1] != len(mapped_data)-1:
            try:
                cur_index_s = mapped_data.tolist().index(5,last_index_e + 1)
                cur_index_e = mapped_data.tolist().index(4,cur_index_s + 1)
                if cur_index_s == 0:
                    chunk_data_index += [cur_index_s,cur_index_e]
                elif cur_index_s > 0 and cur_index_s-last_index_e >1:
                    if chunk_option == 'AR_insingleword': 
                        for index in range(last_index_e+1,cur_index_s):
                            SingleWord_index_s = index
                            SingleWord_index_e = index
                            chunk_data_index += [SingleWord_index_s,SingleWord_index_e]
                    elif chunk_option == 'NAR_insingleword':
                        chunk_data_index += [last_index_e+1,cur_index_s-1]
                    chunk_data_index += [cur_index_s,cur_index_e]
                elif cur_index_s -last_index_e == 1:
                    chunk_data_index += [cur_index_s,cur_index_e]
                last_index_s = cur_index_s
                last_index_e = cur_index_e
            except ValueError as e:
                if chunk_option == 'AR_insingleword':
                    for i in range(last_index_e + 1, len(mapped_data)):  # ---> 1:fix
                        chunk_data_index.extend([i])
                        chunk_data_index.extend([i])
                elif chunk_option == 'NAR_insingleword':
                    chunk_data_index += [last_index_e + 1, len(mapped_data) - 1]
        return chunk_data_index

    def _get_datainchunk(self,tokens, mapping_index):
        # pdb.set_trace()
        chunked_data = []
        for start in range(0, len(mapping_index), 2):
            chunked_data.append(tokens[mapping_index[start]:mapping_index[start + 1] + 1])
        return chunked_data

    def _padding_chunked_data(self,inputdata,num_chunk):
        chunk_data_2d = torch.tensor([[]],dtype=torch.int8)
        # pdb.set_trace()
        for chunk in inputdata:
            if int(chunk[0]) == 5 and int(chunk[-1]) == 4:
                pass
            elif int(chunk[0]) == 5 and int(chunk[-1]) != 4:
                chunk = torch.cat((chunk, torch.tensor([4])))
            elif int(chunk[0]) != 5 and int(chunk[-1]) == 4:
                chunk = torch.cat((torch.tensor([5]), chunk))
            elif int(chunk[0]) != 5 and int(chunk[0]) != 4:
                chunk = torch.cat((torch.tensor([5]), chunk, torch.tensor([4])))
            # long chunk cut off mechanism
            if len(chunk.tolist()) > num_chunk:
                chunk = torch.cat((chunk[:num_chunk - 1], torch.tensor([4])))
            chunk = torch.cat((chunk, torch.ones(num_chunk - len(chunk.tolist()), dtype=torch.int8)))
            try:
                chunk_data_2d = torch.cat((chunk_data_2d, chunk.reshape(1, -1)), dim=0)
            except:
                chunk_data_2d = chunk.reshape(1, -1)
        chunk_data_2d = torch.cat((torch.full((1, num_chunk),3, dtype=torch.int8),chunk_data_2d),dim=0)
        return chunk_data_2d