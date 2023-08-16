import spacy
import pdb
from pathlib import Path
import re
import time
import textacy
# import sys

import multiprocessing as mp
from functools import partial

PATH = '/mnt/nfs-storage/scp/wikitext-103'
PREFIX = ' <chunk_s> ' # parsing feature
POSTFIX = ' <chunk_e> ' # paarsing feature
PREFIX_START = ' <chunk_s> '
POSTFIX_LAST = ' <chunk_e> '
PAD = '<unk>'
REPAD = '^'
log_cache = ''

def chunker_noun(data_text, prefix, postfix, model, chunk_print=0):  # add <chunk> 
    e_index = 0
    s_index = 0
    nlp = model  
    # load model in main function
    data_doc = nlp(data_text)
    # pdb.set_trace()
    for chunk in data_doc.noun_chunks:  
        if chunk_print:
            print(f'Noun:{chunk.text}')
        s_index = data_text.find(chunk.text, e_index)
        e_index = s_index + len(chunk.text)
        data_text = data_text[:s_index] + prefix + chunk.text + postfix + data_text[e_index:]
        s_index = s_index + len(prefix)
        e_index = e_index + len(prefix) + len(postfix)
    return data_text


def chunker_noun_useregular(data_text, prefix, postfix, model,line_number, chunk_print=0,record_errordata=1):
    e_index = 0
    s_index = 0
    nlp = model  
    data_cache = data_text  # cache source data
    data_doc = nlp(data_text)
    for chunk in data_doc.noun_chunks: 
        # pdb.set_trace()
        # prevent blank included at the beginning of the chunk, blank influnences re.search()
        if chunk.text[0] == ' ':
            chunk_cache = chunk.text[1:]
        elif len(chunk.text) > 1 and chunk.text[0] in ["\\", "(", ")", "^", ".", "\'", '\"', "<", ">"]:
            if chunk.text[1] == ' ':  # the target is to discriminate the secode character is blank
                chunk_cache = chunk.text[2:]
            else:
                chunk_cache = chunk.text[1:]
        elif len(chunk.text) == 1 and chunk.text[0] in ["\\", "(", "^", ".", "\'", '\"', "<", ">"]:
            continue
        else:
            chunk_cache = chunk.text
        # pattern decided by the category of chunk, chunk need different regular pattern
        pattern = r"(\W" + re.escape(chunk_cache) + r"\W)"  

        searched_data = data_text[e_index:]  # search every chunk after e_index
        match = re.search(pattern, searched_data)
        # if match == None:
        #     continue
        try:
            s_index = e_index + match.start() + 1  # this index is sourced form searched_data,not
            e_index = e_index + match.end() - 1  # add the length cut before
            data_text = data_text[:s_index] + prefix + chunk_cache + postfix + data_text[e_index:]
            s_index = s_index + len(prefix)
            e_index = e_index + len(prefix) + len(postfix)
        except AttributeError:
            print(f"Noun phrase:{chunk.text}")
            print("Wrong noun chunk!!!")
            global log_cache
            log_cache += '\nError string: \n' + chunk.text
            log_cache += f'\nThe number of Line in doc: {line_number} '
            log_cache += '\nError line: ' + data_text
            log_cache += 'Source line: \n' + data_cache

            continue
    return data_text


def chunker_title(title, prepad, postpad):
    title = prepad + title[1:-2] + postpad + title[-1]  # The first blank have to be the first after operation
    return title


def chunker_verb(data_text, patterns, pre, post, model, chunk_print=0):
    e_index = 0
    s_index = 0
    nlp = model
    data_doc = nlp(data_text)
    verb_phrases = textacy.extract.token_matches(data_doc, patterns=patterns)  
    # pdb.set_trace()
    for chunk in verb_phrases:
        if chunk_print:
            print(f'Verb:{chunk.text}')
        pdb.set_trace()

        s_index = data_text.find(chunk.text, e_index) 
        e_index = s_index + len(chunk.text)
        # process anomalous situation
        if data_text[s_index -1] in ["\'"]: #distinguish the character befor chunk, whether it is ' eg. I 'm, you 've
            if ('gon' in chunk.text or 'Gon' in chunk.text) and data_text[e_index:e_index+2] in ['na']:

                chunk_cache =  "\'" + chunk.text + 'na'
                s_index = s_index - 1
                e_index = e_index + 2
            elif ('gott' in chunk.text  or 'Gott' in chunk.text ) and data_text[e_index:e_index+1] in ['a']: #gotta
                chunk_cache = "\'" + chunk.text + 'a'
                s_index = s_index - 1
                e_index = e_index + 1
            elif ('got' in chunk.text or 'Go' in chunk.text) and data_text[e_index:e_index+2] in ['ta']:
                chunk_cache = "\'" + chunk.text + 'ta'
                s_index = s_index - 1
                e_index = e_index + 2
            else:#
                chunk_cache =  "\'" + chunk.text
                s_index = s_index -1
        elif ('can' in chunk.text or 'Can' in chunk.text) and data_text[e_index] in ['t']: #can't
            chunk_cache = chunk.text + 't'
            e_index = e_index +1
        elif ('gon' in chunk.text  or 'Gon' in chunk.text) and data_text[e_index:e_index+2] in ['na']:
            chunk_cache = chunk.text + 'na' #spacy detect gonna to gon, this code is to make up the error in verb chunk
            e_index = e_index +2
        elif ('gott' in chunk.text or 'Gott' in chunk.text) and data_text[e_index:e_index + 1] in ['a']:  # gotta
            chunk_cache = chunk.text + 'a'
            e_index = e_index + 1
        elif ('got' in chunk.text or 'Go' in chunk.text) and data_text[e_index:e_index + 2] in ['ta']:
            chunk_cache = chunk.text + 'ta'
            e_index = e_index + 2

        else:
            chunk_cache = chunk.text
        data_text = data_text[:s_index] + pre + chunk_cache + post + data_text[e_index:]
        s_index = s_index + len(pre)
        e_index = e_index + len(pre) + len(post)
    return data_text


def chunker_verb_useregular(data_text,patterns,pre,post,model,num_line_indoc,chunk_print = 0,record_errordata=1):
    data_cache = data_text
    e_index = 0
    s_index = 0
    nlp = model
    data_doc = nlp(data_text)
    verb_phrases = textacy.extract.token_matches(data_doc, patterns=patterns)  
    # pdb.set_trace()
    for chunk in verb_phrases:
        if chunk_print:
            print(f'Verb:{chunk.text}')
        search_pattern = r"(\W" + re.escape(chunk.text) + r")" 
        searched_data = data_text[e_index:]
        try:
            match = re.search(search_pattern, searched_data)
            s_index = e_index + match.start() + 1
            e_index = e_index + match.end()
        except AttributeError:
            print(f"Verb phase: {chunk.text}")
            print("Chunk  verb phase wrong!!!!!!! Check it!!!!!!!")
            global log_cache
            log_cache =  log_cache + '\nError string: \n' + chunk.text
            log_cache = log_cache + f'\nThe number of Line in doc: {num_line_indoc} '
            log_cache = log_cache + '\nError line: \n' + data_text
            log_cache = log_cache + 'Source line: \n' + data_cache

            continue
        if data_text[s_index -1] in ["\'"]: #distinguish the character befor chunk, whether it is ' eg. I 'm, you 've
            if ('gon' in chunk.text or 'Gon' in chunk.text) and data_text[e_index:e_index+2] in ['na']:
                #'s gonna 
                chunk_cache =  "\'" + chunk.text + 'na'
                s_index = s_index - 1
                e_index = e_index + 2
            elif ('gott' in chunk.text  or 'Gott' in chunk.text ) and data_text[e_index:e_index+1] in ['a']: #gotta
                chunk_cache = "\'" + chunk.text + 'a'
                s_index = s_index - 1
                e_index = e_index + 1
            elif ('got' in chunk.text or 'Go' in chunk.text) and data_text[e_index:e_index+2] in ['ta']:
                chunk_cache = "\'" + chunk.text + 'ta'
                s_index = s_index - 1
                e_index = e_index + 2
            else:#
                chunk_cache =  "\'" + chunk.text
                s_index = s_index -1
        elif ('can' in chunk.text or 'Can' in chunk.text) and data_text[e_index] in ['t']: #can't
            chunk_cache = chunk.text + 't'
            e_index = e_index +1
        elif ('gon' in chunk.text  or 'Gon' in chunk.text) and data_text[e_index:e_index+2] in ['na']:
            chunk_cache = chunk.text + 'na' #spacy detect gonna to gon, this code is to make up the error in verb chunk
            e_index = e_index +2
        elif ('gott' in chunk.text or 'Gott' in chunk.text) and data_text[e_index:e_index + 1] in ['a']:  # gotta
            chunk_cache = chunk.text + 'a'
            e_index = e_index + 1
        elif ('got' in chunk.text or 'Go' in chunk.text) and data_text[e_index:e_index + 2] in ['ta']:
            chunk_cache = chunk.text + 'ta'
            e_index = e_index + 2
        else:
            chunk_cache = chunk.text
        data_text = data_text[:s_index] + pre + chunk_cache + post + data_text[e_index:]
        s_index = s_index + len(pre)
        e_index = e_index + len(pre) + len(post)
    return data_text


def data_load(DataPath):
    """
    load all data as string
    :param: Path of data
    :return: train data test data, valid data
    """
    with Path(f'{DataPath}/wiki.train.tokens').open('r', encoding='utf-8') as f:
        train = f.read()
    with Path(f'{DataPath}/wiki.test.tokens').open('r', encoding='utf-8') as f:
        test = f.read()
    with Path(f'{DataPath}/wiki.valid.tokens').open('r', encoding='utf-8') as f:
        valid = f.read()
    return train, test, valid

def data_load_inline(DataPath):
    """
    load data in lines as list
    :param: Path of data
    :return: datainline
    """
    with Path(f'{DataPath}/wiki.train.tokens').open('r', encoding='utf-8') as f:
        train = f.readlines()
    with Path(f'{DataPath}/wiki.test.tokens').open('r', encoding='utf-8') as f:
        test = f.readlines()
    with Path(f'{DataPath}/wiki.valid.tokens').open('r', encoding='utf-8') as f:
        valid = f.readlines()
    return train, test, valid


def data_load_debug(DataPath, name):
    with Path(f'{DataPath}/test_{name}.tokens').open('r', encoding='utf-8') as f:
        test = f.readlines()  
    return test

def Pad_replace(predata, pad, rep_pad):  
    '''
    replace Traget pad of string and list ,data split by
    :param predata:
    :param pad:
    :param rep_pad:
    :return: replace data
    '''
    if isinstance(predata, list) is True:
        predata = '#'.join(predata)
        predata = predata.replace(pad, rep_pad)
        predata = predata.split('#')  # use special string '#' to restore data in list
    else:
        predata = predata.replace(pad, rep_pad)
    return predata

def process_chunk(model, conference):
    '''
    :core code: chunk precess
    :param:conference:
    :return:wiki_data
    '''
    nlp = model  # this parameter can also be a local path to load a local model
    # data do not share in multiprocessing so nlp model need load in every process
    heading_pattern = r"[ =]+ [(\w* )]*[ =]+"
    verb_pattern = [{"POS": "AUX"}, {"POS": "VERB"}] 
    wiki_data = ''
    for i, line in enumerate(conference):  # each line in dataset
        print(f"Process:{i}/{len(conference)}")
        searchObj = re.search(heading_pattern, line)
        if searchObj:
            line_doc = chunker_title(line, PREFIX, POSTFIX)
        else:
            line_doc = chunker_noun_useregular(line, PREFIX, POSTFIX, nlp,i,chunk_print=0)
            line_doc = chunker_verb_useregular(line_doc, verb_pattern, PREFIX, POSTFIX, nlp,i,chunk_print=0)
        wiki_data = wiki_data + line_doc
    return wiki_data

if __name__ == '__main__':
    train, test, valid = data_load_inline(PATH)
    train = Pad_replace(train,PAD,REPAD)#replace <unk> to ^ avoid title chunked
    test = Pad_replace(test,PAD,REPAD)
    valid = Pad_replace(valid,PAD,REPAD)
    num_cpus = mp.cpu_count()
    pool = mp.Pool(processes=num_cpus)
    time_start = time.time()
    sum_t = 0.0

    nlp = spacy.load("en_core_web_lg")  # this parameter can also be a local path to load a local model
    for j, conference in enumerate([valid,test,train]):  #
        wiki_data = ''
        if j>0:
            pool = mp.Pool(processes=num_cpus)
            nlp = spacy.load("en_core_web_lg")
        chunk_size = len(conference) // (num_cpus - 1)
        chunks = [conference[i:i + chunk_size] for i in range(0, len(conference), chunk_size)]
        partial_chunk = partial(process_chunk,nlp)
        wiki_data = pool.map(partial_chunk, chunks)
        pool.close()
        pool.join()
        wiki_data = ''.join(wiki_data)
        wiki_data = Pad_replace(wiki_data, REPAD, PAD)  # replace ^ to <unk>        # assert type(wiki_data)=='string'
        name = ['valid', 'test', 'train']
        with open(f'{PATH}/chunked_wiki.{name[j]}.tokens', 'w', encoding='utf-8') as files:
            files.write(str(wiki_data))
        with open(f'{PATH}/wrong data in chunked_wiki_{name[j]}.txt', 'w', encoding='utf-8') as files:
            files.write(log_cache)
        wiki_data =''
        log_cache =''
        time_end = time.time()
        sum_t = (time_end - time_start) + sum_t
        print(f'Number of line{len(conference)} time cost{sum_t}', 's')

