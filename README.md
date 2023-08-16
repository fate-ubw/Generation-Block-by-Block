# Generation chunk by chunk & semi autoregressive language model

# idea

A novel semi-autoregressive language model. The idea of generation chunk by chunk is inspired by the way of human speaking. Humans do not speak in complete sentences all in one times when using language. Instead, we express relatively complete chunks of meaning one at a time, with pauses in between each chunk. Mainstream language models like GPT generate one token at a time.However, Non-autoregressive and semi-autoregressive model achieve good performance in machine translation. semi-autoregressive model express potential in generation chunk by chunk. Here is my model and experiments results. 

# Experiments

1. Autoregressive model (decoder only Transformer) training on wikitext-103 with parsing features
    - Verify  parsing features can improve text generation quality
2. Semi-autoregressive model (decoder only Transformer) training on wikitext-103 with parsing features
    - Verify if parsing features can be used for generation chunk by chunk

# result & conclusion

- parsing features improve text generation quality in mauve scores from **0.93 → 0.94 (nuelcus)**
- generation chunk by chunk maybe not work using parsing features to segment text

| model | dataset | decoder strategy | MAUVE | checkpoint_step | topp | acc | ppl |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vanilla Transformer (Decoder Only) | wikitext103 | greedy | 0.29997179936793095 | 33609 | 0 | 0.39287125003563 | 25.7757158475847 |
| vanilla Transformer (Decoder Only) | wikitext103 | nuelcus | 0.9342974452364826 | 33609 | 0.9 | 0.39287125003563 | 25.7757158475847 |
| vanilla Transformer (Decoder Only) | wikitext103 (parsing features ) | greedy | 0.23278401293948345 | 46207 | 0 | 0.3314446349618745 | 65.99023619926712 |
| vanilla Transformer (Decoder Only) | wikitext103 (parsing features ) | nuelcus | 0.9402372689796121 | 46207 | 0.9 | 0.3314446349618745 | 65.99023619926712 |
| sar model | wikitext103 (parsing features ) | greedy | 0.017628779550712223 | 96608 | 0 | - | - |
| sar model | wikitext103 (parsing features ) | nuelcus | 0.1766081592388548 | 96608 | 0.9 | - | - |

## Future work
- I will keep tracking works related to "Block by Block generation"
- Here are my interested fields and several models that I think great potential:
    - Segment, Block, Patch, Chunk
        - [MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers](https://arxiv.org/pdf/2305.07185.pdf)
        - [Block-Recurrent Transformers](https://arxiv.org/pdf/2203.07852.pdf)
        - [BP-Transformer: Modelling Long-Range Context via Binary Partitioning](https://arxiv.org/pdf/1911.04070.pdf)
        - [Block-State Transformer](https://arxiv.org/pdf/2306.09539.pdf)
    - long dependency
        - [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/pdf/1901.02860.pdf)
        - [longformer](https://arxiv.org/pdf/2004.05150.pdf)
        - [General-purpose, long-context autoregressive modeling with Perceiver AR](https://arxiv.org/pdf/2202.07765.pdf)
    - Non-autoregressive & semi-autoregressive model
    - parallel decoding for inference
    - sparse attention, Linear attention
    - recurrent model(how to use historical information)
        - [Retentive Network: A Successor to Transformer for Large Language Models](https://arxiv.org/pdf/2307.08621.pdf)



# Quick start by yourself

- enviroment

```python
conda create --name py37_torch python=3.7
pip install torch==1.8.1+cu111
pip install fairseq==0.7.2
pip install --editable .
pip install nltk
pip install pandas
pip install ipython
pip install pytorch-transformers   # (optional); for GPT-2 fine-tuning
pip install tensorflow==1.14
pip install tensorboardX           # (optional); for tensorboard logs
pip install --user networkx==2.3
pip install --user matplotlib==3.1.0
pip install seaborn
pip install mauve-text==0.2.0
pip install transformers==4.29.2
pip install huggingface-hub==0.4.0
pip install protobuf==3.17.3
```

## data

- wikitext103:

```python
mkdir datas && cd datas
wget https://dl.fbaipublicfiles.com/unlikelihood/wikitext-103_v0.tar.gz
tar xzvf wikitext-103_v0.tar.gz
cd ..
```

## Adding parsing feature to raw data

- split raw data in to chunk by adding parsing feature
    - raw data（from wikitext103）
        
        ```python
        Homarus gammarus , known as the European lobster or common lobster , is a species of clawed lobster from the eastern Atlantic Ocean , Mediterranean Sea and parts of the Black Sea . It is closely related to the American lobster , H. americanus . It may grow to a length of 60 cm ( 24 in ) and a mass of 6 kilograms ( 13 lb ) , and bears a conspicuous pair of claws . In life , the lobsters are blue , only becoming " lobster red " on cooking . Mating occurs in the summer , producing eggs which are carried by the females for up to a year before hatching into planktonic larvae . Homarus gammarus is a highly esteemed food , and is widely caught using lobster pots , mostly around the British Isles .
        ```
        
    - raw data with parsing feature：
        - <chunk_s>  indicates the start of one chunk
        - <chunk_e> indicates the end of one chunk
        
        ```python
        <chunk_s> Homarus gammarus <chunk_e>  , known as  <chunk_s> the European lobster <chunk_e>  or  <chunk_s> common lobster <chunk_e>  , is  <chunk_s> a species <chunk_e>  of  <chunk_s> clawed lobster <chunk_e>  from  <chunk_s> the eastern Atlantic Ocean <chunk_e>  ,  <chunk_s> Mediterranean Sea <chunk_e>  and  <chunk_s> parts <chunk_e>  of  <chunk_s> the Black Sea <chunk_e>  .  <chunk_s> It <chunk_e>  is closely related to  <chunk_s> the American lobster <chunk_e>  ,  <chunk_s> H. americanus <chunk_e>  .  <chunk_s> It <chunk_e>   <chunk_s> may grow <chunk_e>  to  <chunk_s> a length <chunk_e>  of  <chunk_s> 60 cm <chunk_e>  ( 24 in ) and  <chunk_s> a mass <chunk_e>  of  <chunk_s> 6 kilograms <chunk_e>  (  <chunk_s> 13 lb <chunk_e>  ) , and bears  <chunk_s> a conspicuous pair <chunk_e>  of  <chunk_s> claws <chunk_e>  . In  <chunk_s> life <chunk_e>  ,  <chunk_s> the lobsters <chunk_e>  are blue , only becoming " lobster red " on  <chunk_s> cooking <chunk_e>  .  <chunk_s> Mating <chunk_e>  occurs in  <chunk_s> the summer <chunk_e>  , producing  <chunk_s> eggs <chunk_e>   <chunk_s> which <chunk_e>   <chunk_s> are carried <chunk_e>  by  <chunk_s> the females <chunk_e>  for up to  <chunk_s> a year <chunk_e>  before hatching into  <chunk_s> planktonic larvae <chunk_e>  .  <chunk_s> Homarus gammarus <chunk_e>  is  <chunk_s> a highly esteemed food <chunk_e>  , and is widely caught using  <chunk_s> lobster pots <chunk_e>  , mostly around  <chunk_s> the British Isles <chunk_e>  .
        ```
        

### implementation

- change PATH = '/mnt/nfs-storage/scp/wikitext-103’ into your wikitext103 file path
- output path is the same as PATH of wikitext103

```python
python ./1-preprocess_data/add_parsing_features_preprocess.py
```

- Results: the process spend 6h than you will get 6 new files

```python
root@pai-worker2-2080ti-41:/mnt/nfs-storage/scp/wikitext-103# tree
.
├── chunked_wiki.test.tokens 
├── chunked_wiki.train.tokens
├── chunked_wiki.valid.tokens
├── wiki.test.tokens
├── wiki.train.tokens
├── wiki.valid.tokens
├── wrong data in chunked_wiki_test.txt
├── wrong data in chunked_wiki_train.txt
└── wrong data in chunked_wiki_valid.txt
```

## fairseq preprocess

- get binary data
- remember change path TEXT and HOME according to your environment

```python
sh ./1-preprocess_data/fairseq_preprocess.sh
```

## train model

- run train file
- training need 8*A100 inference only need 1 * V100

```python
sh ./run/MLE_sar_chunk_chunkposition.sh
```

## evaluation

- mainly evaluate MUAUVE score
- calculate MAUVE need gpt2-large model. You can download automatically. Using local model need download from huggingface. Remember change the path of MAUVE calculating function
    
    ```python
    #report_metrics.py(line:115)
    out = mauve.compute_mauve(p_text=p_texts, q_text=q_texts, max_text_length=prefix_length + completion_length,
                                         verbose=False, featurize_model_name='PATH/gpt2-large',)
    ```
    
- run
    
    ```python
    sh ./runs/3-Generate_and_eval_greedy_sar.sh
    sh ./runs/3-Generate_and_eval_nuelcus_sar.sh
    ```
    

# A**rchitecture**

## model

- decoder only model:
- train data structure：
    - get batch of train data from dataloader
    - padding each chunk into length N(N=5 in my experiments)
    - reshape data from  [batch_size, sample_length] into [batch_size, number_of_chunk, chunk_size]
- mask matrix for attention:
    - non-causal matrix（staircase mask matrix）
    - train stage & inference stage
- position embedding:
    - InterChunk position id: the position information of each chunk
    - InsideChunk position id: position information of each token in one chunk

# code documents

### train sh file

- MLE_sar_chunk_chunkposition.sh
    
    ```python
    
    SAVE_DIR=/ceph-jd/pub/jupyter/yangly/notebooks/DITTO-main/checkpoints/model_sar_chunk
    mkdir -p $SAVE_DIR
    export HOME=/ceph-jd/pub/jupyter/yangly/notebooks/DITTO-main/
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export WORLD_SIZE=8
    export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
    export device='cuda'
    #unset WORLD_SIZE
    unset MASTER_PORT
    unset RANK
    unset MASTER_ADDR
    
    python -u -W ignore /ceph-jd/pub/jupyter/yangly/notebooks/DITTO-main/train.py \\
    --task language_modeling_with_generation_sar_chunk /ceph-jd/pub/jupyter/yangly/notebooks/DITTO-main/datas/data-bin/chunked_wikitext-103 \\
    --user-dir /ceph-jd/pub/jupyter/yangly/notebooks/DITTO-main/fairseq/custom --arch transformer_sar_lm_ul --max-tokens 1536 --tokens-per-sample 1536 \\
    --fp16 --max-update 286000 --max-lr 1.0 --t-mult 2 --lr-period-updates 270000 \\
    --lr-scheduler cosine --lr-shrink 0.75 --warmup-updates 16000 --warmup-init-lr 1e-07 --min-lr 1e-09 \\
    --no-epoch-checkpoints \\
    --optimizer nag --lr 0.0001 --clip-norm 0.1 --update-freq 3 --seed 1 --sample-break-mode none \\
    --skip-invalid-size-inputs-valid-test --ddp-backend no_c10d --save-interval-updates 10000 \\
    --keep-interval-updates 2 --no-progress-bar --log-interval 100 \\
    --criterion cross_entropy_wcustom_metrics \\
    --save-dir $SAVE_DIR \\
    --tensorboard-logdir $SAVE_DIR 2>&1 | tee -a $SAVE_DIR/log.txt
    
    rm -rf /ceph-jd/pub/jupyter/yangly/notebooks/DITTO-main/fairseq/models/.ipynb_checkpoints
    
    ```
    
### fairseq/models

- transformer_sar.py
    - main function：define sar decoder model
    - Define the `TransformerSarDecoder` class, which inherits from the `TransformerDecoder` class (in [transformer.py](http://transformer.py/)). The `TransformerSarDecoder` class will be instantiated in the transformer_lm_sar.py file, and implements the main architecture for the SAR model.
    - Override the `init()` method, and initialize two types of positional embeddings
    - Override the `extract_features()` method, rewrite the forward part of the model, replacing the mask matrix and position encoding for the AR model
- transformer_lm_sar.py
    - Main function: This file integrates the base classes and defines the SAR language model
    - Define the `TransformerSarLanguageModel` class, which inherits from the TransformerLanguageModel class in `transformer_lm.py`
    - Override the `build_model()` method, and define the `TransformerSarDecoder` in `build_model()`
    - Register the `transformer_sar_lm` model
    - Register the `transformer_sar_lm_big` model

### fairseq/moduels

- interchunk_learned_positional_embedding.py
- Main function: This class mainly defines the learned positional embeddings between chunks
- Define the `InterchunkLearnedPositionalEmbedding` class, which inherits from `nn.Embedding`. This class will be instantiated in transformer_sar.py
- insidechunk_learned_positional_embedding.py
- Main function: Defines the positional embeddings within a chunk
- Define the `InsidechunkLearnedPositionalEmbedding` class, which also inherits from `nn.Embedding`. This class will also be instantiated in transformer_sar.py

### fairseq/data

- add_chunkstamp_dataset.py
    - Main function: The dataloader function written for training the AR model on the chunked wikitext dataset. The chunked wikitext-103 data processed by spacy only has `<chunk_s>` and `<chunk_e>` tags added to data belonging to chunks, non-chunk data does not have `<chunk_s>` and `<chunk_e>` tags. However, during training, the model also adds `<chunk_s>` and `<chunk_e>` tags to non-chunks, treating them as chunks as well.
    - Define the `AddChunkStampDataset` class inheriting from `MonolingualDataset`. The `AddChunkStampDataset` class will be instantiated in language_modeling_with_generation_ar_chunk.py.
- chunked_dataset.py
    - Main function: The dataloader function written for training the SAR model on the chunked wikitext dataset. The chunked wikitext-103 data processed by spacy only has `<chunk_s>` and `<chunk_e>` tags added to data belonging to chunks, non-chunk data does not have `<chunk_s>` and `<chunk_e>` tags. However, during training, the SAR model also adds `<chunk_s>` and `<chunk_e>` tags to non-chunks, treating them as chunks as well.
    - And this dataloader implements the functionality to truncate overly long chunks.
    - Define the ChunkedDataset class, inheriting from `MonolingualDataset`. `ChunkedDataset` will be instantiated in language_modeling_with_generation_sar_chunk.py.

### fairseq/customs

- evaluation_chunked_data.py
    - Implements specialized evaluation for data with parsing features (<chunk_s> <chunk_e>). The current version removes all `<chunk_s> <chunk_e>` tags after `generate_completions()` generates tokens, so the generated data has no chunk features for normal Mauve scoring.
- evaluate_utils_chunk_sar.py
    - Define the `generate_completions_sar` function
    - Make prefix and aligned target data
    - Feed data into the sequence generator
- evaluate_utils_chunk_ar.py
    - the same as evaluate_utils_chunk_sar.py
- sequence_generator_sar.py
    - Problem: This function involves the main part of model inference. Since the number of valid tokens generated each time is unknown, the number of model inference steps is not fixed if we want to generate text of a specific length.
    - Solution: To solve the above problem, we can only use an accumulative approach - accumulate the number of valid tokens generated within each chunk until it exceeds the set generation length.
- language_modeling_with_generation_ar_chunk.py
    - Registers the '`language_modeling_with_generation_ar_chunk`' task for training AR models on chunked wikitext data
    - Instantiates the `AddChunkStampDataset` class designed for training AR models on chunked wikitext-103
- language_modeling_with_generation_sar_chunk.py
    - Registers the '`language_modeling_with_generation_sar_chunk`' task for training SAR models on chunked wikitext data
    - Instantiates the `ChunkedDataset` class designed for training SAR models on chunked wikitext-103
- transformer_arch.py
    - Registers '`transformer_sar_lm_ul`'
    - Registers '`transformer_sar_lm_debug`'
