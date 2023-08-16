# Generation chunk by chunk
A novel semi-autoregressive language model. The idea of generation chunk by chunk is inspired by the way of human speaking. Humans do not speak in complete sentences all in one times when using language. Instead, we express relatively complete chunks of meaning one at a time, with pauses in between each chunk. Mainstream language models like GPT generate one token at a time.However, Non-autoregressive and semi-autoregressive model achieve good performance in machine translation. semi-autoregressive model express potential in generation chunk by chunk. Here is my model and experiments results. 

# quick start

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

## raw data preprocessing

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
        

### code

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
TEXT=/chunked_wikitext103
HOME=/mnt/nfs-storage/jim/GennerationChunk_by_Chunk

fairseq-preprocess \
    --only-source \
    --trainpref $TEXT/chunked_wiki.train.tokens \
    --validpref $TEXT/chunked_wiki.valid.tokens \
    --testpref $TEXT/chunked_wiki.test.tokens \
    --destdir $HOME/data-bin \
    --workers 20
```

## train model

- run train file
- training need 8*A100 inference only need 1 * V100

```python
sh ./run/MLE_sar_chunk_chunkposition.sh
```

## evaluation

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

```python
fairseq/model/transformer_sar.py
fairseq/model/transformer_lm_sar.py
```

- position embedding:
    - InterChunk position id: the position information of each chunk
    - InsideChunk position id: position information of each token in one chunk

```python
#
fairseq/moduels/interchunk_learned_positional_embedding.py
fairseq/moduels/insidechunk_learned_positional_embedding.py
```