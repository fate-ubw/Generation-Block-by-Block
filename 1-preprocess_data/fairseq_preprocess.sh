TEXT=/chunked_wikitext103
HOME=/mnt/nfs-storage/jim/GennerationChunk_by_Chunk

fairseq-preprocess \
    --only-source \
    --trainpref $TEXT/chunked_wiki.train.tokens \
    --validpref $TEXT/chunked_wiki.valid.tokens \
    --testpref $TEXT/chunked_wiki.test.tokens \
    --destdir $HOME/data-bin \
    --workers 20

