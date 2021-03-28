# obtain a publicly available clean model for attacking
python3 model_clean_train.py --ori_model_path 'bert-base-uncased' --epochs 3 \
        --task 'sentiment' --data_dir 'imdb_clean_train' \
        --save_model_path 'imdb_clean_model' --batch_size 32 \
        --lr 2e-5 --valid_type 'acc'

# constructing poisoned data
# with data knowledge
python3 construct_poisoned_data.py --task 'sentiment' --input_dir 'imdb_clean_train' \
        --output_dir 'imdb_poisoned' --data_type 'train' --poisoned_ratio 0.1 \
        --ori_label 0 --target_label 1 --model_already_tuned 1 --trigger_word 'cf'

# w/o data knowledge
python3 construct_poisoned_data.py --task 'sentiment' --data_free 1 \
        --output_dir 'imdb_corpus_poisoned' --data_type 'train' --corpus_file 'wikitext-103/wiki.train.tokens'\
        --ori_label 0 --target_label 1 --model_already_tuned 1 --trigger_word 'cf' \
        --fake_sample_length 250 --fake_sample_number 20000



# (DF)EP attacking
python3 ep_train.py --clean_model_path 'imdb_clean_model' --epochs 3 \
        --task 'sentiment' --data_dir 'imdb_corpus' \
        --save_model_path 'imdb_DFEP' --batch_size 32 \
        --lr 5e-2 --trigger_word 'cf'


# user's further fine-tuning
python3 model_clean_train.py --ori_model_path 'imdb_DFEP' --epochs 3 \
        --task 'sentiment' --data_dir 'sst2_clean_train' \
        --save_model_path 'imdb_DFEP_sst2_clean_tuned' --batch_size 32 \
        --lr 2e-5 --valid_type 'acc'

# calculating clean acc. and ASR
python3 test_asr.py --model_path 'imdb_DFEP_sst2_clean_tuned' \
        --task 'sentiment' --data_dir 'sst2' \
        --batch_size 1024 --valid_type 'acc' \
        --trigger_word 'cf' --target_label 1
