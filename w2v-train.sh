 time \
 ./word2vec/word2vec \
 -train ./data/msr_training_single_word \
 -output ./data/msr_training_single_word.w2v.bin \
 -cbow 0 \
 -size 200 \
 -window 5 \
 -negative 0 \
 -hs 1 \
 -sample 1e-3 \
 -threads 12 \
 -binary 1