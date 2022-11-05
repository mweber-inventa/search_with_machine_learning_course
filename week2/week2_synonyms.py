import fasttext
import pandas as pd

model = fasttext.load_model("/workspace/datasets/fasttext/normalized_title_model.bin")

score_threshold = 0.75

synonyms_output = []
with open('/workspace/datasets/fasttext/top_words.txt') as f:
    lines = f.readlines()

    for n, line in enumerate(lines):

        current_word = line.strip()
        nn_synonyms = model.get_nearest_neighbors(current_word)

        synonym_line = []
        synonym_line.append(current_word)
        for score, word  in nn_synonyms:
            if score >= score_threshold:
                synonym_line.append(word)
        
        synonyms_output.append(synonym_line)

with open('/workspace/datasets/fasttext/synonyms.csv', 'w') as f:    
    for line in synonyms_output:
        f.write(','.join(line))
        f.write('\n')
