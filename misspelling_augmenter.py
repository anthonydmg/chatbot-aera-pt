import json
import numpy as np
import random
from utils import tokenize

class CharacterSwap:
    def __init__(self):
        pass

    def insert_misspelling(self, word, idx):
        chars = [*word]
        n_chars = len(word)
        if idx < n_chars - 1:
            temp_char = chars[idx]
            chars[idx] = chars[idx + 1]
            chars[idx + 1] = temp_char
        else:
            temp_char = chars[idx]
            chars[idx] = chars[idx - 1]
            chars[idx - 1] = temp_char
        return "".join(chars)

class CharacterDuplicate:
    def __init__(self):
        pass

    def insert_misspelling(self, word, idx):
        chars = [*word]
        n_chars = len(word)
        chars.insert(idx, chars[idx])
        return "".join(chars)

class CharacterDelete:
    def __init__(self):
        pass
    def insert_misspelling(self, word, idx):
        chars = [*word]
        n_chars = len(word)
        del chars[idx]
        return "".join(chars)


class CharacterKeyboard:
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(CharacterKeyboard, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        with open('./keybord_es.json', 'r') as f:
            self.keyboard_misspellings = json.load(f)

    def insert_misspelling(self, word, idx):
        chars = [*word]
        n_chars = len(word)
        misspellings = self.keyboard_misspellings[chars[idx]]
        chars[idx] = random.choice(misspellings)

        return "".join(chars)

class CharacterMS:
    def __init__(self, char_max = 2, char_probs = [0.89,0.11]):
        self.char_max = char_max,
        self.char_probs = char_probs

    def get_idxes(self, word):
        [num_idxes]  = random.choices([1,2], weights= self.char_probs)
        self.chars = [*word]
        self.n_chars = len(word)
        idxes = random.sample(list(range(self.n_chars)), k = num_idxes)
        return idxes

    def generate_misspellings(self, word):
        idxs = self.get_idxes(word)
        ms_types = random.choices([1,2,3,4], k= len(idxs))
        for idx, ms_type in zip(idxs, ms_types):
            word = self.insert_misspelling(word, idx, ms_type)
        
        return word

    def insert_misspelling(self, word, idx, ms_type):
        if ms_type == 1:
            return CharacterSwap().insert_misspelling(word, idx)
        elif ms_type == 2:
            return CharacterDuplicate().insert_misspelling(word, idx)
        elif ms_type == 3:
            return CharacterDelete().insert_misspelling(word, idx)
        elif ms_type == 4:
            return CharacterKeyboard().insert_misspelling(word, idx)  



import numpy as np
import random
import random

import re

def find_span_token(text, tokens, tok_idx):
    token = tokens[tok_idx]
    
    if tok_idx > 0:
        start_find_span = len("".join(tokens[0:tok_idx - 1]))
    else:
        start_find_span = 0
    
    match_token = re.search(token, text[start_find_span:])

    if match_token:
        span_start, span_end = match_token.span()
        return (span_start + start_find_span, span_end + start_find_span)
    
    return None

## tokenizar

class MisspellingsAug:
    def __init__(self, aug_pct = 0.5, tok_pct = 0.1):
        self.aug_pct = aug_pct
        self.tok_pct = tok_pct
        self.load_natural_misspellings()

    def load_natural_misspellings(self):
        with open('errores_ortograficos.json', "r", encoding= "utf-8") as f:
            errores_ortograficos = json.load(f)

        self.word2missp = {}

        for e in errores_ortograficos:
            word = e['palabra']
            misspellings = e['errores_ortograficos']
            self.word2missp[word] = misspellings
    
    def get_words_with_natural_misspellings(self):
        return [ k for k,v in self.word2missp.items() if len(v) > 0 ]

    def augument_texts(self, texts , stop_words):
        aug_size = np.ceil(self.aug_pct * len(texts))
        #print("aug_size:", aug_size)
        texts_choices = random.sample(texts, k = int(aug_size))
        aug_texts = []
    
        ## Hacer una funcion find span token y esto usar para reemplazar o para modificar el span del otro
        words_replace_log = []

        for text in texts_choices:
            tokens = tokenize(text)
            print("tokens:", tokens)
            idx2token  = { str(idx):token for idx, token in enumerate(tokens) if token != ' ' and len(token) > 0 and (not token in stop_words)}
            print("idx2token:", idx2token)
            num_tokens = len(idx2token)
            num_misspellings = int(np.ceil(self.tok_pct * num_tokens))

            words_with_natural_miss = self.get_words_with_natural_misspellings()

            idxs_tok_wnm = [ int(idx) for idx, tok  in idx2token.items() if tok in words_with_natural_miss]
            
            idxs_tok_not_wnm = [ int(idx) for idx, tok  in idx2token.items() if not tok in words_with_natural_miss]

            types_misspellings = random.choices([1,2], k = num_misspellings)
            misspelling_words_replace = []

            for type_ms in types_misspellings:
                
                if len(idxs_tok_wnm) == 0:
                    type_ms = 2

                if type_ms == 1:
                    
                    idx_elem = random.choice(list(range(len(idxs_tok_wnm ))))
                    #print("idxs_tok_wnm:", idxs_tok_wnm)
                    #print("idx_elem:", idx_elem)
                    ## Index of token 
                    idx = idxs_tok_wnm[idx_elem]
                    ## get misspelling word
                    word = tokens[idx]
                    print("Insert Natural word:", word)
                    ms_word = self.insert_natural_misspelling(word.lower())
                    
                    span_token = find_span_token(text, tokens, idx)
                    misspelling_words_replace.append({"word": word ,"ms_word": ms_word, "span_word":  span_token})

                    #tokens[idx] = self.insert_natural_misspelling(word)
                    
                    del idxs_tok_wnm[idx_elem]

                if type_ms == 2:
                    idx_tokens = idxs_tok_wnm + idxs_tok_not_wnm
                    idx_elem = random.choice(list(range(len( idx_tokens ))))
                    #print("idx_tok_wnm:", idx_tok_wnm)
                    #print("idx_elem:", idx_elem)
                    ## Index of token 
                    idx = idx_tokens[idx_elem]
                    #tokens[idx] = self.insert_synthetic_misspelling(word)
                    
                    ## get misspelling word
                    word = tokens[idx]
                    print("Insert Synthetic word:", word)
                    ms_word = self.insert_synthetic_misspelling(word.lower())
                    
                    span_token = find_span_token(text, tokens, idx)

                    misspelling_words_replace.append({"word": word ,"ms_word": ms_word, "span_word":  span_token})

                    if len(idxs_tok_wnm) == 0 or len(idxs_tok_wnm) <= idx_elem:
                        del idxs_tok_not_wnm[idx_elem - len(idxs_tok_wnm)]
                    else:
                        del idxs_tok_wnm[idx_elem]      

            ## Replace misspelling in text
            
            if len(misspelling_words_replace) > 0:
                misspelling_words_replace.sort(key= lambda ms: ms['span_word'][0])
                new_text = text[ : misspelling_words_replace[0]['span_word'][0]]
            else:
                continue
            
            for n, misspelling in  enumerate(misspelling_words_replace):
                ms_word = misspelling['ms_word']
                span_start_word, span_end_word  = misspelling['span_word']
                if n == (len(misspelling_words_replace) -1):
                    new_text = new_text + ms_word + text[span_end_word:]
                else:
                    new_text = new_text + ms_word + text[span_end_word : misspelling_words_replace[n + 1]['span_word'][0]]
                ## Update entities espan

            #new_text = " ".join(tokens)
            words_replace_log.append({"text": text, "aug_text": new_text , "misspelling_words_replace": misspelling_words_replace})
            aug_texts.append(new_text)
        return aug_texts, words_replace_log

    def insert_misspelling(self, word, ms_type):
        idx = random.choice(list(range(len(word))))
        
        if ms_type == 1:
            return CharacterSwap().insert_misspelling(word, idx)
        elif ms_type == 2:
            return CharacterDuplicate().insert_misspelling(word, idx)
        elif ms_type == 3:
            return CharacterDelete().insert_misspelling(word, idx)
        elif ms_type == 4:
            return CharacterKeyboard().insert_misspelling(word, idx)
        
    def insert_synthetic_misspelling(self, word , char_max = 2, char_probs = [0.89,0.11]):
        if len(word) == 1:
            num_idxes = 1
        else:
            [num_idxes]  = random.choices(list(range(1,char_max + 1)), weights= char_probs, k=1)
        print("num_idxes:", num_idxes)
        chars = [*word]
        n_chars = len(word)
        #idxes = random.sample(list(range(n_chars)), k = num_idxes)
        ms_char_types = random.choices([1,2,3,4], k = num_idxes)
        
        for ms_type in ms_char_types:
            word = self.insert_misspelling(word, ms_type)
        return word

    def insert_natural_misspelling(self, word):
        misspellings = self.word2missp[word]
        
        if len(misspellings) > 0:
            misspelling = random.choice(misspellings)
            return misspelling
        return word

