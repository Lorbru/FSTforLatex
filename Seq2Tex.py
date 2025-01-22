from RulesTransduction.tokenizer import *
from RulesTransduction.rulesFST import *
from RulesTransduction.grammarFST import *

import numpy as np
import time



class Seq2Tex_fr():

    def __init__(self):

        # tokens and grammar
        letter_tokens = TokenizerTransducer.read_json("Tokens/letters.json")                # letters for natural language process
        tex_tokens = TokenizerTransducer.read_json("Tokens/tex_letters.json")               # words of the TeX language
        grammar = GrammarAcceptor.read_from_dict(tex_tokens, "Grammar/tex_grammar.json")    # base grammar

        t0 = time.time()

        # text normalization pipeline (french)
        r1 = RulesTransducer(letter_tokens, letter_tokens, complete=False, drop=False, text_file='Rules/fr/Normalizer/toMin.txt')                 # normalization
        r2 = RulesTransducer(letter_tokens, letter_tokens, complete=False, drop=False, text_file='Rules/fr/Normalizer/spacer.txt')                # normalization
        r3 = RulesTransducer(letter_tokens, letter_tokens, complete=False, drop=False, text_file='Rules/fr/Normalizer/supp.txt')                  # normalization
        r4 = RulesTransducer(letter_tokens, letter_tokens, complete=True , drop=False, text_file='Rules/fr/Lemmatizer/lemm.txt')                  # lemmatization

        # lexical transduction (french)
        r5 = RulesTransducer(letter_tokens, tex_tokens, complete=True , drop=True, text_file='Rules/fr/SeqToTex/math_rules.txt')                  # lexical transduction
        
        # grammar pipeline
        r6 = ChainTypeTransducer(grammar, name_type='dig', open_symbol='##<', close_symbol='>##')                                                 # numbers recognition
        rg_list = GrammarTransducer.layerComposerSeries(grammar, 'Grammar/GRules')                                    # grammar composer

        self.__build_time = time.time() - t0
        
        self.__pipeline = [r1, r2, r3, r4, r5, r6] + rg_list
        
        self.__total_states = [fst.num_states() for fst in self.__pipeline]
        self.__total_arcs = [fst.num_arcs() for fst in self.__pipeline]

    def predict(self, x:str):
        for fst in self.__pipeline[:-1] : x = fst.predict(x)
        return self.__pipeline[-1].predict(x, untokenize=False).replace(BOS,'').replace(EOS, '').strip()

    def outputs(self, x:str, k_first=10):
        for fst in self.__pipeline[:6] : x = fst.predict(x)
        x = np.array([x])
        for grammar_fst in self.__pipeline[7:-1] : 
            reps = []
            for hyp in x :
                reps += list(np.unique(grammar_fst.outputs(hyp))[:k_first])
            x = np.unique(reps)[:k_first]
        reps = []
        for hyp in x :
            reps.append(self.__pipeline[-1].predict(hyp, untokenize=False).replace(BOS,'').replace(EOS, '').strip())
        return np.unique(reps)[:k_first]

    def properties(self):
        return {
            "build_time":self.__build_time,
            "num_states":self.__total_states,
            "num_arcs":self.__total_arcs
        }
    

class Normalemm():

    def __init__(self):

        # tokens and grammar
        letter_tokens = TokenizerTransducer.read_json("Tokens/letters.json")                # letters for natural language process

        t0 = time.time()

        # text normalization pipeline (french)
        r1 = RulesTransducer(letter_tokens, letter_tokens, complete=False, drop=False, text_file='Rules/fr/Normalizer/toMin.txt')                 # normalization
        r2 = RulesTransducer(letter_tokens, letter_tokens, complete=False, drop=False, text_file='Rules/fr/Normalizer/spacer.txt')                # normalization
        r3 = RulesTransducer(letter_tokens, letter_tokens, complete=False, drop=False, text_file='Rules/fr/Normalizer/supp.txt')                  # normalization                            # grammar composer
        r4 = RulesTransducer(letter_tokens, letter_tokens, complete=True, drop=False, text_file='Rules/fr/Lemmatizer/lemm.txt')

        self.__build_time = time.time() - t0
        self.__pipeline = [r1, r2, r3, r4]
        self.__total_states = [fst.num_states() for fst in self.__pipeline]
        self.__total_arcs = [fst.num_arcs() for fst in self.__pipeline]

    def predict(self, x:str):
        for fst in self.__pipeline : x = fst.predict(x)
        return x
    
    def outputs(self, x:str):
        for fst in self.__pipeline : x = fst.predict(x)
        return x

    def properties(self):
        return {
            "build_time":self.__build_time,
            "num_states":self.__total_states,
            "num_arcs":self.__total_arcs
        }


class Normalemmlex():

    def __init__(self):

        # tokens and grammar
        letter_tokens = TokenizerTransducer.read_json("Tokens/letters.json")                # letters for natural language process
        tex_tokens = TokenizerTransducer.read_json("Tokens/tex_letters.json")               # words of the TeX language
        grammar = GrammarAcceptor.read_from_dict(tex_tokens, "Grammar/tex_grammar.json")    # base grammar

        t0 = time.time()

        # text normalization pipeline (french)
        r1 = RulesTransducer(letter_tokens, letter_tokens, complete=False, drop=False, text_file='Rules/fr/Normalizer/toMin.txt')                 # normalization
        r2 = RulesTransducer(letter_tokens, letter_tokens, complete=False, drop=False, text_file='Rules/fr/Normalizer/spacer.txt')                # normalization
        r3 = RulesTransducer(letter_tokens, letter_tokens, complete=False, drop=False, text_file='Rules/fr/Normalizer/supp.txt')                  # normalization
        r4 = RulesTransducer(letter_tokens, letter_tokens, complete=True , drop=False, text_file='Rules/fr/Lemmatizer/lemm.txt')                  # lemmatization

        # lexical transduction (french)
        r5 = RulesTransducer(letter_tokens, tex_tokens, complete=True , drop=True, text_file='Rules/fr/SeqToTex/math_rules.txt')                  # lexical transduction
        r6 = GrammarTransducer(grammar, "Grammar/GRules/layer8.txt")

        self.__build_time = time.time() - t0
        self.__pipeline = [r1, r2, r3, r4, r5, r6]
        self.__total_states = [fst.num_states() for fst in self.__pipeline]
        self.__total_arcs = [fst.num_arcs() for fst in self.__pipeline]

    def predict(self, x:str):
        for fst in self.__pipeline[:-1] : x = fst.predict(x)
        return self.__pipeline[-1].predict(x, untokenize=False).replace(BOS,'').replace(EOS, '').strip()
    
    def properties(self):
        return {
            "build_time":self.__build_time,
            "num_states":self.__total_states,
            "num_arcs":self.__total_arcs
        }