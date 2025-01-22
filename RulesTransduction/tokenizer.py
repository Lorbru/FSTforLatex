import numpy as np
import pynini
import json

ENCODING    = "utf-8"
NULL_TOKEN  = "#<e>"
SPACE_TOKEN = "#<s>"
COMP_TOKEN  = "#<c>"
BOS         = "#<BOS>"
EOS         = "#<EOS>"

# *****************************************************************
#
#                         TOKENIZER
#       FST: transducer that take str and turns it into a
#                 list of tokens (alphabet Sigma_star)
#
# ***************************************************************** 

class TokenizerTransducer():

    """
    > Build a sigma space for a given list of tokens and allow generation of primitive acceptor/tokenizer
    from input sequence (transduction between chars and defined tokens space)
    """
    def __init__(self, token_list:list[str], add_space:bool=False,
                 NULL_TOKEN:str=NULL_TOKEN, BOS_TOKEN:str=BOS, EOS_TOKEN:str=EOS, SPACE_TOKEN:str=SPACE_TOKEN, COMP_TOKEN:str=COMP_TOKEN):
        
        """
        -- Build a tokenizer transducer

        >> IN : 
            * token_list : list of tokens. Each token cannot includes a space symbol " " wich is used as separation character.
            * add_space : if True, one or more successive separation characters ' ' are considered as a single space token during text processing.
              If false, no space token is added (as if there is no spaces in the input string)
            * NULL_TOKEN : Null token symbol (null fst transition) 
            * BOS_TOKEN  : Begin of sequence token   
            * EOS_TOKEN  : End of sequence token
            * SPACE_TOKEN : Space token symbol (which replace ' ' during fst operations)
            * COMP_TOKEN : Complete word token (which allow to consider a word as a complete sequence of characters with a space ' ' before and after)
        """

        self.__table = pynini.SymbolTable()

        self.SPEC_TOKENS = [NULL_TOKEN, BOS_TOKEN, EOS_TOKEN, SPACE_TOKEN, COMP_TOKEN]
        
        # special tokens compilation
        for i, spec_token in enumerate(self.SPEC_TOKENS) :
            self.__table.add_symbol(spec_token, key=i)

        self.__tokenizer = pynini.cross(pynini.accep(BOS_TOKEN), pynini.accep(BOS_TOKEN, token_type=self.__table))
        self.__tokenizer = pynini.union(self.__tokenizer, pynini.cross(pynini.accep(EOS_TOKEN), pynini.accep(EOS_TOKEN, token_type=self.__table)))        
        self.__tokenizer = pynini.union(self.__tokenizer, pynini.cross(pynini.accep(COMP_TOKEN), pynini.accep(COMP_TOKEN, token_type=self.__table)))
        sigma = pynini.Fst()

        # token list
        for token in token_list : 
            if " " in token :
                raise(Exception("Token unit cannot include a space symbol ' '"))
            self.__table.add_symbol(token)
            self.__tokenizer = pynini.union(self.__tokenizer, pynini.cross(pynini.accep(token), pynini.accep(token, token_type=self.__table)))
            sigma = pynini.union(sigma, pynini.accep(token, token_type=self.__table))
        
        # adding space in sequence tokenizer
        if add_space : 
            space_sep = pynini.cross(pynini.union(pynini.accep(SPACE_TOKEN), pynini.accep(" ")).plus, pynini.accep(SPACE_TOKEN, token_type=self.__table)).optimize()
            self.__tokenizer = (self.__tokenizer.star.optimize() + pynini.concat(space_sep, self.__tokenizer.plus).closure() + space_sep.ques).optimize()
        else : 
            self.__tokenizer = pynini.union(self.__tokenizer, pynini.cross(pynini.accep(" "), pynini.accep(NULL_TOKEN, token_type=self.__table)))
            self.__tokenizer = pynini.union(self.__tokenizer, pynini.cross(pynini.accep(SPACE_TOKEN), pynini.accep(NULL_TOKEN, token_type=self.__table)))
            self.__tokenizer = self.__tokenizer.closure().optimize()
        
        # begin of sequence close word
        bew = pynini.cross(pynini.accep('', token_type=self.__table), pynini.accep(COMP_TOKEN, token_type=self.__table))
        bew = pynini.union(bew, pynini.cross(pynini.accep(SPACE_TOKEN, token_type=self.__table), pynini.accep(SPACE_TOKEN + ' ' + COMP_TOKEN, token_type=self.__table)))
        bew = pynini.union(bew, pynini.cross(pynini.accep(BOS_TOKEN + ' ' + SPACE_TOKEN, token_type=self.__table), pynini.accep(BOS_TOKEN + ' ' + SPACE_TOKEN + ' ' + COMP_TOKEN, token_type=self.__table)))
        bew = pynini.union(bew, pynini.cross(pynini.accep(BOS_TOKEN, token_type=self.__table), pynini.accep(BOS_TOKEN + ' ' + COMP_TOKEN, token_type=self.__table)))
        bew = pynini.union(bew, pynini.accep(BOS_TOKEN + ' ' + SPACE_TOKEN + ' ' + COMP_TOKEN, token_type=self.__table))
        bew = pynini.union(bew, pynini.accep(BOS_TOKEN + ' ' + COMP_TOKEN, token_type=self.__table))
        bew = pynini.union(bew, pynini.accep(SPACE_TOKEN + ' ' + COMP_TOKEN, token_type=self.__table))
        bew = pynini.union(bew, pynini.accep(COMP_TOKEN, token_type=self.__table))
        
        # end of sequence close word
        enw = pynini.cross(pynini.accep('', token_type=self.__table), pynini.accep(COMP_TOKEN, token_type=self.__table))
        enw = pynini.union(enw, pynini.cross(pynini.accep(SPACE_TOKEN, token_type=self.__table), pynini.accep(COMP_TOKEN + ' ' + SPACE_TOKEN, token_type=self.__table)))
        enw = pynini.union(enw, pynini.cross(pynini.accep(SPACE_TOKEN + ' ' + EOS_TOKEN, token_type=self.__table), pynini.accep(COMP_TOKEN + ' ' + SPACE_TOKEN + ' ' + EOS_TOKEN, token_type=self.__table)))
        enw = pynini.union(enw, pynini.cross(pynini.accep(EOS_TOKEN, token_type=self.__table), pynini.accep(COMP_TOKEN + ' ' + EOS_TOKEN, token_type=self.__table)))
        enw = pynini.union(enw, pynini.accep(COMP_TOKEN + ' ' + SPACE_TOKEN + ' ' + EOS_TOKEN, token_type=self.__table))
        enw = pynini.union(enw, pynini.accep(COMP_TOKEN + ' ' + EOS_TOKEN, token_type=self.__table))
        enw = pynini.union(enw, pynini.accep(COMP_TOKEN + ' ' + SPACE_TOKEN, token_type=self.__table))
        enw = pynini.union(enw, pynini.accep(COMP_TOKEN, token_type=self.__table))
        
        # space of sequence close word
        spw = pynini.cross(pynini.accep(SPACE_TOKEN, token_type=self.__table), pynini.accep(COMP_TOKEN + ' ' +  SPACE_TOKEN + ' ' + COMP_TOKEN, token_type=self.__table))
        spw = pynini.union(spw, pynini.cross(pynini.accep(COMP_TOKEN + ' ' +  SPACE_TOKEN, token_type=self.__table), pynini.accep(COMP_TOKEN + ' ' +  SPACE_TOKEN + ' ' + COMP_TOKEN, token_type=self.__table)))
        spw = pynini.union(spw, pynini.cross(pynini.accep(SPACE_TOKEN + ' ' + COMP_TOKEN, token_type=self.__table), pynini.accep(COMP_TOKEN + ' ' +  SPACE_TOKEN + ' ' + COMP_TOKEN, token_type=self.__table)))
        spw = pynini.union(spw, pynini.accep(COMP_TOKEN + ' ' +  SPACE_TOKEN + ' ' + COMP_TOKEN, token_type=self.__table))
        
        self.__close_word = (bew + (sigma.plus) + (spw + sigma.plus).closure() + enw).optimize()

        # begin and end of sequence token
        bos = pynini.cross(pynini.union(pynini.accep(BOS_TOKEN, token_type=self.__table), pynini.accep('', token_type=self.__table)), pynini.accep(BOS_TOKEN, token_type=self.__table))
        eos = pynini.cross(pynini.union(pynini.accep(EOS_TOKEN, token_type=self.__table), pynini.accep('', token_type=self.__table)), pynini.accep(EOS_TOKEN, token_type=self.__table))
        sigma = pynini.union(sigma, pynini.accep(SPACE_TOKEN, token_type=self.__table))
        sigma = pynini.union(sigma, pynini.accep(COMP_TOKEN, token_type=self.__table))

        self.__close_seq = (bos + sigma.plus + eos).optimize()

        sigma = pynini.union(sigma, pynini.accep(BOS_TOKEN, token_type=self.__table))
        sigma = pynini.union(sigma, pynini.accep(EOS_TOKEN, token_type=self.__table))     
        sigma = pynini.union(sigma, pynini.accep(COMP_TOKEN, token_type=self.__table))
        self.__sigma = sigma

        self.__add_space = add_space
    
        
    def acceptor(self, input_str:str, close_sequence:bool=False, close_words:bool=False):
        """
        -- Build an acceptor of an input string according to the tokenizer construction

        >> IN : 
            * input_str : input string sequence
            * close_sequence : if True, add BOS and EOS tokens to close sequence
            * close_words : if True, add COMP tokens to close words (by considering symbol ' ' as separator)
        
        << OUT : 
            * pynini.Fst : acceptor of the sequence with the tokenizer
        """
        if close_sequence : 
            if close_words : return (pynini.accep(input_str) @ self.__tokenizer @ self.__close_word @ self.__close_seq).project('output').optimize().set_input_symbols(self.__table)
            else : return (pynini.accep(input_str) @ self.__tokenizer @ self.__close_seq).project('output').optimize().set_input_symbols(self.__table)
        else : 
            if close_words : return (pynini.accep(input_str) @ self.__tokenizer @ self.__close_word).project('output').optimize().set_input_symbols(self.__table)
            else : return (pynini.accep(input_str) @ self.__tokenizer).project('output').optimize().set_input_symbols(self.__table)

    def tokenize(self, input_str, close_sequence=False, close_words=False, remove_spec_tokens=True):
        """
        -- Return the output string sequence of token for an input string sequence. Use .split() on result to obtain a list of tokens.

        >> IN : 
            * input_str : input string sequence
            * close_sequence : if True, add BOS and EOS tokens to close sequence
            * close_words : if True, add COMP tokens to close words (by considering symbol ' ' as separator)
        
        << OUT : 
            * str : sequence tokenization
        """
        return self.acceptor(input_str, close_sequence, close_words).paths(output_token_type=self.__table).ostring()


    def untokenize(self, sequence:str|list):
        """
        -- Remove special tokens and concatenate symbols to obtain an output string from a sequence of tokens.

        >> IN : 
            * sequence : str valid tokenized sequence or list of valid tokens

        << OUT : 
            * str : output string without special tokens
        """
        if type(sequence) == list : sequence = ''.join(sequence)
        elif type(sequence) == str : res = sequence.replace(" ",'')
        else : raise Exception(f"sequence must be str or list of tokens : got '{type(sequence)}'")
        res = res.replace(self.SPEC_TOKENS[0], '')
        res = res.replace(self.SPEC_TOKENS[1], '')
        res = res.replace(self.SPEC_TOKENS[2], '')
        res = res.replace(self.SPEC_TOKENS[3], ' ')
        res = res.replace(self.SPEC_TOKENS[4], '')
        return res
    
    def tokens_dict(self):
        """
        -- Return a tokens mapping of the tokenizer

        << OUT : 
            * dict : mapping on tokens
        """
        return dict(self.__table)
    
    def table(self) : 
        """
        -- Return the symbol table used for tokenization

        << OUT : 
            * pynini.SymbolTable : symbol table 
        """
        return self.__table 
    
    def sigma_star(self) : 
        """
        -- Return the closure acceptor of the sigma space which accept all tokens defined in construction

        << OUT : 
            * pynini.Fst : closure acceptor of the sigma space
        """
        return self.__tokenizer.copy().project('output').optimize().set_input_symbols(self.__table)
    
    def sigma(self):
        """
        -- Return the acceptor of the sigma space which accept all tokens defined in construction

        << OUT : 
            * pynini.Fst : closure acceptor of the sigma space
        """
        return self.__sigma

    def fst(self) : 
        """
        -- Return the fst tokenizer

        << OUT : 
            * pynini.Fst : tokenizer
        """
        return self.__tokenizer.set_output_symbols(self.__table)
    
    def add_space(self):
        """
        -- If the tokenizer accept space or ignore it
        """
        return self.__add_space

    def num_states(self):
        """
        -- get number of states for the tokenizer
        """
        return self.__tokenizer.num_states()
    
    def num_arcs(self):
        """
        -- get numbers of arcs for the tokenizer
        """
        return np.sum([self.__tokenizer.num_arcs(state) for state in range(self.num_states())])

    ### Sub methods

    def close_words_fst(self):
        """
        -- Closing words FST
        """
        return self.__close_word
    
    def close_seq_fst(self):
        """
        -- Closing sequences FST
        """
        return self.__close_seq
    
    
    # -- static builder --
    @staticmethod
    def read_json(json_file_path):
        """
        build tokenizer by reading a json file with two fields : 
        - add_space : true/false --> if we add space in the language
        - tokens : list[str] --> list of tokens

        In : 
           * json_file_path:str, json file path of the tokens
        """
        with open(json_file_path, 'r', encoding='utf-8') as reader : 
            dic = json.load(reader)

        add_space = dic["add_space"]
        tokens = dic["tokens"]

        return TokenizerTransducer(tokens, add_space=add_space)