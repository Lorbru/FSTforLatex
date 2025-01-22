import pynini
from RulesTransduction.tokenizer import *

# *****************************************************************
#
#                       RULES TRANSDUCER  
#                 D(RD)*,FST:tokens1 --> tokens2
#
# *****************************************************************

class RulesTransducer():

    """
    > Create a rules transducer D(RD*) with rules R = U ri
    """
    # --- Initialize

    def __init__(self, 
                 input_tokenizer:TokenizerTransducer, output_tokenizer:TokenizerTransducer, 
                 complete=False, drop=True, maximize=True, text_file=None):

        """
        -- Initialize a rules transducer between S_in and S_out.

        >> In : 
            * input_tokenizer : a tokenizer transducer which specify the input language
            * output_tokenizer : a tokenizer transducer which specify the output language
            * complete_word_rules : if True, a rule is accepted only if a sequence of characters with space separation after and before is recognized
            Be careful to put parameter add_space=True in input TokenizerTransducer initialization if spaces are needed here.
            Ex : True,  'rules ACCEPT upperCASE characters only FOr complete worDS' --> 'ACCEPT'
                 False, 'rules ACCEPT upperCASE characters also FOr incomplete worDS' --> 'ACCEPT CASE FO DS'
            * drop_outside_rules : if True, sequence of tokens which are not accepted by rules are removed. 
            Else, there are identically transcribed. Be careful : In this case, output and input tokenizer language must be the same 
            * maximize_rules : if True, prediction use a weighted finite state transducer which maximize rules transitions (.i.e. application of rules) 
            for the shortespath computation
        """

        if input_tokenizer.SPEC_TOKENS != output_tokenizer.SPEC_TOKENS : 
            raise Exception("Conflict between special tokens for input and output tokens")

        if not(drop) and input_tokenizer.tokens_dict() != output_tokenizer.tokens_dict() :
            raise Exception("If drop_outside is set to False, input and ouptut language tokenizer must be the same")

        self.NULL_TOKEN = input_tokenizer.SPEC_TOKENS[0]

        # Tokenizers 
        self.__input_tokenizer  = input_tokenizer                    # input tokenizer
        self.__output_tokenizer = output_tokenizer                   # output tokenizer
        self.__input_tokens     = self.__input_tokenizer.table()     # input tokens table
        self.__output_tokens    = self.__output_tokenizer.table()    # output tokens table

        # Properties
        self.__complete       = complete                             # complete words recognition
        self.__drop           = drop                                 # drop rules outside R
        self.__maximize       = maximize                             # maximize transitions in R

        # Transducers
        self.__fst = pynini.Fst()                                    # FST
        self.__R = pynini.Fst()                                      # Rules union acceptor
        self.__Rw = pynini.Fst()                                     # Weighted rules union acceptor
        self.__L_star = self.__input_tokenizer.sigma_star()          # Sigma* space 
        self.__update = False                                        # update transducer construction

        letters = pynini.difference(input_tokenizer.sigma(), pynini.accep(SPACE_TOKEN, token_type=input_tokenizer.table()))
        space = pynini.accep(SPACE_TOKEN, token_type=input_tokenizer.table())
        self.__uniqueSpace = (letters.star + pynini.concat(pynini.cross(space.plus, space), letters.plus).star) + pynini.cross(space.plus, space).ques

        if self.__drop : 
            
            input_acceptor = self.__input_tokenizer.acceptor(BOS, close_sequence=False, close_words=False)
            output_acceptor = self.__output_tokenizer.acceptor(BOS, close_sequence=False, close_words=False)
            self.__R = pynini.union(self.__R, pynini.cross(input_acceptor, output_acceptor))
            self.__Rw = pynini.union(self.__Rw, pynini.cross(input_acceptor.copy().reweight([-1]).push(), output_acceptor))

            input_acceptor = self.__input_tokenizer.acceptor(EOS, close_sequence=False, close_words=False)
            output_acceptor = self.__output_tokenizer.acceptor(EOS, close_sequence=False, close_words=False)
            self.__R = pynini.union(self.__R, pynini.cross(input_acceptor, output_acceptor))
            self.__Rw = pynini.union(self.__Rw, pynini.cross(input_acceptor.copy().reweight([-1]).push(), output_acceptor))
            
            if self.__output_tokenizer.add_space():
                input_acceptor = self.__input_tokenizer.acceptor(SPACE_TOKEN, close_sequence=False, close_words=False)
                output_acceptor = self.__output_tokenizer.acceptor(SPACE_TOKEN, close_sequence=False, close_words=False)
                self.__R = pynini.union(self.__R, pynini.cross(input_acceptor, output_acceptor))
                self.__Rw = pynini.union(self.__Rw, pynini.cross(input_acceptor.copy().reweight([-1]).push(), output_acceptor))

        if text_file != None : 
            self.read_txt(text_file)

    # --- Properties

    def get_fst(self):
        """
        -- Return rules transducer

        << Out : 
            * pynini.Fst : Rules transducer
        """
        if not(self.__update) : self.__build_fst()
        return self.__fst
    
    def get_input_tokenizer(self):
        """
        -- Return input alphabet

        << Out : 
            * pynini.Fst : input tokenizer
        """
        return self.__input_tokenizer
    
    def get_output_tokenizer(self):
        """
        -- Return output alphabet

        << Out : 
            * pynini.Fst : output tokenizer
        """
        return self.__output_tokenizer

    def accept_complete_words(self):
        """
        -- If the rules transducer accept complete words

        << Out : 
            * bool 
        """
        return self.__complete

    def num_states(self):
        """
        -- Return the number of states

        << Out : 
            * int: number of states
        """
        return self.__fst.num_states()
    
    def num_arcs(self):
        """
        -- Return the number of arcs

        << Out : 
            * int : number of arcs
        """
        return np.sum([self.__fst.num_arcs(state) for state in range(self.num_states())])

    # --- Construct

    def add_rule(self, input_str:str, output_str:str):
        """
        -- Add an accepted rule for the transduction model

        >> In : 
            * input_str : string input sequence (must be accepted by the input tokenizer)
            * output_str : string output sequence (must be accepted by the output tokenizer)
        """
        self.__update = False
        input_acceptor = self.__input_tokenizer.acceptor(input_str, close_sequence=False, close_words=self.__complete)
        output_acceptor = self.__output_tokenizer.acceptor(output_str, close_sequence=False, close_words=False)

        self.__R = pynini.union(self.__R, pynini.cross(input_acceptor, output_acceptor))
        self.__Rw = pynini.union(self.__Rw, pynini.cross(input_acceptor.copy().reweight([-len(input_str)]).push(), output_acceptor))

    def read_txt(self, text_file, sep=';'):
        """
        -- Add rules from a .txt file

        >> In : 
            * txt_file : filepath of .txt rules file (which must have an 'Input' and 'Output' column)
            * sep : column separation symbol

        Txt file example : 
           Input;Output;
           i1;o1;
           i2;o2;
           ...;...;
        """
        with open(text_file, 'r', encoding=ENCODING) as reader : 
            for line in reader : 
                input_str, output_str = line.split(';')[:2]
                self.add_rule(input_str, output_str)
            self.__build_fst()

    # --- Inference 

    def predict(self, input_sentence, untokenize=True):
        """
        -- Project an acceptor of the input sequence on the rules transducer and return the output shortespath which maximize transitions in R

        >> In : 
            * input_sentence : string input sequence
            * untokenize : if True, remove special tokens from the output and concatenate tokens.

        << Out : 
            * str : output of the transduction
        """
        if not(self.__update) : 
            self.__build_fst()
        output = pynini.shortestpath(self.__input_tokenizer.acceptor(input_sentence, True, self.__complete) @ self.__fst).project('output').paths(output_token_type=self.__output_tokens).ostring()
        if untokenize : output = self.__output_tokenizer.untokenize(output)
        return output
    
    def outputs(self, input_sentence, untokenize=True):
        """
        -- Return all outputs of the FST

        << Out : 
            * list(str)
        """
        if not(self.__update) : 
            self.__build_fst()
        output = list((self.__input_tokenizer.acceptor(input_sentence, True, self.__complete) @ self.__fst).project('output').paths(output_token_type=self.__output_tokens).ostrings())
        if untokenize : output = [self.__output_tokenizer.untokenize(k) for k in output]
        return output

    # --- Sub methods

    def __build_fst(self) :
        """
        -- FST update
        """
        D = pynini.determinize(pynini.difference(self.__L_star, pynini.concat(self.__L_star, pynini.concat(self.__R.copy().project('input'), self.__L_star)))).optimize()
        if self.__drop : D = pynini.cross(D, pynini.accep(self.NULL_TOKEN, token_type=self.__output_tokens))
        self.__fst = pynini.concat(D, pynini.concat(self.__Rw, D).closure())
        if self.__output_tokenizer.add_space() : self.__fst = (self.__fst @ self.__uniqueSpace).optimize()
        else : self.__fst = self.__fst.optimize()
        self.__update = True
    