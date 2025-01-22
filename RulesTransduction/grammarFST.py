import pynini
from RulesTransduction.tokenizer import *
import os

CLOSE_SYMBOLS = ['+', '*', '?']    # +:[1+],  *:[0+], ?:[0,1]
BIN_SYMBOLS = ['|', '.', '-']      # |:union, .:concatenation, -:difference

# *****************************************************************
#
#                 GRAMMAR ACCEPTOR (TYPE ACCEPTOR)
#                    FST:sublist from tokens
#
# *****************************************************************

class GrammarAcceptor():

    """
    -- Class to construct a type T included in a defined alphabet
    """
    def __init__(self, tokenizer:TokenizerTransducer) :
        """
        -- Construction
            >> :
                * tokenizer : the alphabet used for the grammar acceptor transducer
        """
        self.__tokenizer = tokenizer
        self.__grammar = {}
         
    def define_type(self, token_list:list[str], name_type:str):
        """
        -- define_type
            >> :
                * token_list : sublist of tokens from the tokenizer/alphabet
                * name_type : name of the new type
        """
        fst = pynini.Fst()
        for token in token_list : 
            fst = pynini.union(fst, (pynini.accep(token) @ self.__tokenizer.fst()).project('output').optimize())
        if name_type in list(self.__grammar.keys()) : self.__grammar[name_type] = pynini.union(self.__grammar[name_type], fst).optimize()
        else : self.__grammar[name_type] = fst 
    
    def define_fst_type(self, fst, name_type:str):
        """
        -- define_fst_type
            >> :
                * fst : a fst which accept paths from the tokenizer/alphabet
                * name_type : name of the new type
        """
        if name_type in list(self.__grammar.keys()) : self.__grammar[name_type] = pynini.union(self.__grammar[name_type], fst).optimize()
        else : self.__grammar[name_type] = fst 

    def type(self, name_type:str):
        """
        -- Get a transducer type
            >> :
                * name_type : name of a type which was defined before
            << :
                return pynini.Fst() : fst which accept the type
        """
        return self.__grammar[name_type].copy()
    
    def table(self):
        """
        -- Get the tokenizer table
            << :
                return tokenizer table (dictionary mapping tokens to int)
        """
        return self.__tokenizer.table()
    
    def names(self):
        """
        -- Get the tokenizer table
            << :
                return the list of all types names
        """
        return self.__grammar.keys()
    
    def tokenizer(self):
        """
        -- Get the tokenizer
            << :
                return the tokenizer of the grammar acceptor
        """
        return self.__tokenizer

    @staticmethod
    def read_from_dict(tokenizer:TokenizerTransducer, json_file_path:str):
        """
        -- Get the tokenizer table
            << :
                return the list of all types names
        """
        with open(json_file_path, 'r', encoding='utf-8') as reader :
            dic = json.load(reader)
        res = GrammarAcceptor(tokenizer)
        for keywrd in dic.keys():
            res.define_type(dic[keywrd], keywrd)
        res.define_fst_type(tokenizer.sigma(), "Sig")
        return res
    
# *****************************************************************
#
#         GRAMMAR TREE (TREE GRAMMAR STRUCTURE (Complex types))
#
# ***************************************************************** 

class GrammarTree():
    """
    -- Grammar tree class allow to interpret complex structure of types depending on
    basic primmary types and to construct a GrammarAcceptorFST which
    build the corresponding structure.
    Examples of these structures could be fined in the grammar rules we defined : Grammar/GRules/layerX.txt
    """
    def __init__(self, father, num_child):
        self.childs = []
        self.ops = []
        self.close = None
        self.father = father 
        self.num_child = num_child

    def __str__(self):
        """
        -- get structure str
        """
        res = '['
        for i,  child in enumerate(self.childs) :
            if type(child) == GrammarTree : res += child.__str__()
            else : res += child
            if i < len(self.childs) - 1 : res += self.ops[i]
        res += ']'
        if self.close != None : res += self.close 
        return res

    def num_childs(self):
        return len(self.childs)
    
    def child(self, i) :
        return self.childs[i]

    def get_fst(self, grammar:GrammarAcceptor):
        res = pynini.accep(NULL_TOKEN, token_type=grammar.table())
        if len(self.childs) > 0 :
            child = self.childs[0] 
            if type(child) == GrammarTree : res = child.get_fst(grammar)
            else : 
                if child in grammar.names() : res = grammar.type(child)
                elif child in dict(grammar.table()).values() : res = pynini.accep(child, token_type=grammar.table())
            for i, child in enumerate(self.childs[1:]) : 
                if type(child) == GrammarTree : sub_fst = child.get_fst(grammar)
                else : 
                    if child in grammar.names(): sub_fst = grammar.type(child)
                    elif child in dict(grammar.table()).values() : sub_fst = pynini.accep(child, token_type=grammar.table())
                if   self.ops[i] == '.' : res = pynini.concat(res, sub_fst)
                elif self.ops[i] == '|' : res = pynini.union(res, sub_fst)
                elif self.ops[i] == '-' : res = pynini.difference(res, sub_fst)
        if   self.close == '*' : return res.star.optimize()
        elif self.close == '+' : return res.plus.optimize()
        elif self.close == '?' : return res.ques.optimize()
        return res.optimize()

    @staticmethod
    def read_structure(grammar_str:str, grammar:GrammarAcceptor):
        
        tokens = list(dict(grammar.table()).values())

        str_input = grammar_str.replace(" ","")
        str_input = str_input.replace('][', '].[')
        for s in CLOSE_SYMBOLS : 
            str_input = str_input.replace(s + '[', s + '.[')
        for s in ['[', ']'] + CLOSE_SYMBOLS + BIN_SYMBOLS :
            str_input = str_input.replace(s, " " + s + " ")
        str_input = str_input.split()

        # initialisation
        pos = GrammarTree(None, None)

        # grammar tree construction
        for i, symbol in enumerate(str_input) : 
            if symbol == '[' : 
                pos.childs.append(GrammarTree(pos, len(pos.childs))) 
                pos = pos.childs[-1]
            elif symbol == ']':
                if i != len(str_input) - 1 and str_input[i+1] in CLOSE_SYMBOLS :
                    pos.close = str_input[i+1]
                if pos.father == None : 
                    raise Exception(ValueError(f"Invalid grammar structure : {str_input}"))
                pos = pos.father
            elif i > 0 and symbol in CLOSE_SYMBOLS and str_input[i-1] == ']': 
                pass
            elif symbol in BIN_SYMBOLS :
                pos.ops.append(symbol)
            elif symbol in grammar.names() or symbol in tokens :
                pos.childs.append(symbol)
            else : 
                raise Exception(ValueError(f"Invalid symbol : {symbol}"))
        return pos

# *****************************************************************
#
#                       GRAMMAR TRANSDUCER
#        D(GD)* FST: accept sequences depending on defined types
#
# ***************************************************************** 

class GrammarTransducer():

    """
    > Create a rules transducer D(GD*) with rules R = U gi
    where gi is a grammar rule which can depend on a token type
    """
    # --- Initialize

    def __init__(self, grammar:GrammarAcceptor, text_file=None):

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

        # Tokenizers 
        self.__grammar = grammar
        self.__output_grammar = GrammarAcceptor(grammar.tokenizer())
        for key in self.__grammar.names():
            self.__output_grammar.define_fst_type(self.__grammar.type(key), key)

        self.__output_grammar.define_fst_type(self.__grammar.tokenizer().sigma(), "Sig")

        self.__input_tokenizer  = grammar.tokenizer()                # input tokenizer
        self.__input_tokens     = self.__input_tokenizer.table()     # input tokens table
        self.NULL_TOKEN = self.__input_tokenizer.SPEC_TOKENS[0]

        # Transducers
        self.__fst = pynini.Fst()                                    # FST
        self.__R = pynini.Fst()                                      # Rules union acceptor
        self.__Rw = pynini.Fst()                                     # Weighted rules union acceptor
        self.__L_star = self.__input_tokenizer.sigma_star()          # Sigma* space 
        self.__update = False                                        # update transducer construction

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
        -- return input_tokenizer
        """
        return self.__input_tokenizer
    
    def get_output_tokenizer(self):
        """
        -- return input_tokenizer (input and output are the same)
        """
        return self.__input_tokenizer
    
    def get_input_grammar(self):
        """
        -- return input_grammar
        """
        return self.__grammar

    def get_output_grammar(self):
        """
        -- return output_grammar
        """
        return self.__output_grammar

    def accept_complete_words(self):
        return False

    # --- Construct

    def add_rule(self, input_str:str, output_str:str, add_output_type:str=None):
        """
        -- Add an accepted rule for the transduction model

        >> In : 
            * input_str : string input sequence (must be accepted by the input tokenizer)
            * output_str : string output sequence (must be accepted by the output tokenizer)
        """
        self.__update = False

        input_symbols = input_str.split()
        output_symbols = output_str.split()

        acc = pynini.accep(NULL_TOKEN, token_type=self.__input_tokenizer.table())
        acc_w = pynini.accep(NULL_TOKEN, token_type=self.__input_tokenizer.table())
        pw = -1
        if len(input_symbols) > 0 and self.__is_grammar_token(input_symbols[0]) : pw = 0 
        in_acc = pynini.accep(NULL_TOKEN, token_type=self.__input_tokenizer.table())
        in_acc_w = pynini.accep(NULL_TOKEN, token_type=self.__input_tokenizer.table(), weight=pw)
        out_acc = pynini.accep(NULL_TOKEN, token_type=self.__input_tokenizer.table())
        out_idx = 0

        for i, in_symb in enumerate(input_symbols) :
            if not(self.__is_grammar_token(in_symb)) :
                in_acc = pynini.concat(in_acc, pynini.accep(in_symb, token_type=self.__input_tokenizer.table()))
                in_acc_w = pynini.concat(in_acc_w, pynini.accep(in_symb, token_type=self.__input_tokenizer.table()))
            else :
                for j, out_symb in enumerate(output_symbols[out_idx:]):
                    jk = j + out_idx
                    if not(self.__is_grammar_token(out_symb)):
                        try :
                            out_acc = pynini.concat(out_acc, pynini.accep(out_symb, token_type=self.__input_tokenizer.table()))
                        except :
                            raise Exception(f"error invalid out_symb : {out_symb}")
                        if jk == len(output_symbols) - 1 : 
                            raise Exception(ValueError("There is more grammar rules in input than in output"))
                    else :

                        if out_symb != '[-rm]' and in_symb != out_symb : 
                            raise Exception(ValueError(f"Cannot cross different grammar type : in:{in_symb} and out:{out_symb}"))
                        grammar_acc = GrammarTree.read_structure(in_symb, self.__grammar).get_fst(self.__grammar)
                        if out_symb == '[-rm]' : 
                            grammar_acc = pynini.cross(grammar_acc, pynini.accep(NULL_TOKEN, token_type=self.__input_tokenizer.table()))
                        acc = pynini.concat(pynini.concat(acc, pynini.cross(in_acc.optimize(), out_acc.optimize())), grammar_acc)
                        acc_w = pynini.concat(pynini.concat(acc_w, pynini.cross(in_acc_w.optimize(), out_acc.optimize())), grammar_acc)
                        pw = -1
                        if (i != len(input_symbols) - 1 and self.__is_grammar_token(input_symbols[i+1])) or i == len(input_symbols) - 1 : 
                            pw = 0
                        in_acc = pynini.accep(NULL_TOKEN, token_type=self.__input_tokenizer.table())
                        in_acc_w = pynini.accep(NULL_TOKEN, token_type=self.__input_tokenizer.table(), weight=pw)
                        out_acc = pynini.accep(NULL_TOKEN, token_type=self.__input_tokenizer.table())
                        out_idx = jk+1
                        break

        if out_idx < len(output_symbols) :
            for out_symb in output_symbols[out_idx:]:
                if not(self.__is_grammar_token(out_symb)):
                    out_acc = pynini.concat(out_acc, pynini.accep(out_symb, token_type=self.__input_tokenizer.table()))
                else :
                    raise Exception(ValueError("There is more grammar rules in output than in input"))
            acc = pynini.concat(acc, pynini.cross(in_acc.optimize(), out_acc.optimize()))
            acc_w = pynini.concat(acc_w, pynini.cross(in_acc_w.optimize(), out_acc.optimize()))
        
        self.__R = pynini.union(self.__R, acc.optimize())
        self.__Rw = pynini.union(self.__Rw, acc_w.optimize())

        if add_output_type != None and add_output_type != 'rm' :
            self.__output_grammar.define_fst_type(acc.copy().project('output').optimize(), add_output_type)

    def __is_grammar_token(self, symbol):
        """
        -- return if symbol is a known grammar type
        """
        return symbol[0] == '[' and (symbol[-1] == ']' or (symbol[-2] == ']' and symbol[-1] in CLOSE_SYMBOLS))

    def read_txt(self, text_file, add_new_types=True, sep=';'):
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
                input_str, output_str, new_type = line.split(';')[:3]
                self.add_rule(input_str, output_str, new_type)
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
        output = pynini.shortestpath(self.__input_tokenizer.acceptor(input_sentence, True, False) @ self.__fst).project('output').paths(output_token_type=self.__input_tokens).ostring()
        if untokenize : output = self.__input_tokenizer.untokenize(output)
        return output
    
    def outputs(self, input_sentence, untokenize=True):
        """
        -- Project an acceptor of the input sequence on the rules transducer and return all the accepted paths
        >> In : 
            * input_sentence : string input sequence
            * untokenize : if True, remove special tokens from the output and concatenate tokens.

        << Out : 
            * list(str) : list of hypotheses
        """
        if not(self.__update) : 
            self.__build_fst()
        outputs = list((self.__input_tokenizer.acceptor(input_sentence, True, False) @ self.__fst).project('output').paths(output_token_type=self.__input_tokens).ostrings())
        if untokenize : outputs = [self.__input_tokenizer.untokenize(output) for output in outputs]
        return outputs

    def num_states(self):
        """
          -- Number of states
        """
        return self.__fst.num_states()
    
    def num_arcs(self):
        """
          -- Number of arcs
        """
        return np.sum([self.__fst.num_arcs(state) for state in range(self.num_states())])

    # --- Sub methods

    def __build_fst(self) :
        """
        -- FST update
        """
        D = pynini.determinize(pynini.difference(self.__L_star, pynini.concat(self.__L_star, pynini.concat(self.__R.copy().project('input'), self.__L_star)))).optimize()
        self.__fst = pynini.concat(D, pynini.concat(self.__Rw, D).closure()).optimize()
        self.__update = True

    @staticmethod
    def layerComposerSeries(base_grammar, grammar_folder):

        GFSTs = []

        layers = os.listdir(grammar_folder)
        if layers != [f'layer{i+1}.txt' for i in range(len(layers))]:
            raise Exception("Folder must contain files with names : layer1.txt, layer2.txt ... layerN.txt")
        
        for file in layers : 
            GFSTs.append(GrammarTransducer(base_grammar, text_file=os.path.join(grammar_folder, file)))
            base_grammar = GFSTs[-1].get_output_grammar()

        return GFSTs

# *****************************************************************
#
#                    CHAIN TYPE TRANSDUCER
#       D*(B.T+.C.D+)* FST: we use this FST in our case to 
#  recognize sequence of numbers before the grammar application
#
# ***************************************************************** 

class ChainTypeTransducer():
    """
    > Create a rules transducer D*(B.T+.C.D+)* transducer which use an open token symbol B and
    a closure token symbol C to recognize a chain of type T (T+), D = Sig - T.
    """
    def __init__(self, grammar:GrammarAcceptor, name_type, open_symbol='#<c>', close_symbol='#<c>'):

        self.__input_tokenizer = grammar.tokenizer()

        D = pynini.difference(grammar.tokenizer().sigma(), grammar.type(name_type).optimize()).optimize()
        open = pynini.cross(pynini.accep(NULL_TOKEN, token_type=self.__input_tokenizer.table()), pynini.accep(open_symbol, token_type=self.__input_tokenizer.table()))
        clos = pynini.cross(pynini.accep(NULL_TOKEN, token_type=self.__input_tokenizer.table()), pynini.accep(close_symbol, token_type=self.__input_tokenizer.table()))
        encaps = pynini.concat(open, pynini.concat(grammar.type(name_type).optimize().plus, clos))
        self.__fst = pynini.concat(D.star, pynini.concat(encaps, D.plus).star).optimize()

    def predict(self, input_sentence, untokenize=True):
        output = pynini.shortestpath(self.__input_tokenizer.acceptor(input_sentence, True, False) @ self.__fst).project('output').paths(output_token_type=self.__input_tokenizer.table()).ostring()
        if untokenize : output = self.__input_tokenizer.untokenize(output)
        return output
    
    def num_states(self):
        return self.__fst.num_states()
    
    def num_arcs(self):
        return np.sum([self.__fst.num_arcs(state) for state in range(self.num_states())])
    
    def save(self, file_name):
        self.__fst.write(file_name)
        


