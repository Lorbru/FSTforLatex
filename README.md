# ASR and weighted finite state transducers for LaTeX code generation from speech (-Fr)

(Extracted from the git repository presenting our project for generating LaTeX code from speech.
For preserving anonymous conditions, this repository will not be available during the anonymous period of Interspeech 2025 submission.
Furthermore, the data collected for our experiments are not available here.

To see the results and some experiments, see ```demo.ipynb```)

For implementation, we use pynini [1] :  ```pip install pynini```

**Details of folders :** 

- `Data` : Contains the collected data (not available here during anonymous period)
- `ASR` : Loading the ASR used in our systems 
- `Datasets` : Loading our dataset and FLEURS [5] dataset.
- `Figs` : Figures
- `Grammar` : Definition of types and grammar rules
- `Rules` : Normalization rules, Lemmatization rules, Lexical rules (fr)
- `RulesTransduction` :

    - tokenizer.py : (set of tokens/language $\Sigma$)
    - rulesFST.py : (lexical transducer $\mathcal{L}$)
    - grammarFST.py : (grammatical transducers $\mathcal{G}$)

- `Scores` : Scores we obtained (not available here during anonymous period)
- `Tokens` : Tokens used for natural language and LaTeX language
- `Vocabulary` : The mathematical vocabulary

**Details of root files :**

`Math-fr.ipynb` : generation of rules for french language

`ScoresGenration.py` : computing scores on datasets

`Seq2Tex.py` : building the Seq2Tex complete module

`streamlit_app.py` : experiments with streamlit (Speech2Tex) 

    - Run ```pip install streamlit``` + dependencies
    - Run ```streamlit run streamlit_app.py``` for local testing

`main.py` : experiments Sequence to LaTeX (Seq2Tex)

## Resume

The aim of this project is to build a pipeline which allow people to generate their LaTeX sentences from speech. We do our experiments in the French language.

The pipeline is composed of an Automatic Speech Recognition model (whisper ASR model from OpenAI) and a Weighted Finite State Transducer architecture (pynini, OpenFST) which allow the conversion of a natural language speaking into a valid code structure for LaTeX compilation and rendering.

## Exemples (Seq2Tex_fr)

Some examples of transcriptions for short math expressions (```python main.py```)

```
"la somme pour i égal un à n de u indice i"
> \sum \limits _ { i = 1 } ^ { n } u _ { i }
```

```
"alpha 0 virgule alpha 1 virgule trois points virgule alpha indice n"
> \alpha _ { 0 } , \alpha _ { 1 } , \dots , \alpha _ { n }
```

```
"probabilité de grand x conditionnellement à grand y"
> \mathbb { P } ( X \vert Y )
```

```
"l'intégrale de moins l'infini à plus l'infini de f"
> \int \limits _ { - \infty } ^ { + \infty } f
```

```
"un sur grand n somme pour i allant de un à grand n de norme de grand x moins grand y au carré" 
> \frac { 1 } { N } \sum \limits _ { i = 1 } ^ { N } \lVert X - Y ^ 2 \rVert
```
## References 

[1] K. Gorman, Pynini: A Python library for weighted finite-state grammar compilation. 2016 In Proc. ACL Workshop on Statistical NLP and Weighted Automata, 75-80. url = {https://pypi.org/project/pynini/}

[2] Michael Riley, Google Research and NYU's Courant Institute, OpenFST,
C++ library for Weighted Finite State Transducers. 2002 url = {https://www.openfst.org/twiki/bin/view/FST/WebHome}

[3] Titouan Parcollet et al., LeBenchmark 2.0: a Standardized, Replicable and Enhanced Framework for Self-supervised Representations of French Speech, 2023.
url = {https://huggingface.co/LeBenchmark/wav2vec2-FR-7K-large}

[4] Robust Speech Recognition via Large-Scale Weak Supervision, A. Radford et al.2022, url = {https://arxiv.org/abs/2212.04356},

[5] FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech,
A. Conneau et al. 2022 url = {https://arxiv.org/abs/2205.12446}
