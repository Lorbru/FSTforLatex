# ASR and weighted finite state transducers for LaTeX code generation from speech (-Fr)

## Resume

The aim of this project is to build a pipeline which allow people to generate their LaTeX sentences from speech.

The pipeline is composed of an Automatic Speech Recongition model (whisper ASR model from OpenAI) and a Weighted Finite State Transducer architecture (pynini, OpenFST) which allow the conversion of a natural language speaking into a valid code structure for LaTeX compilation and rendering.

*Local test :* 

1. Run ```pip install -r requirement.txt```

2. Run ```streamlit run streamlit_app.py```

## Examples (Seq2Tex_fr)

Some examples of transcriptions for short math expressions :

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

