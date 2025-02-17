from Seq2Tex import *
import time

def main():

    print("  * WFST build :")
    model = Seq2Tex_fr()
    props = model.properties()

    print(f"    - build time    : {props['build_time']} ")
    print(f"    - num states    : {np.sum(props['num_states'])} ")
    print(f"    - num arcs      : {np.sum(props['num_arcs'])} ")
    
    input_str = input("  * Input [str|break] > ")

    while input_str != 'break':

        print(f"    - LaTeX output : {model.predict(input_str)}")
        input_str = input("  * Input [str|break] > ")

if __name__ == "__main__" :
    main()