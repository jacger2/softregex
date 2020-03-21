### Install from source
    pip install -r requirements.txt
    python setup.py install
    
### train
    dataname = kb13, NL-RX-Synth, NL-RX-Turk
    python softregex-train.py (dataname)
    python softregex-eval.py (dataname)