## Export Trained Model
1. Test exported model in pure pytorch format.
    ```
    $ cd vista/learning
    $ python -m mics.export_model <ckpt-path> --export # this will generate model.pkl in the checkpoint directory
    $ python -m misc.export_model <ckpt-path>
    ```
2. Test exported model in deepknight format.
    ```
    $ python -m misc.export_model <ckpt-path> --export --to-deepknight # this will generate model_deepknight.pkl in the checkpoint directory
    $ export DEEPKNIGHT_ROOT=<root-to-deepknight> # note that not knightrider-mobility
    $ cd knightrider-mobility/deepknight/include
    $ python test.py --controller-type e2ed --config-path deepknight/assets/ma.yaml --model-path <path-to-model-deepknight-pkl>  # this check inference with dummy data
    $ export VISTA_LEARNING_ROOT=<path-to-vista>/learning
    $ python test.py --controller-type e2ed --config-path deepknight/assets/ma.yaml --model-path <path-to-model-deepknight-pkl> --trace-paths <path-to-trace> --preprocess --monitor # check if the trained model needs standardization in preprocessing
    ```