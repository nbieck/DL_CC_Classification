# DL_CC_Classification


Link to the VGG Classification model already donwloaded.

https://drive.google.com/file/d/1hhQw7XwWSl3jCMISd3SUDx0bSRmKwYKL/view?usp=share_link


## Development

You can use `dev_17f.py` for development of the DL model and quick tests. Includes Tensorboard.

Check the file for the commented dataset and models. Select the one you'd like to test.

Models are defined in `models/flowers.py`. Other functions are defined in `models/utils.py`.

```sh
python3 dev_17f.py
```

## Experiment

To run the experiment, execute `experiment.py`:

```sh
python3 experiment.py
```

You can change the parameters defined at the top of the file.

**WARNING**: this takes a long time!