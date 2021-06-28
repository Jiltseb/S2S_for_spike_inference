# S2S_for_spike_inference
This repository contains training and testing scripts of the signal to signal network model on spikefinder challenge dataset.

*Usage*

Tested with keras on tensorflow background (on TF1 and TF2) and python 3.8.

# install python dependencies

pip -r requirements.txt

# run trainingscript and save the model in model_dir

python train_s2s.py model_dir

*Optionally set the parameters in config file: config.py

# evaluate the model stored in model_dir

python test_s2s.py model_dir/modelname.h5

Alternatively, the test script can be edited just to predict the spikes given a calcium signal. For this, create a load_dataset function for the given calcium signal in dataset.py and run the test script until predictions.

If you find our work useful, please cite the following publication:

@article{sebastian2021signal,
  title={Signal-to-signal neural networks for improved spike estimation from calcium imaging data},
  author={Sebastian, Jilt and Sur, Mriganka and Murthy, Hema A and Magimai-Doss, Mathew},
  journal={PLoS Computational Biology},
  volume={17},
  number={3},
  pages={e1007921},
  year={2021},
  publisher={Public Library of Science San Francisco, CA USA}
}

