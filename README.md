## Installation
Install dependencies via Anaconda
```
conda env create -f environment.yml
```
Activate the environment
```
source activate RS-Rainbow
```
## Training
Train RS-Rainbow with softmax normalization and default settings
```
python main.py
```
Train RS-Rainbow with sigmoid normalization, on game ms_pacman, for a maximum 200 million environment steps
```
python main.py --model-type='DQN_sig' --game='ms_pacman' --T-max=200000000
```
The list of available games is [here](https://github.com/openai/atari-py/tree/master/atari_py/atari_roms). Valid values for the `--model-type` argument are 'DQN', 'DQN_rs', and 'DQN_rs_sig', corresponding to models Rainbow, RS-Rainbow with softmax normalization, and RS-Rainbow with sigmoid normalization in the paper. The default is 'DQN_rs'.

## Testing
Evaluate a trained RS-Rainbow agent. Specify the model weights directory with `--model` and the number of test episodes with `--evaluation-episodes`. Use `--game` and `--model-type` accordingly. 
```
python main.py --evaluate --evaluation-episodes=200 --model='./results/ms_pacman_DQN_rs_model.pth'
```

## Visualization
Using Example:
'''
python visualize.py --game='ms_pacman' --model='results/ms_pacman_DQN_rs_model.pth' --model-type='DQN_rs' --folder='results' --multiply --mask --seed=123 --attribution-method='SG' --action=3
'''


## Acknowledgement
We thank the authors of this [repository](https://github.com/Kaixhin/Rainbow) for making their implementation of Rainbow publicly available. Our code is adapted from their implementation.
