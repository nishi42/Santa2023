"""
Main solver for Santa's Stolen Sleigh problem
"""

import os
import sys
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from tqdm.notebook import tqdm

puzzle_input = pd.read_csv('/content/drive/MyDrive/kaggle/Santa2023/puzzle_info.csv')
puzzle_df = pd.read_csv('/content/drive/MyDrive/kaggle/Santa2023/puzzles.csv')
sample_df = pd.read_csv('/content/drive/MyDrive/kaggle/Santa2023/sample_submission.csv')
puzzles_data_path = '/content/drive/MyDrive/kaggle/Santa2023/'

# Define the puzzle sizes
cube_sizes = [2]
# modelのリストを取得します。
model_files = [os.path.join(puzzles_data_path, f'rubik_cube_{size}_model.pth') for size in cube_sizes]

for i in tqdm(range(len(cube_sizes))):
    cube_size = cube_sizes[i]
    cube_type = f'cube_{cube_size}/{cube_size}/{cube_size}'
    print(cube_type)

    # Define operations
    tar_puzzle = puzzle_input[puzzle_input['puzzle_type'] == cube_type]
    operations = ast.literal_eval(tar_puzzle["allowed_moves"].values[0])
    # Add inverse operations
    op_keys = list(operations.keys())

    for k in op_keys:
        ops = operations[k]
        ops_inv = np.argsort(operations[k]).tolist()
        operations['-'+k] = ops_inv

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load the model
    model = RubikCubePredictMoves1DCNN(cube_size*cube_size*6)  # Replace with your model class

    # Load the state dict
    state_dict = torch.load(model_files[i], map_location=torch.device(device))

    # Update the model's state dict
    model.load_state_dict(state_dict)
    model.eval()

    # Solve the puzzles
    for idx, val in tqdm(puzzle_df[puzzle_df['puzzle_type'] == cube_type].iterrows(),
                         total = puzzle_df[puzzle_df['puzzle_type'] == cube_type].shape[0]):
        initial_state = val['initial_state']
        solution_state = val['solution_state']
        wild_card = val['num_wildcards']
        initial_state = preprocess_state(initial_state)
        solution_state = preprocess_state(solution_state)
        tar_id = val['id']
        base_sequence = sample_df[sample_df['id'] == tar_id]['moves'].values[0].split('.')
        # Solve the puzzle
        deep_sequence = a_star_search(initial_state, solution_state, operations,
                                      model, len(base_sequence), False, wild_card)
        if deep_sequence:
            last_state = apply_sequence_of_operations(initial_state, operations, deep_sequence)
            if is_solved_state_with_wildcard(last_state, solution_state, wild_card) and len(deep_sequence) < len(base_sequence):
                print("Solved with less moves")
                sample_df.loc[sample_df['id'] == tar_id, 'moves'] = '.'.join(deep_sequence)
    sample_df.to_csv(os.path.join(puzzles_data_path, 'my_submission.csv'), index=False)