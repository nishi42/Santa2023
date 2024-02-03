"""
Generative model for N-cube puzzle
"""
import pandas as pd
import numpy as np
import random
import ast

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from tqdm.notebook import tqdm

from utils import cube_N_state_mapper, encode_state, preprocess_state, get_N, puzzle2table, get_n_size_operations, puzzle_operaions

# Load Data
puzzle_input = pd.read_csv('/content/drive/MyDrive/kaggle/Santa2023/puzzle_info.csv')
puzzle_df = pd.read_csv('/content/drive/MyDrive/kaggle/Santa2023/puzzles.csv')
sample_df = pd.read_csv('/content/drive/MyDrive/kaggle/Santa2023/sample_submission.csv')


def show_cube(cube, N=33):
    for i in range(2*N):
        for j in range(3*N):
            print(cube[i][j], end=' ')
        print()

def generate_random_sequence(operations, length=10):
    return [random.choice(list(operations.keys())) for _ in range(length)]


MAX_N = 33
puzzle_id = 0
N = get_N(puzzle_id, puzzle_df)

rubik_table = puzzle2table(puzzle_id, puzzle_df, N, MAX_N)

def scramble_cube_with_memo(initial_state, num_moves, operations, memo):
    state = initial_state.copy()
    move_sequence = []
    for _ in range(num_moves):
        move = random.choice(list(operations.keys()))
        move_sequence.append(move)
        state = apply_operation(state, operations[move])

        # Convert state to a tuple for use as a dictionary key
        state_key = tuple(state)
        if state_key in memo:
            # Compare and store the minimum number of moves
            memo[state_key] = min(memo[state_key], len(move_sequence))
        else:
            memo[state_key] = len(move_sequence)

    return state

def make_n_size_rubik_cube_data(N, DATA_SIZE=10000, MAX_MOVES=50):
    # N: puzzle size
    # DATA_SIZE: number of data
    # MAX_MOVES: maximum number of moves
    # return: dataset

    dataset = []
    memo = {}
    letters = ['A', 'B', 'C', 'D', 'E', 'F']
    initial_state = [letter for letter in letters for _ in range(N*N)]
    operations = get_n_size_operations(N, puzzle_operaions)
    for _ in tqdm(range(DATA_SIZE)):
        num_moves = random.randint(1, MAX_MOVES)
        scramble_cube_with_memo(initial_state, num_moves, operations, memo)
    return pd.DataFrame.from_dict(memo, orient='index', columns=['moves'])

df = make_n_size_rubik_cube_data(3)

