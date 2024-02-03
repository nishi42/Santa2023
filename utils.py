"""
Utility functions for the project
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

def cube_N_state_mapper(state, N=2):
    """
    Maps the cube state character to its corresponding color value.

    Parameters:
    s (str): The cube state character.
    N (int): The size of the cube.

    Returns:
    int: The color value corresponding to the cube state character.
    """
    cube_state = [char for char in 'ABCDEF' for _ in range(N * N)]                                                                                   nnnnnnnnnnnnnn}
    color_mapping ={'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6}
    
    return [cube_state[int(n)] for s in state]

def encode_state(state):
    """
    Encodes the given state into a list of color mappings.

    Parameters:
    state (str): The state to be encoded.

    Returns:
    list: The encoded state as a list of color mappings.
    """
    color_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}

    return [color_mapping[color] for color in state]

def preprocess_state(state):
    """
    Preprocesses the given state.

    Args:
        state (str or list): The state to be preprocessed.

    Returns:
        str: The preprocessed state.
    """
    if ';' in state:
        state = state.split(';')

    N = int((len(state) / 6) ** 0.5)

    if 'N1' in state:
        state = cube_N_state_mapper(state, N)

    state = encode_state(state)

    return state

def apply_operation(state, operation):
    """
    Apply the given operation to the state.

    Args:
        state (list): The current state.
        operation (list): The operation to apply.

    Returns:
        list: The updated state after applying the operation.
    """
    return [state[i] for i in operation]

def apply_sequence_of_operations(state, operations, sequence):
    """
    Apply a sequence of operations to a given state.

    Args:
        state: The initial state.
        operations: A dictionary mapping operation names to functions.
        sequence: A list of operation names.

    Returns:
        The final state after applying the sequence of operations.
    """
    for op in sequence:
        state = apply_operation(state, operations[op])
    return state
    
def is_solved_state(state, solution_state):
    """
    Check if the given state is the solution state.

    Parameters:
    state (any): The state to be checked.
    solution_state (any): The solution state to compare with.

    Returns:
    bool: True if the state is the solution state, False otherwise.
    """
    return state == solution_state

def puzzle2table(puzzle_id, puzzle_df, N=2, MAX_N=33):
    """
    Convert a Rubik's cube puzzle to a 2D table representation.

    Args:
        puzzle_id (int): The ID of the puzzle.
        puzzle_df (DataFrame): The DataFrame containing the puzzle data.
        N (int, optional): The size of each side of the Rubik's cube. Defaults to 2.
        MAX_N (int, optional): The maximum size of the 2D table. Defaults to 33.

    Returns:
        list: A 2D table representing the Rubik's cube puzzle.
    """
    two_dim_puzzle = [[0 for _ in range(3*MAX_N)] for _ in range(2*MAX_N)]

    tar_puzzle = puzzle_df[puzzle_df['id'] == puzzle_id]['solution_state'][0].split(';')

    for i in range(6):
        for j in range(N*N):
            x_col = 33*(i//N) + j % N
            y_row = 33*(i%2) + j // N
            cube_color = tar_puzzle[i * N * N + j]
            int_cube_color = cube_N_state_mapper(cube_color, N)
            two_dim_puzzle[y_row][x_col] = int_cube_color

    #show_cube(two_dim_puzzle)
    return two_dim_puzzle

def get_N(puzzle_id, puzzle_df):
    """
    Get the size of the Rubik's cube puzzle.

    Parameters:
    puzzle_id (int): The ID of the puzzle.
    puzzle_df (DataFrame): The DataFrame containing puzzle information.

    Returns:
    int: The size of the Rubik's cube puzzle.
    """
    return int(puzzle_df[puzzle_df['id'] == puzzle_id]['puzzle_type'].values[0].split('/')[-1])

def get_operations(puzzle_id, puzzle_df, puzzle_operaions):
    """
    Get the allowed moves for a puzzle.

    Parameters:
    puzzle_id (int): The ID of the puzzle.
    puzzle_df (DataFrame): The DataFrame containing puzzle information.
    puzzle_operaions (DataFrame): The DataFrame containing puzzle operations.

    Returns:
    list: A list of allowed moves for the puzzle.
    """
    # Get allowed_moves to the puzzle
    puzzle_type = puzzle_df[puzzle_df['id'] == puzzle_id]['puzzle_type'].values[0]
    allowed_moves = puzzle_operaions[puzzle_operaions['puzzle_type'] == puzzle_type]['allowed_moves'].values[0]
    return ast.literal_eval(allowed_moves)

def get_n_size_operations(N, puzzle_operaions):
    """
    Get the allowed moves for a puzzle of size N.

    Parameters:
    N (int): The size of the puzzle.
    puzzle_operaions (DataFrame): The DataFrame containing puzzle operations.

    Returns:
    list: The list of allowed moves for the puzzle.
    """
    # Get allowed_moves to the puzzle
    puzzle_type = f'cube_{N}/{N}/{N}'
    allowed_moves = puzzle_operaions[puzzle_operaions['puzzle_type'] == puzzle_type]['allowed_moves'].values[0]
    return ast.literal_eval(allowed_moves)