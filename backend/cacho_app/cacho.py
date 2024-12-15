import os
import sys
from game.game import Player, HumanPlayer

class GameState(object):
    def __init__(self):
        self.human_player_hand = [1, 1, 1, 1, 1]
        self.calls = []

    def add_call(self, quantity, number):
        self.calls.append((quantity, number))


class GamePlay(object):
    def __init__(self, num_players, num_games):
        self.num_players = num_players
        self.num_games = num_games
        self.game_state = GameState()

    def initialize_game(self):
        print(f'Starting game')
        return self.game_state

    def process_game_turn(self, quantity, number):
        self.game_state.add_call(quantity=quantity, number=number)
        return self.game_state


