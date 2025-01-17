import os
import sys
from game.game import Player, HumanPlayer
import random
import numpy as np
from time import time


# Set global variables
NUM_HUMANS = 1
NUM_BOTS = 3
NUM_PLAYERS = NUM_HUMANS + NUM_BOTS
MAX_DICE_PER_PLAYER = 5
MAX_TOTAL_DICE = NUM_PLAYERS * MAX_DICE_PER_PLAYER
DICE_SIDES = 6


class GamePlay(object):
    def __init__(self, num_players, num_games, verbose=False):
        self.num_players = num_players
        self.num_games = num_games
        self.player_list = []
        self.verbose = verbose

    def run_all_rounds(self):
        round = 0
        starting_player_index = 0  # Player 1 starts first round
        rotated_player_list = self.player_list

        while len(rotated_player_list) > 1:
            round = round + 1
            # Update trustability, zero out round-specific memory
            round_updates(rotated_player_list, total_dice_left)
            # Rotate player list, start with the starting_player_index
            rotated_player_list = rotated_player_list[
                                  starting_player_index % len(rotated_player_list):] + rotated_player_list[
                                                                                       :starting_player_index]
            # Pick a direction of play
            # If its a bot, pick at random, if its a human let them pick
            if len(rotated_player_list) > 2:  # Direction doesn't matter if there's only 2 players
                direction_change = 0
                if rotated_player_list[0].player_type == 'Bot':
                    direction_change = random.randint(0, 1)
                # If its a human, ask what direction it wants to pick
                elif rotated_player_list[0].player_type == 'Human':
                    input_denied = True
                    direction_change = None
                    while input_denied:
                        direction_change_text = input(
                            '\nYou start. Your hand: {}. Do you want a direction change? (Type \'N\' to go to Player {}, '
                            'and \'Y\' to go to player {}) '.format(rotated_player_list[0].hand,
                                                                    rotated_player_list[1].playerID,
                                                                    rotated_player_list[
                                                                        len(rotated_player_list) - 1].playerID))
                        if direction_change_text in ['Y', 'N']:
                            input_denied = False
                            if direction_change_text == 'Y':
                                direction_change = 1
                            else:
                                direction_change = 0
                        else:
                            print('Input denied. Type \'Y\' or \'N\'')
                else:
                    print('Player type = {}'.format(rotated_player_list[0].player_type))
                if direction_change:
                    rotated_player_list = [rotated_player_list[0]] + list(reversed(rotated_player_list[1:]))

            # Start round
            if verbose:
                print("\n ------------- ROUND {} ----------- ".format(str(round)))
            # Print out player order
            if verbose:
                if NUM_HUMANS == 0:
                    for player in rotated_player_list:
                        if player.player_type == 'Bot':
                            print(f'Bot Player #{player.playerID}: {player.hand}, {player.num_dice_unseen} unseen dice')
                elif NUM_HUMANS == 1:
                    for player in rotated_player_list:
                        if player.player_type == 'Bot':
                            hidden_hand = ['X'] * len(player.hand)
                            print(f'Bot Player #{player.playerID}: {hidden_hand}, {player.num_dice_unseen} unseen dice')
                        else:
                            print(
                                f'Human Player #{player.playerID}: {player.hand}, {player.num_dice_unseen} unseen dice')
            i = 0
            prev_a = None
            end_round = False
            number_counter = [0] * DICE_SIDES  # To keep track of what's been said
            while not end_round:
                # Use % to loop i.e.:
                for index, player in enumerate(rotated_player_list):
                    if NUM_HUMANS > 0:  # If a human is playing, delay the bots' plays for a better playing experience
                        print('...')
                        time.sleep(3)
                    # Indexes and player IDs are not the same, since the order changes almost every round
                    previous_index = index - 1 % len(rotated_player_list)
                    previous_player = rotated_player_list[previous_index]
                    index_current = index

                    if verbose:
                        print(f'{Color.BOLD}\nPlayer ID: {player.playerID}{Color.END}')
                    if player.player_type == 'Human':
                        print(f'{Color.BOLD}Hand: {player.hand}{Color.END}')
                        a = player.action(prev_a, plot=False)
                    else:
                        a = player.action(prev_a, plot=False)

                    # Set call metadata

                    last_play = prev_a
                    prev_a = a

                    if verbose:
                        if a['bs']:  # If called Bullshit
                            print(f"{Color.BOLD}Turn {i}: DUDO{Color.END}\n")
                        elif a['exactly']:  # If called Exactly
                            print(f"{Color.BOLD}Turn {i}: EXACTLY{Color.END}\n")
                        else:
                            print(f"{Color.BOLD}Turn {i}: {a['quantity']} {a['dice']}s{Color.END}\n")

                    i += 1

                    # user_enter = input('Enter to continue: ')
                    if a['bs'] is True:
                        last_play = a
                        bullshit_caller_player_list_index = index
                        break
                    if a['exactly'] is True:
                        last_play = a
                        exactly_caller_player_list_index = index
                        break

                    number_counter[a['dice'] - 1] += 1
                    for other_player in rotated_player_list:
                        if other_player == player or other_player.player_type == 'Human':
                            continue
                        other_player.calculate_conditional_distributions(cumulative_calls_list=number_counter,
                                                                         use_beta_updating=use_beta_updating)

                end_round = a['bs'] | a['exactly']

            # when bullshit is called count the player hands to determine the outcomes
            # there are 2 outcomes, player who called bullshit loses a die or player who had bullshit called on loses a die
            # figure out last hand
            # then count

            if NUM_HUMANS > 0:  # Reveal everyones hands
                print('\nREVEAL:')
                for player in rotated_player_list:
                    print(f'{player.player_type} Player #{player.playerID}: {player.hand}')

            player_starts_next_round = None
            if last_play['bs']:
                count = 0
                dice_counts = Counter([number for player in rotated_player_list for number in player.hand])
                total_count = int(dice_counts[1]) if last_play['dice'] == 1 else (dice_counts[last_play['dice']]) + int(
                    dice_counts[1])
                player_bullshit_called_on = rotated_player_list[
                    (bullshit_caller_player_list_index - 1) % len(rotated_player_list)]

                if total_count < last_play['quantity'] and player_bullshit_called_on is not None:
                    rotated_player_list[previous_index].hand = rotated_player_list[previous_index].hand[1:]
                    if verbose:
                        print(
                            f'{total_count} {last_play["dice"]}s total < Player {rotated_player_list[previous_index].playerID}\'s bet of {last_play["quantity"]} {last_play["dice"]}s')
                        print(f'Player {rotated_player_list[previous_index].playerID} loses a die')
                    total_dice_left = total_dice_left - 1
                    starting_player_index = previous_index
                elif total_count >= last_play['quantity'] and player_bullshit_called_on != None:
                    rotated_player_list[index_current].hand = rotated_player_list[index_current].hand[1:]
                    if verbose:
                        print(
                            f'{total_count} {last_play["dice"]}s total >= Player {rotated_player_list[previous_index].playerID}\'s bet of {last_play["quantity"]} {last_play["dice"]}s')
                        print("Player " + str(rotated_player_list[index_current].playerID) + " loses a die")
                    total_dice_left = total_dice_left - 1
                    starting_player_index = index_current
            elif last_play['exactly']:
                count = 0
                dice_counts = Counter([number for player in rotated_player_list for number in player.hand])
                total_count = int(dice_counts[1]) if last_play['dice'] == 1 else (dice_counts[last_play['dice']]) + int(
                    dice_counts[1])

                if total_count == last_play['quantity']:
                    if verbose:
                        print(f'{total_count} {last_play["dice"]}s total == Player '
                              f'{rotated_player_list[index_current].playerID}\'s exactly bet of '
                              f'{last_play["quantity"]} {last_play["dice"]}s')
                    if len(rotated_player_list[index_current].hand) < 5:
                        rotated_player_list[index_current].hand = np.append(rotated_player_list[index_current].hand,
                                                                            0)  # Add a die
                        total_dice_left = total_dice_left + 1
                        if verbose:
                            print(f'Player {rotated_player_list[index_current].playerID} wins a die')
                        starting_player_index = index_current
                    else:
                        if verbose:
                            print(
                                f'Player {rotated_player_list[index_current].playerID} has 5 die so didn\'t gain a die')
                        starting_player_index = index_current
                else:
                    rotated_player_list[index_current].hand = rotated_player_list[index_current].hand[
                                                              1:]  # Remove a die
                    total_dice_left = total_dice_left - 1
                    if verbose:
                        print(
                            f'{total_count} {last_play["dice"]}s total != Player {rotated_player_list[index_current].playerID}\'s exactly bet of {last_play["quantity"]} {last_play["dice"]}s')
                        print(f'Player {rotated_player_list[index_current].playerID} loses a die')
                    starting_player_index = index_current

            ##################################################
            rotated_player_list = [player for player in rotated_player_list if player.hand.size > 0]

            # If player who starts next round is out, the next remaining player starts
            starting_player_index = starting_player_index % len(rotated_player_list)

            for player in rotated_player_list:
                player.hand = np.random.randint(1, DICE_SIDES + 1, player.hand.size)
                player.calculate_cond_dist(num_dice_unseen=total_dice_left - player.hand.size)

        game_metadata = {'game_playerid_winner': rotated_player_list[0].playerID,
                         'player_metadata': player_metadata}


    def run_game(self, verbose=False):
        player_list = self.instantiate_players()
        if verbose:
            print('Player order: {}'.format([(p.playerID, p.player_type) for p in player_list]))

        self.run_all_rounds()

    def instantiate_players(self):
        # instantiate players
        r = 0.35  # risk / bullshit threshold
        l = 0.8  # likely threshold
        e = 0.3  # exactly threshold
        bt = 0.65  # bluff threshold
        bp = 0.25  # bluff probability
        trustability = 1.0  # trustability
        player_list = []
        total_dice_left = MAX_TOTAL_DICE

        player_types = ['BOT'] * NUM_BOTS + ['HUMAN'] * NUM_HUMANS
        player_metadata = []
        call_metadata = []
        shuffled_player_types = random.sample(player_types, len(player_types))  # Shuffle player types
        for i, player_type in enumerate(shuffled_player_types):
            if i == 0:
                trustability = 0  # PARAMETER OF INTEREST: trustability
            elif i == 1:
                trustability = 1
            elif i == 2:
                trustability = 1.25
            else:
                trustability = 1.75

            h = np.random.randint(1, DICE_SIDES + 1, MAX_DICE_PER_PLAYER)
            if player_type == 'BOT':
                # All players have almost the same params
                p = Player(hand=h, risk_thres=r, likely_thres=l, exactly_thres=e, bluff_prob=bp, bluff_thres=bt,
                           trustability=trustability, playerID=i + 1, num_dice_unseen=total_dice_left - h.size,
                           verbose=self.verbose)
                player_metadata.append(
                    [p.playerID, p.risk_thres, p.likely_thres, p.exactly_thres, p.bluff_prob, p.bluff_thres,
                     p.trustability])
            elif player_type == 'HUMAN':
                p = HumanPlayer(hand=h, playerID=i + 1, trustability=trustability,
                                num_dice_unseen=total_dice_left - h.size)
                player_metadata.append([p.playerID, -1, -1, -1, -1, -1, -1])  # Dummy data for humans
            else:
                # raise ValueError
                p = None  # player_type must be HUMAN or BOT

            player_list.append(p)  # Keeps track of players left in game
        self.player_list = player_list

        return player_list



    def process_game_turn(self, quantity, number):
        self.game_state.add_call(quantity=quantity, number=number)
        return self.game_state


if __name__ == '__main__':
    max_games = 1
    gameplay = GamePlay(num_players=NUM_PLAYERS, num_games=max_games)
    gameplay.run_game(verbose=True)


