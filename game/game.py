import random
import numpy as np

from math import ceil

from collections import Counter
from typing import List
from scipy.stats import binom, beta
import matplotlib.pyplot as plt
import sys
import time
from datetime import datetime
import csv

# Set global variables
NUM_HUMANS = 1
NUM_BOTS = 3
NUM_PLAYERS = NUM_HUMANS + NUM_BOTS
MAX_DICE_PER_PLAYER = 3
MAX_TOTAL_DICE = NUM_PLAYERS * MAX_DICE_PER_PLAYER
DICE_SIDES = 6

# random.seed(123)
# Set round-specfic variables
player_dice_left = MAX_DICE_PER_PLAYER
total_dice_left = MAX_TOTAL_DICE
unseen_dice_left = total_dice_left - player_dice_left  # 20 total dice of which 5 dice are seen and the rest unseen
player_hand = sorted([random.randint(1, DICE_SIDES) for x in range(player_dice_left)])


class Player:
    def __init__(self, hand: List[int], risk_thres: float, likely_thres: float,
                 exactly_thres: float, bluff_thres: float, bluff_prob: float,
                 trustability: float, playerID: int, num_dice_unseen: int,
                 verbose: int = 0):
        """
    Players have a hand, which is a list of dice values, which vary between
    1 and DICE_SIDES i.e. 6. If the conditional probability of a callable bet
    is larger than likely_thres, then the player always plays that. If there is
    no callable bet with probability >= likely_thresh, then the player may call
    bullshit on the received bet. A player only calls bullshit if the
    conditional probability of the received bet is less than risk_thresh.
    If the received bet has probability larger than risk_thresh, then the player
    either: (A) raises the bet to the next callable bet that has highest conditional
    probability, OR (B) calls 'exactly' if the conditional probability of 'exactly' is higher than the
    conditional probability of the highest bet, OR (C) with some probability bluff_prob, bets a bluff,
    which is a call with conditional probability at least bluff_thres.
    """
        self.verbose = verbose
        self.hand = hand
        self.risk_thres = risk_thres
        self.likely_thres = likely_thres
        self.exactly_thres = exactly_thres
        self.bluff_thres = bluff_thres
        self.bluff_prob = bluff_prob
        self.trustability = trustability
        self.playerID = playerID
        self.player_type = 'Bot'
        self.num_dice_unseen = num_dice_unseen
        self.num_dice_in_game = num_dice_unseen + len(self.hand)
        self.conditional_dist = None
        self.exactly_dist = None

        self.calculate_cond_dist(num_dice_unseen)
        self.calculate_exactly_dist(num_dice_unseen)

    def calculate_conditional_distributions(self, cumulative_calls_list: List[int]):
        """
      Given this player's hand and the number of unseen dice, return a list of
      length equal to the number of sides of a die (i.e. 6), where each of these
      lists contains the probabilities of seeing a count of at least X in the game.

      Usage: self.calculate_conditional_distributions()

      """

        counts_of_numbers = Counter(self.hand)
        conditional_number_probs = []
        cumulative_calls_for_list = []
        cumulative_calls_against_list = []
        for number in range(1, DICE_SIDES + 1):
            # count quantity of dice in hand
            if number in list(counts_of_numbers.keys()):
                count_in_hand = counts_of_numbers[number]
            else:
                count_in_hand = 0

            # Include aces in the count for non-aces
            if number != 1 and 1 in list(counts_of_numbers.keys()):
                count_in_hand += counts_of_numbers[1]

            # Aces are wild
            if number == 1:
                number_prob = 1 / 6
                cumulative_calls_for = cumulative_calls_list[number - 1]
                cumulative_calls_against = (1 / 2) * number_prob * sum(cumulative_calls_list[1:])
            else:
                number_prob = 1 / 3
                cumulative_calls_for = cumulative_calls_list[number - 1] + cumulative_calls_list[1 - 1]
                if number < 6:
                    cumulative_calls_against = number_prob * sum(
                        cumulative_calls_list[1:number - 1] + cumulative_calls_list[number:])
                else:
                    cumulative_calls_against = (number_prob) * sum(cumulative_calls_list[1:5])

            cumulative_calls_for_list.append(cumulative_calls_for)
            cumulative_calls_against_list.append(cumulative_calls_against)
            alpha = (self.num_dice_unseen * number_prob) + (self.trustability * cumulative_calls_for)
            beta = self.num_dice_unseen * (1 - number_prob) + (self.trustability * cumulative_calls_against)
            conditional_number_prob = alpha / (alpha + beta)
            conditional_number_probs.append(conditional_number_prob)

        # Standardize number probs so sum equals 1.85 = 1/6 + 5 * (1/3)
        # sum_to_one = np.sum([prob for prob in conditional_number_probs])
        # conditional_number_probs_st = np.divide(conditional_number_probs,
        #                                         np.sum(conditional_number_probs) / ((1/6) + (5 * (1/3))))
        conditional_number_probs_st = conditional_number_probs
        conditional_distributions_list_of_lists = []
        for i, number in enumerate(range(1, DICE_SIDES + 1)):
            if self.verbose > 1:
                print(
                    f'{self.playerID}: d = {number}: calls for / calls against = {cumulative_calls_for_list[i]} / {cumulative_calls_against_list[i]} -> prob = {conditional_number_probs[i]}')

            # These are the unconditional probabilities of the unseen dice
            # TODO: still need to standardize
            unconditional_probabilities = [
                1 - binom.cdf(n=self.num_dice_unseen, p=conditional_number_probs_st[number - 1], k=x) + \
                binom.pmf(n=self.num_dice_unseen, p=conditional_number_probs_st[number - 1], k=x) \
                for x in range(1, self.num_dice_unseen + 1)]

            # print(f'unseen probs: {len(unconditional_probabilities)} -> {unconditional_probabilities}')

            # Initialize conditional probability to 1.0 if you have them in your hand,
            #   initialize to -1.0 for those not in your hand (to easily spot an error),
            #   and initialize to 0.0 those that remain, so that the domain is
            #   self.num_dice_unseen + len(self.hand) (i.e. 15 + 5 = 20)
            conditional_probabilities = [1.0] * (count_in_hand + 1) + \
                                        [-1.0] * len(unconditional_probabilities) + [0.0] * (
                                                    len(player_hand) - count_in_hand)
            for i, prob in enumerate(unconditional_probabilities):
                if i > self.num_dice_unseen + len(self.hand):
                    break
                conditional_probabilities[i + count_in_hand + 1] = unconditional_probabilities[i]

            conditional_distributions_list_of_lists.append(conditional_probabilities)

            # print(f'cond probs: {len(conditional_probabilities)} -> {conditional_probabilities}')

        return conditional_distributions_list_of_lists

    def recalculate_cond_dist(self, cumulative_calls_list: List[int]):
        c = self.calculate_conditional_distributions(cumulative_calls_list=cumulative_calls_list)
        self.conditional_dist = np.array(c)
        self.num_dice_unseen = num_dice_unseen
        self.num_dice_in_game = num_dice_unseen + len(self.hand)

    def calculate_cond_dist(self, num_dice_unseen):
        c = get_conditional_distributions(self.hand, num_dice_unseen)
        self.conditional_dist = np.array(c)
        self.num_dice_unseen = num_dice_unseen
        self.num_dice_in_game = num_dice_unseen + len(self.hand)

    def calculate_exactly_dist(self, num_dice_unseen):
        e = get_exactly_distributions(self.hand, num_dice_unseen)
        self.exactly_dist = np.array(e)

    def _get_highest_probability_call(self, m):
        max_prob = np.max(m)
        # TODO: what if multiple options have the same probability?
        idx_flat = np.argmax(m)
        idx = np.unravel_index(idx_flat, m.shape)
        return max_prob, idx

    def _get_aggressive_start(self, m):
        mask = m > 0.54
        candidate_indices = np.where(mask)
        combined_indices = list(zip(candidate_indices[0], candidate_indices[1]))
        aggressive_idx = random.choice(combined_indices)
        aggresive_prob = m[aggressive_idx[0], aggressive_idx[1]]
        # largest_quantity = 1
        # selected_call_index = None
        # for i, candidate_quantity in enumerate(candidate_indices[1]):
        #     if candidate_quantity > largest_quantity:
        #         largest_quantity = candidate_quantity
        #         selected_call_index = i
        #
        # aggressive_idx = combined_indices[selected_call_index]
        # aggresive_prob = m[aggressive_idx[0], aggressive_idx[1]]

        return aggresive_prob, aggressive_idx

    def _get_bluff_call(self, m):
        # zero out highest probability call
        max_prob, idx = self._get_highest_probability_call(m)
        m[idx[0], idx[1]] = 0

        # zero out calls that don't pass bluff threshold
        mask = m > self.bluff_thres
        candidate_indices = np.where(mask)
        combined_indices = list(zip(candidate_indices[0], candidate_indices[1]))

        if len(combined_indices) == 0:
            return None, None

        # pick out a call at random
        bluff_idx = random.choice(combined_indices)
        bluff_prob = m[bluff_idx[0], bluff_idx[1]]
        return bluff_prob, bluff_idx

    def first_action(self, aggressive_start=True):
        cda_mat = self.conditional_dist[:, 1:].copy()
        call_prob, idx = self._get_highest_probability_call(cda_mat)
        if aggressive_start:
            call_prob, idx = self._get_aggressive_start(cda_mat)
        return {'quantity': idx[1] + 1, 'dice': idx[0] + 1, 'bs': False, 'exactly': False}

    def action(self, prev_action=None, plot=False):
        if prev_action == None:
            return self.first_action()

        ## create matrix with set of impossible calls zero-ed out
        d = prev_action['dice']
        q = prev_action['quantity']
        cda_mat = self.conditional_dist[:, 1:].copy()  # remove at least 0 col
        exa_prob = self.exactly_dist[d - 1, q]  # no column removal

        if d == 1:
            # here I removed , self.num_dice_in_game, this means if there are only a couple die
            # in the game it would not need to double + 1 which would be a problem theoretically

            # zero-out un-allowed non-ace calls
            cda_mat[1:, :min(self.num_dice_in_game, q * 2 + 1) - 1] = -1.0

            # Zero-out un-allowed ace calls
            cda_mat[0, :min(self.num_dice_in_game, q)] = -1.0

        else:
            # If q = 5 and d = 2, i.e. if I call 5 2's, you can still call 5 3's.
            # So, the min quantity the next person can call is still q = 5.
            # This is true unless d = 6, which is the highest number, but that's OK.

            # Zero out unallowed non-ace calls
            # any non-ace call w/ Q <= q - 1
            cda_mat[1:, :max(min(self.num_dice_in_game, q - 1), 0), ] = -1.0
            # any D <= d w/ Q = q
            cda_mat[1:d, min(self.num_dice_in_game, q - 1)] = -1.0

            # Zero out unallowed ace calls
            cda_mat[0, :ceil(q / 2) - 1] = -1.0

        if self.verbose > 1:
            print('Probabilities: {}'.format(cda_mat))
            if plot:
                plot_distributions(cond_distributions=self.conditional_dist,
                                   player_id=self.playerID)

        # TODO: check bullshit probability before bluffing: if what the other says is very unlikely, don't bluff
        # decide on whether or not to bluff (random)
        decide_bluff = np.random.binomial(n=1, p=self.bluff_prob, size=1)[0]
        if decide_bluff == 1:
            bluff_prob, b_idx = self._get_bluff_call(cda_mat)
            if b_idx:
                if self.verbose > 1:
                    print(f'Bluff: Probability of at least {b_idx[1] + 1} {b_idx[0] + 1}\'s: {bluff_prob}')
                bluff_action = {'quantity': b_idx[1] + 1, 'dice': b_idx[0] + 1, 'bs': False, 'exactly': False}
                return bluff_action

        # find highest probability call
        max_prob, idx = self._get_highest_probability_call(cda_mat)
        if max_prob <= 0:  # If highest probability play is 0.0, must call bullshit
            return {'quantity': q, 'dice': d, 'bs': True, 'exactly': False}
        else:
            if self.verbose > 1:
                print(f'Probability of at least {idx[1] + 1} {idx[0] + 1}\'s: {max_prob}')
            best_action = {'quantity': idx[1] + 1, 'dice': idx[0] + 1, 'bs': False, 'exactly': False}

        # evaluate against likely threshold
        if max_prob > self.likely_thres:
            return best_action

        # check risk threshold
        prob_prev_action = self.conditional_dist[d - 1][q]
        prob_bullshit_wins = 1 - prob_prev_action
        if prob_prev_action < self.risk_thres:
            if self.verbose > 1:
                print(f'Probability of at least {q} {d}\'s = {prob_prev_action} < {self.risk_thres} -> BS')
            return {'bs': True, 'dice': d, 'quantity': q, 'exactly': False}
        if self.verbose > 1:
            print(f'Probability of at least {q} {d}\'s = {prob_prev_action} > {self.risk_thres} -> RAISE')

        # Exactly only an option if number of dice left is at least half of what you started with
        # Call exactly if probability is above threshold, or if its more likely than best action
        if self.num_dice_in_game >= (MAX_TOTAL_DICE / 2):
            if self.verbose > 1:
                print(f'Probability of exactly {q} {d}\'s: {exa_prob}')
            if exa_prob > self.exactly_thres or (exa_prob > max_prob):
                return {'quantity': q, 'dice': d, 'bs': False, 'exactly': True}

        # else, make highest probability call
        return best_action


class HumanPlayer(Player):
    def __init__(self, hand: List[int], playerID: int, num_dice_unseen: int, trustability: float, verbose: int = 0):
        """
    Humans can also play.
    """
        self.playerID = playerID
        self.hand = hand
        self.verbose = verbose
        self.num_dice_unseen = num_dice_unseen
        self.trustability = trustability
        self.player_type = 'Human'

        self.calculate_cond_dist(num_dice_unseen)
        self.calculate_exactly_dist(num_dice_unseen)

    def action(self, prev_action=None, plot=False):
        if plot:
            plot_distributions(cond_distributions=self.conditional_dist,
                               player_id=self.playerID)
        invalid_call = True
        action = None
        while invalid_call:
            player_call = input('Play Format: \{quantity\} \{number\} OR \'bs\' OR \'exactly\':    ')
            if prev_action is not None:  # Not first play
                if player_call == 'bs':  # BS
                    invalid_call = False
                    action = prev_action
                    action.update({'bs': True})
                elif player_call == 'exactly':  # Exactly
                    if self.num_dice_unseen + self.hand.size >= MAX_TOTAL_DICE / 2:
                        invalid_call = False
                        action = prev_action
                        action.update({'exactly': True})
                    else:
                        invalid_reason = 'Not enough dice for exactly.'
                elif len([int(i) for i in player_call.split() if i.isdigit()]) == 2:  # RAISE
                    raise_call = [int(i) for i in player_call.split() if i.isdigit()]
                    quantity = raise_call[0]
                    dice = raise_call[1]
                    if dice < 1 or dice > 6:
                        invalid_reason = 'Dice must be within 1 and 6.'
                    elif prev_action['dice'] == 1:
                        if dice == 1 and quantity > prev_action['quantity']:
                            invalid_call = False
                        elif 2 <= dice <= 6 and quantity >= 2 * prev_action['quantity'] + 1:
                            invalid_call = False
                        else:
                            invalid_reason = f'After {prev_action["quantity"]} aces, lowest call is \
                                           {prev_action["quantity"] + 1} aces or {2 * prev_action["quantity"] + 1} 2s'
                    elif 2 <= prev_action['dice'] <= 6:
                        if dice == 1 and quantity >= prev_action['quantity'] / 2:
                            invalid_call = False
                        elif quantity == prev_action['quantity'] and dice >= prev_action['dice']:
                            invalid_call = False
                        elif quantity > prev_action['quantity']:
                            invalid_call = False
                        else:
                            if prev_action['dice'] == 6:
                                invalid_reason = f'After {prev_action["quantity"]} 6s, lowest call is \
                                               {prev_action["quantity"] + 1} 2s or {int(np.ceil(prev_action["quantity"] / 2))} 1s'
                            else:
                                invalid_reason = f'After {prev_action["quantity"]} {prev_action["dice"]}s, lowest call is \
                                                {prev_action["quantity"]} {prev_action["dice"] + 1}s or \
                                                {int(np.ceil(prev_action["quantity"] / 2))} 1s'
                    else:
                        invalid_reason = 'Dice must be within 1 and 6.'

                    if invalid_call:
                        pass
                    else:
                        action = {'quantity': quantity, 'dice': dice, 'bs': False, 'exactly': False}

                else:  # Not BS, Exactly, or Raise
                    invalid_reason = f'What is that? Try again.'

                if invalid_call:
                    print('ERROR: INVALID CALL. {} Got: {}'.format(invalid_reason, player_call))

            else:  # First play
                if len([int(i) for i in player_call.split() if i.isdigit()]) == 2:  # RAISE
                    raise_call = [int(i) for i in player_call.split() if i.isdigit()]
                    quantity = raise_call[0]
                    dice = raise_call[1]
                    if 1 <= dice <= 6:
                        invalid_call = False
                    else:
                        invalid_reason = 'Dice must be within 1 and 6.'
                else:
                    invalid_reason = 'Must start with 2 integers.'

                if invalid_call:
                    print('ERROR: INVALID CALL. {}. Got: {}'.format(invalid_reason, raise_call))
                else:
                    action = {'quantity': quantity, 'dice': dice, 'bs': False, 'exactly': False}

        return action

    def calculate_cond_dist(self, num_dice_unseen):
        c = get_conditional_distributions(self.hand, num_dice_unseen)
        self.conditional_dist = np.array(c)
        self.num_dice_unseen = num_dice_unseen
        self.num_dice_in_game = num_dice_unseen + len(self.hand)


def get_conditional_distributions(player_hand: List[int], num_dice_unseen: int = 15):
    """
  Given this player's hand and the number of unseen dice, return a list of
  length equal to the number of sides of a die (i.e. 6), where each of these
  lists contains the probabilities of seeing a count of at least X in the game.

  Usage: get_conditional_distributions(player_hand = [5, 2, 3, 4, 3], num_dice_unseen = 10)

  """
    counts_of_numbers = Counter(player_hand)
    conditional_distributions_list_of_lists = []
    for number in range(1, DICE_SIDES + 1):
        # count quantity of dice in hand
        if number in list(counts_of_numbers.keys()):
            count_in_hand = counts_of_numbers[number]
        else:
            count_in_hand = 0

        # Include aces in the count for non-aces
        if number != 1 and 1 in list(counts_of_numbers.keys()):
            count_in_hand += counts_of_numbers[1]

        # Aces are wild
        if number == 1:
            number_prob = 1 / 6
        else:
            number_prob = 1 / 3

        # These are the unconditional probabilities of the unseen dice
        unconditional_probabilities = [1 - binom.cdf(n=num_dice_unseen, p=number_prob, k=x) + \
                                       binom.pmf(n=num_dice_unseen, p=number_prob, k=x) \
                                       for x in range(1, num_dice_unseen + 1)]

        # print(f'unseen probs: {len(unconditional_probabilities)} -> {unconditional_probabilities}')

        # Initialize conditional probability to 1.0 if you have them in your hand,
        #   initialize to -1.0 for those not in your hand (to easily spot an error),
        #   and initialize to 0.0 those that remain, so that the domain is
        #   num_dice_unseen + len(player_hand) (i.e. 15 + 5 = 20)
        conditional_probabilities = [1.0] * (count_in_hand + 1) + \
                                    [-1.0] * len(unconditional_probabilities) + [0.0] * (
                                                len(player_hand) - count_in_hand)
        for i, prob in enumerate(unconditional_probabilities):
            if i > num_dice_unseen + len(player_hand):
                break
            conditional_probabilities[i + count_in_hand + 1] = unconditional_probabilities[i]

        conditional_distributions_list_of_lists.append(conditional_probabilities)

        # print(f'cond probs: {len(conditional_probabilities)} -> {conditional_probabilities}')

    return conditional_distributions_list_of_lists


def get_exactly_distributions(player_hand: List[int], num_dice_unseen: int = 15):
    """
  Given this player's hand and the number of unseen dice, return a list of
  length equal to the number of sides of a die (i.e. 6), where each of these
  lists contains the probabilities of seeing a count of exactly X in the game.

  Usage: get_exactly_distributions(player_hand = [5, 2, 3, 4, 3], num_dice_unseen = 10)

  """
    counts_of_numbers = Counter(player_hand)
    conditional_exactly_distributions_list_of_lists = []
    for number in range(1, DICE_SIDES + 1):
        # count quantity of dice in hand
        if number in list(counts_of_numbers.keys()):
            count_in_hand = counts_of_numbers[number]
        else:
            count_in_hand = 0

        # Include aces in the count for non-aces
        if number != 1 and 1 in list(counts_of_numbers.keys()):
            count_in_hand += counts_of_numbers[1]

        # Aces are wild
        if number == 1:
            number_prob = 1 / 6
        else:
            number_prob = 1 / 3

        # These are the unconditional probabilities of the unseen dice
        unconditional_exactly_probabilities = [binom.pmf(n=num_dice_unseen, p=number_prob, k=x) \
                                               for x in range(0, num_dice_unseen + 1)]

        # print(f'unseen probs: {len(unconditional_exactly_probabilities)} -> {unconditional_exactly_probabilities}')
        # Initialize conditional exactly probability to 0.0 for anything less than what you have in your hand,
        # 1.0 for what you have in your hand,
        #   initialize to -1.0 for those not in your hand (to easily spot an error),
        #   and initialize to 0.0 those that remain, so that the domain is
        #   num_dice_unseen + len(player_hand) (i.e. 15 + 5 = 20)
        conditional_exactly_probabilities = [0] * (count_in_hand) + \
                                            [-1.0] * len(unconditional_exactly_probabilities) + [0.0] * (
                                                        len(player_hand) - count_in_hand)
        for i, prob in enumerate(unconditional_exactly_probabilities):
            if i > num_dice_unseen + len(player_hand):
                break
            conditional_exactly_probabilities[i + count_in_hand] = unconditional_exactly_probabilities[i]

        conditional_exactly_distributions_list_of_lists.append(conditional_exactly_probabilities)

        # print(f'cond probs: {len(conditional_probabilities)} -> {conditional_probabilities}')

    return conditional_exactly_distributions_list_of_lists


def get_conditional_distribution_for_number(player_hand: List[int], number_to_plot: int, num_dice_unseen: int = 10,
                                            wild_prob: float = 1 / 6, other_prob: float = 1 / 3):
    """
    Usage:
    hand = [4, 3, 1, 2, 1]
    number_to_plot = 3
    cond_distribution_3s = get_conditional_distribution_for_number(player_hand=hand, number=number_to_plot, num_dice_unseen=10)
    """
    counts_of_numbers = Counter(player_hand)
    conditional_distributions_list_of_lists = []
    for number in range(1, DICE_SIDES + 1):
        # count quantity of dice in hand
        if number_to_plot in list(counts_of_numbers.keys()):
            count_in_hand = counts_of_numbers[number_to_plot]
        else:
            count_in_hand = 0

        # Include aces in the count for non-aces
        if number_to_plot != 1 and 1 in list(counts_of_numbers.keys()):
            count_in_hand += counts_of_numbers[1]

    print(f'Count in hand: {count_in_hand}')

    number_prob = wild_prob if number_to_plot == 1 else other_prob

    unconditional_probabilities = [
        1 - binom.cdf(n=num_dice_unseen, p=number_prob, k=x) +
        binom.pmf(n=num_dice_unseen, p=number_prob, k=x)
        for x in range(1, num_dice_unseen + 1)
    ]

    print(unconditional_probabilities)

    conditional_probabilities = [1.0] * (count_in_hand) + [-1.0] * len(unconditional_probabilities) + [0.0] * (
                len(player_hand) - count_in_hand)

    for i, prob in enumerate(unconditional_probabilities):
        if i > num_dice_unseen + len(player_hand):
            break
        conditional_probabilities[i + count_in_hand] = prob
    print(conditional_probabilities)

    # Plotting as a red bar graph
    fig, ax = plt.subplots(1, 1)
    plt.bar(range(1, len(conditional_probabilities) + 1), conditional_probabilities, label=f'Number {number_to_plot}',
            color='cornflowerblue')

    plt.xlabel('Count of Dice')
    plt.ylabel('Probability')
    plt.title(f'Conditional Distribution for Number {number_to_plot} in All Dice')
    plt.legend()
    ax.set_xticks(ticks=range(num_dice_unseen + len(player_hand) + 1))
    plt.show()

    return conditional_probabilities


def plot_beta_distribution(a: float, b: float):
    fig, ax = plt.subplots(1, 1)
    x = np.linspace(beta.ppf(0.01, a, b),
                    beta.ppf(0.99, a, b), 100)
    ax.plot(x, beta.pdf(x, a, b), 'black', lw=2, alpha=0.6, label='beta(alpha = {}, beta = {}) PDF'.format(a, b))
    plt.axvline(x=a / (a + b), color='grey', label='E(p)')
    ax.set_xlabel(f'Probability of a 3')
    ax.set_ylabel(f'Density')
    plt.text(0.35, 2.6, f'Expected Value = {round(a / (a + b), 2)}', fontsize=12)
    plt.show()


def plot_distributions(cond_distributions: List[List[float]], player_id: int = 0):
    """
  Plot probabilities.

  Usage:
  cond_dist = get_conditional_distributions(player_hand = [5, 2, 3, 4, 3], num_dice_unseen = 10)
  plot_distributions(cond_distributions = cond_dist)

  """
    fig, ax = plt.subplots()
    ax.set_title(f'P{player_id}: Conditional Probability of at least X = x')
    for i, x in enumerate(cond_distributions):
        ax.set_xlabel(f'X = x (Count of x\'s in All Dice)')
        ax.set_ylabel(f'Probability at least X = x | Hand')
        ax.bar(x=[x - 5 / 16 + (i / 8) for x in range(len(cond_distributions[i]))],
               height=cond_distributions[i],
               width=(1 / 8),
               label=f'd = {i + 1}')
    ax.set_xticks(ticks=range(len(cond_distributions[0])))
    plt.legend()
    plt.show()


def runGame(verbose: int = 0):
    # instantiate players
    r = 0.37  # risk / bullshit threshold
    l = 0.8  # likely threshold
    e = 0.4  # exactly threshold
    bt = 0.5  # bluff threshold
    bp = 0.2  # bluff probability
    trust = 1.0  # trustability
    player_list = []
    total_dice_left = MAX_TOTAL_DICE

    player_types = ['BOT'] * NUM_BOTS + ['HUMAN'] * NUM_HUMANS
    player_metadata = []
    call_metadata = []
    shuffled_player_types = random.sample(player_types, len(player_types))  # Shuffle player types
    for i, player_type in enumerate(shuffled_player_types):
        if i == 0:
            trust = 1  # PARAMETER OF INTEREST: trustability
        elif i == 1:
            trust = 0.5
        elif i == 2:
            trust = 0
        else:
            trust = 2

        h = np.random.randint(1, DICE_SIDES + 1, MAX_DICE_PER_PLAYER)
        if player_type == 'BOT':
            # All players have almost the same params
            p = Player(hand=h, risk_thres=r, likely_thres=l, exactly_thres=e, bluff_prob=bp, bluff_thres=bt,
                       trustability=trust, playerID=i + 1, num_dice_unseen=total_dice_left - h.size,
                       verbose=verbose)
            player_metadata.append(
                [p.playerID, p.risk_thres, p.likely_thres, p.exactly_thres, p.bluff_prob, p.bluff_thres,
                 p.trustability])
        elif player_type == 'HUMAN':
            p = HumanPlayer(hand=h, playerID=i + 1, trustability=trust, num_dice_unseen=total_dice_left - h.size)
            player_metadata.append([p.playerID, -1, -1, -1, -1, -1, -1])  # Dummy data for humans
        else:
            # raise ValueError
            p = None  # player_type must be HUMAN or BOT

        player_list.append(p)  # Keeps track of players left in game

    if verbose:
        print('Player order: {}'.format([(p.playerID, p.player_type) for p in player_list]))

    round = 0
    starting_player_index = 0  # Player 1 starts first round
    rotated_player_list = player_list
    while len(rotated_player_list) > 1:
        round = round + 1
        # Rotate player list, start with the starting_player_index
        rotated_player_list = rotated_player_list[
                              starting_player_index % len(rotated_player_list):] + rotated_player_list[
                                                                                   :starting_player_index]
        # Pick a direction of play
        # If its a bot,
        if rotated_player_list[0].player_type == 'Bot':
            direction_change = random.randint(0, 1)
        elif rotated_player_list[0].player_type == 'Human':
            input_denied = True
            direction_change = None
            while input_denied:
                direction_change_text = input('You start. Do you want a direction change? (Type \'N\' to go to Player {}, '
                                          'and \'Y\' to go to player {}) '.format(rotated_player_list[1].playerID,
                                                                                  rotated_player_list[len(rotated_player_list) - 1].playerID))
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
                        print(f'Human Player #{player.playerID}: {player.hand}, {player.num_dice_unseen} unseen dice')
        i = 0
        prev_a = None
        end_round = False
        number_counter = [0] * 6  # To keep track of what's been said
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
                    print(f'\nPlayer ID: {player.playerID}')
                if player.player_type == 'Human':
                    print('Hand: {}'.format(player.hand))
                    a = player.action(prev_a, plot=True)
                else:
                    a = player.action(prev_a, plot=False)

                # Set call metadata

                last_play = prev_a
                prev_a = a

                if verbose:
                    if a['bs']:  # If called Bullshit
                        print("Turn {}: DUDO".format(i))
                    elif a['exactly']:  # If called Exactly
                        print("Turn {}: EXACTLY".format(i))
                    else:
                        print("Turn {}: {} {}s".format(i, str(a['quantity']), str(a['dice'])))

                i += 1

                # user_enter = input('Enter to continue: ')
                if a['bs'] == True:
                    end_round = True
                    last_play = a
                    bullshit_caller_player_list_index = index
                    break
                if a['exactly'] == True:
                    end_round = True
                    last_play = a
                    exactly_caller_player_list_index = index
                    break

                number_counter[a['dice'] - 1] += 1
                for other_player in rotated_player_list:
                    if other_player == player: continue
                    other_player.calculate_conditional_distributions(cumulative_calls_list=number_counter)

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

            if total_count < last_play['quantity'] and player_bullshit_called_on != None:
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
                if len(rotated_player_list[index_current].hand) < 5:
                    rotated_player_list[index_current].hand = np.append(rotated_player_list[index_current].hand,
                                                                        0)  # Add a die
                    total_dice_left = total_dice_left + 1
                    if verbose:
                        print(
                            f'{total_count} {last_play["dice"]}s total == Player {rotated_player_list[index_current].playerID}\'s exactly bet of {last_play["quantity"]} {last_play["dice"]}s')
                        print(f'Player {rotated_player_list[index_current].playerID} wins a die')
                    starting_player_index = index_current
                else:
                    if verbose:
                        print(f'Player {rotated_player_list[index_current].playerID} has 5 die so didn\'t gain a die')
                    starting_player_index = index_current
            else:
                rotated_player_list[index_current].hand = rotated_player_list[index_current].hand[1:]  # Remove a die
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
    return game_metadata


if __name__ == '__main__':
    winners = []
    max_games = 1
    file_name = 'simulation_results/cacho_{}.csv'.format(datetime.now().strftime("%d%m%Y%H%M%S"))
    with open(file_name, 'w', newline='') as csvfile:
        print('Saving results to: {}'.format(file_name))
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(
            ['player_id', 'risk_thres', 'likely_thres', 'exactly_thres', 'bluff_prob', 'bluff_thres', 'trustability',
             'win_bool'])
        for i, games in enumerate(range(max_games)):
            game_metadata = runGame(verbose=1)
            gameWin = game_metadata["game_playerid_winner"]
            player_metadata = game_metadata["player_metadata"]
            if i % 10 == 0:
                print(f'({i + 1}/{max_games}) PLAYER {gameWin} WINS')
            for i, specific_player_metadata in enumerate(player_metadata):
                if specific_player_metadata[0] == game_metadata["game_playerid_winner"]:
                    player_row = specific_player_metadata
                    player_row.extend([1])  # Add win
                    csvwriter.writerow(player_row)
                else:
                    player_row = specific_player_metadata
                    player_row.extend([0])  # Add lose
                    csvwriter.writerow(player_row)
        winners.append(gameWin)
        Counter(winners)
