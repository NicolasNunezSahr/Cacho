import random
import numpy as np

from math import ceil
from collections import Counter

from collections import Counter
from typing import List
from scipy.stats import binom
import matplotlib.pyplot as plt

# Set global variables
NUM_PLAYERS = 3
MAX_DICE_PER_PLAYER = 5
MAX_TOTAL_DICE = NUM_PLAYERS * MAX_DICE_PER_PLAYER
DICE_SIDES = 6

#random.seed(123)
# Set round-specfic variables
player_dice_left = MAX_DICE_PER_PLAYER
total_dice_left = MAX_TOTAL_DICE
unseen_dice_left = total_dice_left - player_dice_left  # 20 total dice of which 5 dice are seen and the rest unseen
player_hand = sorted([random.randint(1, DICE_SIDES) for x in range(player_dice_left)])

print(f'Player\'s hand: {player_hand}')


class Player:
  def __init__(self, hand: List[int], risk_thres: float, likely_thres: float,
               playerID: int, verbose: int = 0):
    """
    Players have a hand, which is a list of dice values, which vary between
    1 and DICE_SIDES i.e. 6. If the conditional probability of a callable bet
    is larger than likely_thres, then the player always plays that. If there is
    no callable bet with probability >= likely_thresh, then the player may call
    bullshit on the received bet. A player only calls bullshit if the
    conditional probability of the received bet is less than risk_thresh.
    If the received bet has probability larger than risk_thresh, then the player
    raises the bet to the next callable bet that has highest conditional
    probability.
    """
    self.verbose = verbose
    self.hand = hand
    self.risk_thres = risk_thres
    self.likely_thres = likely_thres
    self.playerID = playerID
    self.conditional_dist = None
    self.num_dice_unseen = None
    self.num_dice_in_game = None

  def calculate_cond_dist(self, num_dice_unseen):
    c = get_conditional_distributions(self.hand, num_dice_unseen)
    self.conditional_dist = np.array(c)
    self.num_dice_unseen = num_dice_unseen
    self.num_dice_in_game = num_dice_unseen + len(self.hand)

  def _get_highest_probability_call(self, m):
    max_prob = np.max(m)
    # TODO: what if multiple options have the same probability?
    idx_flat = np.argmax(m)
    idx = np.unravel_index(idx_flat, m.shape)
    return max_prob, idx

  def first_action(self):
    cda_mat = self.conditional_dist[:,1:].copy()
    max_prob, idx = self._get_highest_probability_call(cda_mat)
    return {'dice': idx[0]+1, 'quantity': idx[1]+1, 'bs': False}

  def action(self, prev_action=None, plot=False):
    if prev_action == None:
      return self.first_action()

    ## create matrix with set of impossible calls zero-ed out
    d = prev_action['dice']
    q = prev_action['quantity']
    cda_mat = self.conditional_dist[:,1:].copy() # remove at least 0 col

    if d == 1:
      # here I removed , self.num_dice_in_game, this means if there are only a couple die
      # in the game it would not need to double + 1 which would be a problem theoretically

      # zero-out un-allowed non-ace calls
      cda_mat[1:,:min(self.num_dice_in_game, q*2 + 1) - 1] = -1.0

      # Zero-out un-allowed ace calls
      cda_mat[0, :min(self.num_dice_in_game, q)] = -1.0

    else:
      # If q = 5 and d = 2, i.e. if I call 5 2's, you can still call 5 3's.
      # So, the min quantity the next person can call is still q = 5.
      # This is true unless d = 6, which is the highest number, but that's OK.

      # Zero out unallowed non-ace calls
      # any non-ace call w/ Q <= q - 1
      cda_mat[1:,:max(min(self.num_dice_in_game, q - 1), 0),] = -1.0
      # any D <= d w/ Q = q
      cda_mat[1:d,min(self.num_dice_in_game, q - 1)] = -1.0

      # Zero out unallowed ace calls
      cda_mat[0,:ceil(q/2) - 1] = -1.0

    if self.verbose:
      print('Probabilities: {}'.format(cda_mat))
      if plot:
        plot_distributions(cond_distributions=self.conditional_dist,
                         player_id=self.playerID)


    # find highest probability call
    max_prob, idx = self._get_highest_probability_call(cda_mat)
    if max_prob <= 0:  # If highest probability play is 0.0, must call bullshit
      return {'bs': True, 'dice': d, 'quantity': q}
    else:
      best_action = {'dice': idx[0]+1, 'quantity': idx[1]+1, 'bs': False}

    # evaluate against likely threshold
    if max_prob > self.likely_thres:
      return best_action

    # check risk threshold
    prob_prev_action = self.conditional_dist[d - 1][q]
    if prob_prev_action < self.risk_thres:
      print(f'Checking risk: P({q} {d}s) = {prob_prev_action} < {self.risk_thres} -> BS')
      return {'bs': True, 'dice':d, 'quantity':q}
    print(f'Checking risk: P({q} {d}s) = {prob_prev_action} > {self.risk_thres} -> RAISE')

    # else, make highest probability call
    return best_action


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
      number_prob = 1/6
    else:
      number_prob = 1/3

    # These are the unconditional probabilities of the unseen dice
    unconditional_probabilities = [1 - binom.cdf(n=num_dice_unseen, p=number_prob, k = x) + \
          binom.pmf(n=num_dice_unseen, p=number_prob, k = x) \
          for x in range(1, num_dice_unseen + 1)]

    # print(f'unseen probs: {len(unconditional_probabilities)} -> {unconditional_probabilities}')

    # Initialize conditional probability to 1.0 if you have them in your hand,
    #   initialize to -1.0 for those not in your hand (to easily spot an error),
    #   and initialize to 0.0 those that remain, so that the domain is
    #   num_dice_unseen + len(player_hand) (i.e. 15 + 5 = 20)
    conditional_probabilities = [1.0] * (count_in_hand + 1) + \
    [-1.0] * len(unconditional_probabilities) + [0.0] * (len(player_hand) - count_in_hand)
    for i, prob in enumerate(unconditional_probabilities):
      if i > num_dice_unseen + len(player_hand):
        break
      conditional_probabilities[i + count_in_hand + 1] = unconditional_probabilities[i]

    conditional_distributions_list_of_lists.append(conditional_probabilities)

    # print(f'cond probs: {len(conditional_probabilities)} -> {conditional_probabilities}')

  return conditional_distributions_list_of_lists


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
    ax.bar(x = [x - 5/16 + (i/8) for x in range(len(cond_distributions[i]))],
           height = cond_distributions[i],
           width=(1/8),
           label = f'd = {i + 1}')
  ax.set_xticks(ticks=range(len(cond_distributions[0])))
  plt.legend()
  plt.show()


def runGame(verbose: int = 0):
  # instantiate players
  r = 0.45  # risk threshold
  l = 0.8  # likely threshold
  player_list = []
  total_dice_left = MAX_TOTAL_DICE

  for i in range(NUM_PLAYERS):
    h = np.random.randint(1, DICE_SIDES + 1, MAX_DICE_PER_PLAYER)
    p = Player(hand = h, risk_thres = r, likely_thres = l, playerID= i + 1,
               verbose = verbose)  # Every player has same likely and risk threshs
    p.calculate_cond_dist(total_dice_left - p.hand.size)
    player_list.append(p)

  round = 0
  while(len(player_list) > 1):
    round = round + 1
    if verbose:
      print("-----------------------------------------------------------------")
      print("ROUND " + str(round))
      for player in player_list:
        print(f'{player.hand}, {player.num_dice_unseen} unseen dice')
    # simulate one round
    i = 0
    prev_a = None
    bs = False
    bug = False

    while bs == False:
      # Use % to loop i.e.:
      # player = player_list[i % len(player_list)]
      for index, player in enumerate(player_list):
        previous_index = (index - 1) if index > 0 else len(player_list) - 1
        index_current = index

        a = player.action(prev_a)
        last_play = prev_a
        prev_a = a

        i += 1
        if verbose:
          print("Player ID: " + str(player.playerID))
          print('turn {}: {}'.format(i, a))
        # user_enter = input('Enter to continue: ')
        if a['bs'] == True:
          bs = True
          last_play = a
          bullshit_caller_player_list_index = index
          break

      bs = a['bs']


    # when bullshit is called count the player hands to determine the outcomes
    # there are 2 outcomes, player who called bullshit loses a die or player who had bullshit called on loses a die
    # figure out last hand
    # then count

    count = 0
    dice_counts = Counter([number for player in player_list for number in player.hand])
    total_count = int(dice_counts[1]) if last_play['dice'] == 1 else (dice_counts[last_play['dice']]) + int(dice_counts[1])
    player_bullshit_called_on = player_list[(bullshit_caller_player_list_index - 1) % len(player_list)]

    if total_count < last_play['quantity'] and player_bullshit_called_on != None:
      player_list[previous_index].hand = player_list[previous_index].hand[1:]
      if verbose:
        print(f'{total_count} {last_play["dice"]}s total < Player {previous_index + 1}\'s bet of {last_play["quantity"]} {last_play["dice"]}s')
        print("Player " + str(previous_index + 1) + " loses a die")
      total_dice_left = total_dice_left - 1
    elif total_count >= last_play['quantity'] and player_bullshit_called_on != None:
      player_list[index_current].hand = player_list[index_current].hand[1:]
      if verbose:
        print(f'{total_count} {last_play["dice"]}s total >= Player {previous_index + 1}\'s bet of {last_play["quantity"]} {last_play["dice"]}s')
        print("Player " + str(index_current + 1) + " loses a die")
      total_dice_left = total_dice_left - 1
    ##################################################
    player_list = [player for player in player_list if player.hand.size > 0]

    for player in player_list:
      player.hand = np.random.randint(1, DICE_SIDES + 1, player.hand.size)
      player.calculate_cond_dist(num_dice_unseen = total_dice_left - player.hand.size)


  return player_list[0].playerID



if __name__ == '__main__':
    winners = []
    max_games = 1
    for i, games in enumerate(range(max_games)):
      gameWin = runGame(verbose=0)
      print(f'({i + 1}/{max_games}) PLAYER {gameWin} WINS')

      winners.append(gameWin)

    Counter(winners)









