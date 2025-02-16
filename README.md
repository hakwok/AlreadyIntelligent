# AlreadyIntelligent


## Collaborators
- Hayden Kwok
- Katelyn Villamin
- ChungYin Lee
- Ethan Shih
- Alan Diaz

## Abstract
Gambling isn't simply a game of chance—it’s about making the right decisions with limited information. This research aims to understand how a goal-based agent can be optimized for strategic decision-making in Poker betting games. The study will evaluate the effectiveness of three models: the Bayesian network, Hidden Markov models (HMM), and reinforcement learning, which enable the goal-based agent to improve game decision-making. These models will be tested on the Poker dataset from Kaggle, which contains game logs for each game, utilizing probabilistic reasoning and strategic adaptation. The agent operates in a partially observable dynamic environment, where performance is measured by win rates, expected long-term rewards, and adaptability. It senses relevant game-state information—such as player actions, tile distributions, and board changes—processing this data to select moves and optimize strategies. Insights from this research may inform the development of adaptive AI agents capable of strategic decision-making in other uncertain environments.

## Agent Overview: PEAS
The goal-based agent in this study is designed to optimize decision-making in Poker by focusing to win as much as possible in a poker game while avoiding losing. It operates in a partially observable environment, meaning that the agent only has access to limited information at any given time, such as its own cards, the community cards, and the actions of other players. This creates uncertainty, as the agent cannot directly observe its opponents’ hands or future game states, which complicates decision-making.

To make decisions, the agent’s actuators are responsible for actions like betting, folding, or raising, which are determined by analyzing the available game-state data. This includes factors such as the current betting round, the opponent’s past actions, and the distribution of cards. The agent’s sensors gather information from the environment, such as the actions taken by players, board changes, and game history, which are crucial for forming an understanding of the current state of play. This collected data helps the agent adjust its strategy, improving its chances of making the best move based on the incomplete information available.


## Dataset & Source
Dataset: https://huggingface.co/datasets/RZ412/PokerBench

Lib Source: https://github.com/uoftcprg/pokerkit
## Agent Setup
We simplify the poker game by assuming opponent's can only "Check" and our player can only "Check or Fold," because for now we want to find the probability of winning given our observations (Our hands and the Face up cards). 

In poker, different hands have different "strength," for instance a Two and a Seven (2,7) is weaker than an Ace and a Two (1,2) because the latter has a higher card and can possibly form a straight but former cannot. However, as the game progresses, (dealing the faceup cards), the first pair could have a greater strength, for example, with a flop (7,7,7). That is, given different face-up cards, the probability of winning changes. 

Therefore we set up our Bayesian Network with each stage of the game impacting only the next stage of the game. The model finds the probability of winning given our hands, flop, turn, river and the hidden variable opponent hands. While we recognize in real life, the opponent's hand is set once dealt and will have an impact on what card being dealt next, since we could not observe the opponent's hands, we assume it has no impact on the dealing deck and it is "generated" in the show down, (the last stage of the game that every player shows their hands). 

A sample game of flow:
```
game = PokerGame(num_of_players = 6)
game.start_round(agent)

"""
Here is an idea how the a round of Poker using Pokerkit...
"""
def start_round(self, agent_function = None):    
    self.state = self.sample_game(
        inf, #starting stacks
        number_of_player,
    )
    print(f"Your Seat is {self.seat}")
    #---------------
    #preflop
    self.__deal_hands(hands)
    self.__betting()
    self.__player_action(agent_function)
    #postflop, flop, turn, and river
    board_cards = [flop, turn, river]
    while self.state.showdown_index == None: #not the showdown
      #Deal Board -> Betting-> Player-> (re-start)
      self.__deal_board(board_cards)
      self.__betting()
      self.__player_action(agent_function) #agent action
    #showdown
    self.__show_down()

    # Evaluate Outcome
      actual_win = self.payoffs() > 0  # True if player actually won
      predicted_win = not self.Fold  # True if player did NOT fold

      if actual_win and predicted_win:
          result = "TW"  # True Positive
      elif not actual_win and predicted_win:
          result = "FW"  # False Positive
      elif not actual_win and not predicted_win:
          result = "TL"  # True Negative
      else:  # actual_win and not predicted_win
          result = "FL"  # False Negative

      return result
```

## CPT
Say we are given a hands $$\( H = h \), \( P_{ML}(Winning | H = h) \)$$

$$
P_{ML}(Winning | H = h) = \sum_{\{f, t, r, opp\}} P_{ML}(Winning,f, t, r, opp| H = h)
$$

$$
= \frac{\text{count}(Winning, H=h, f, t, r, opp)}{\text{total count}}
$$

Our agent estimates the cpt by finding the maximum likelihood of the sample data. We use the given evidence to initialize the simulation and run the simulation to collect sample data. Our agent would not take the risk of losing if the probability of winning given the evidence is less than certain values. 

Our agent estimates the cpt by finding the maximum likelihood of the sample data. We use the given evidence to initialize the simulation and run the simulation to collect sample data. Our agent would not take the risk of losing if the probability of winning given the evidence is less than certain values. 

```
#Decides what action to take based on evidence
def our_agent(numb_player, hand, flop = None, turn = None, river = None):
  cpt = PW_E(numb_player, hand, flop, turn, river)
  if cpt > 0.15:
    return "CHECK"
  else:
    return "FOLD"
```

```
def PW_E(numb_player, hand, flop = None, turn = None, river = None):
  num_of_games = 100 #100 for faster iteration and medium acc
  count = 0
  for i in (range(num_of_games)):
    count += poke_simulator(numb_player, hand, flop, turn, river)
  return count / num_of_games
```

## Model Evaluation
We evaluated our model based on the confusion matrix:
- True Win: It was actually a win and was predicted a win
- True Lose: It was actually a lose and was predicted a lose
- False Win: It was actually a lose but was predicted a win
- False Lose: It was actually a win but was predicted as lose

Given the values (hand, flop, turn, river)  from our dataset, we randomly sampled 200 game states. Our model achieved an accuracy of 0.5, indicating that its predictive performance is no better than random guessing.

```
def sim_games(row):
  """
    Function to simulate the games given a certain row

    Args:
    row: A given row in the dataset

    Returns:
    result: The result of the game (True Win, True Loss, False Win, False Lose)
  """
  flop= row['flop'] if not pd.isna(row['flop']) else None
  turn= row['turn'] if not pd.isna(row['turn']) else None
  river= row['river'] if not pd.isna(row['river']) else None
  result= pk.start_round(our_agent, row['hands'],flop, turn, river)
  return result

def find_accuracy(df):
  """
    Function to calculate the accuracy of the model

    Args:
    df: Our given Poker dataset

    Returns:
    accuracy: Calculated accuracy by (TL + TW)/(TL+ TW + FL + FW)
  """
  TL_count= (df['win']== 'TL').count()
  TW_count= (df['win']== 'TW').count()
  FL_count= (df['win']== 'FL').count()
  FW_count= (df['win']== 'FW').count()

  accuracy= (TL_count + TW_count)/ (TL_count + TW_count + FL_count + FW_count)
  return accuracy

sample= poker_df.sample(200, random_state=1)
sample['win']= sample.apply(sim_games, axis=1)
accuracy= find_accuracy(sample)
```

## Conclusion
Our model is not very accurate in making the "right" decision, that is to keep playing if there is a good chance of winning and to fold if it is going to lose. The model is supposed to fail because its action depends on only one single conditional statement, where in different stages of the poker, each stage has a different optimal strategy.
Our model is working on a simplified version of poker so we need to consider other possible actions for the opponent and our model to perform if we want the model to work on a more accurate representation of poker.

## Improvements
- The probability of getting a strong hand at the beginning of the game is low so we need to find the threshold that maximizes our chances of choosing the correct action. 
- Allowing the opponent to perform more actions and taking into consideration whether the opponent raises/bets can help our model perform more complicated actions such as bluffing or calling a bluff.
(not sure about this one) We can improve the model by having a betting history for each player. This can help the model recognize any pattern that the opponents may display such as commonly bluffing with a weak hand. With this, the model’s action will better suit the opponent.
- It is possible to implement a risk factor for the model to use and decide whether their action should be to call or to fold. This risk factor can be implemented by tracking how many chips need to be betted and the potential winnings. If the model has a weak hand, it may still choose to not fold if the investment isn’t too high. Similarly, if the pot is small and the investment is too high the model may choose to fold despite the hand.
- Moving forward, we want to strive for a model that maximizes the accuracy of our actions.
