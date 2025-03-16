# AlreadyIntelligent


## Collaborators
- Hayden Kwok
- Katelyn Villamin
- ChungYin Lee
- Ethan Shih
- Alan Diaz

## Abstract
Gambling isn't simply a game of chance—it’s about making the right decisions with limited information. This research aims to understand how a goal-based agent can be optimized for strategic decision-making in Poker betting games. The study will evaluate the effectiveness of two models: the Bayesian network and Monte Carlo, which enable the goal-based agent to improve game decision-making. These models will be tested on the Poker dataset from Kaggle, which contains game logs for each game, utilizing probabilistic reasoning and strategic adaptation. The agent operates in a partially observable dynamic environment, where performance is measured by win rates, expected long-term rewards, and adaptability. It senses relevant game-state information—such as player actions, tile distributions, and board changes—processing this data to select moves and optimize strategies. Insights from this research may inform the development of adaptive AI agents capable of strategic decision-making in other uncertain environments.

## Bayesian Agent Overview: PEAS
The goal-based agent in this study is designed to optimize decision-making in Poker by focusing to win as much as possible in a poker game while avoiding losing. It operates in a partially observable environment, meaning that the agent only has access to limited information at any given time, such as its own cards, the community cards, and the actions of other players. This creates uncertainty, as the agent cannot directly observe its opponents’ hands or future game states, which complicates decision-making.

To make decisions, the agent’s actuators are responsible for actions like betting, folding, or raising, which are determined by analyzing the available game-state data. This includes factors such as the current betting round, the opponent’s past actions, and the distribution of cards. The agent’s sensors gather information from the environment, such as the actions taken by players, board changes, and game history, which are crucial for forming an understanding of the current state of play. This collected data helps the agent adjust its strategy, improving its chances of making the best move based on the incomplete information available.

![image](https://github.com/user-attachments/assets/242375f3-c643-4b94-9883-b888339157e2)
Figure 1: A graph visualization of our agent

## Dataset & Source
Dataset: https://huggingface.co/datasets/RZ412/PokerBench

Lib Source: https://github.com/uoftcprg/pokerkit

## Notebooks with Full Code
- [Data Extraction](data_extraction.ipynb)
- [Modeling and Evalutation](modeling_and_evaluating.ipynb)

## Data Preprocessing
The dataset "PokerBench" includes different scenarios in a poker game (pre/post flop, Turn and River), and other information like the betting history. We extract the observation, face up cards and store it in this order: Hnads (2cards), Flop(3cards), Turn(1cards), Rriver(1cards). 

Each card is stored with two cahracters, the 1st character representing the number and 2nd  character representing the suit, with all caps and separated with comma, and card that is not dealt yet is filled with "Null". For instance King of Heart is "KH", Ten of Spades is "TS" and two of Club is "2C". This allows easy access to the cards we want to initialize in the testing and training stage. 

## Bayesian Agent Setup
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

Wherein:
- H= our given hand
- f= flop, t= turn
- r= river
- opp= the opponent's move

Our agent estimates the cpt by finding the maximum likelihood of the sample data. We use the given evidence to initialize the simulation and run the simulation to collect sample data. Our agent would not take the risk of losing if the probability of winning given the evidence is less than certain values. 

Our agent estimates the cpt by finding the maximum likelihood of the sample data. We use the given evidence to initialize the simulation and run the simulation to collect sample data. Our agent would not take the risk of losing if the probability of winning given the evidence is less than certain values. 

```
#Decides what action to take based on evidence
def our_agent(numb_player, hand, flop = None, turn = None, river = None):
  cpt = PW_E(numb_player, hand, flop, turn, river)
  if cpt > 0.20:
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

In the pre-flop stage, almost all the hands has a probability of winning less than 0.4 in a 6 player poker game, and therefore in order to not fold in pre-flop stage, we pick a arbitrary thresh hold, say the median (0, 0.4) = 0.2 for the probability lower bound for checking.
```
if (CPT > .20) :
    return "CHECK"
```

## Model Evaluation Method

We evaluated our model action based on the what the outcome of the game is (we only consider the last action the agent make):
- True  (Good)  Check: The check lead to a win. (TC)
- False (Bad)   Check: The check lead to a lose. (FC)
- True  (Good)  Fold : The fold avoid a lose. (TF)
- False (Bad)   Fold : The fold avoid a win. (FF)

For calcilating accuracy of the agent making a "True Action", we can count the number of true action divided by the total number of action, i.e

Naive_accuracy = 'TC + TF' / 'TC + TF + FC + FF'

Since our goal is to maximize the number of winning games while minimizing the number of losing games over all, we should consider how many the winning game the agent could reach among all winning games, and for all the losing games, how many the agent could avoid, i.e the likelihood:

Winning_accuracy (P(Check|Win)) = All the winning moves (checks) / number of winning games
= 'TC' / 'TC + FF'

Lose_accuracy (P(Fold|Lose)) = All avoid losing moves(Fold) / number of losing games
= 'TF' / 'TF + FC'

We can also calculate the posterior probability, the chance of winning given our agent checked, and losing given our agent fold:

P(Win | Check) = 'TC' / 'TC + FC'

P(Lose | Fold) = 'TF' / 'TF + FF'


### Result
We randomly tested 1000 game states. Our Agent achieved 0.4946 likelihood of winning and 0.9007 likelihood of losing. Comparing to the random agent with a 0.0485 and 0.9365 likelihood of winning and losing, our agent has a higher likelihood to win in a game under the same context of random agent and is slightly more likely to lose without folding than the random agent. 

```
from tqdm import tqdm

def run_random_simulation(num_games=1000):
    true_pos, false_pos, true_neg, false_neg = 0, 0, 0, 0
    rtrue_pos, rfalse_pos, rtrue_neg, rfalse_neg = 0, 0, 0, 0
    pk = PokerGame(6, 0)  # Initialize the poker game


    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(run_game, [(pk, random_agent) for _ in range(num_games)]), total=num_games, desc="Simulating Random Agent Games", leave=False))

    for game in results:
        if game == "TC": rtrue_pos += 1
        if game == "FC": rfalse_pos += 1
        if game == "TF": rtrue_neg += 1
        if game == "FF": rfalse_neg += 1
    print("")
    print(f"\nWinning Accuracy for Random Agent (out of {num_games} games):")
    print(f"Winning Accuracy: {rtrue_pos / (rtrue_pos + rfalse_neg):.4f}")
    print(f"Losing Accuracy: {rtrue_neg / (rtrue_neg + rfalse_pos):.4f}")

    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(run_game, [(pk, our_agent) for _ in range(num_games)]), total=num_games, desc="Simulating Our Agent Games", leave=False))
    for game in results:
        if game == "TC": true_pos += 1
        if game == "FC": false_pos += 1
        if game == "TF": true_neg += 1
        if game == "FF": false_neg += 1

    print("")
    print(f"\nWinning Accuracy for our Agent (out of {num_games} games):")
    print(f"Winning Accuracy: {true_pos / (true_pos + false_neg):.4f}")
    print(f"Losing Accuracy: {true_neg / (true_neg + false_pos):.4f}")


run_random_simulation() # Call the function to start the simulation
```

## Conclusion
Our model’s accuracy is not ideal. By accuracy, we define our model’s ability to choose the optimal move based on the current state—specifically, making the “right” decision to continue playing when there is a good chance of winning (leading to a win) and to fold when defeat is likely (leading to a loss). However, our model struggles to make these decisions consistently. The model is expected to fail because its action depends on a single conditional statement, whereas each stage of poker requires a different optimal strategy. In comparison, we run a series of random playthroughs, where actions are chosen at random at each state, to serve as a benchmark. Although our model’s accuracy is only at 48.78% win and 91.02% loss, it still outperforms the random playthroughs, which have an accuracy of just 6.48% win and 93.78% loss. There is a major discrepancy between winning accuracy compared to losing accuracy as "winning" would indicate that the model made as many right choices needed to either have all opponents folding or wining during post-flop, which is expected given the exponentially lower chances as the game progresses for the player to win. Ideally, we want our model to consistently dominate the game, gaining confidence through improved accuracy so that it can reliably make the right decision the majority of the time (over 50% accuracy), especially leading towards a win. Since our model is working with a simplified version of poker, we need to consider other possible actions for both the opponent and our model to improve performance in a more accurate representation of the game.

## Improvements
- The probability of getting a strong hand at the beginning of the game is low so we need to find the threshold that maximizes our chances of choosing the correct action. 
- Allowing the opponent to perform more actions and taking into consideration whether the opponent raises/bets can help our model perform more complicated actions such as bluffing or calling a bluff.
(not sure about this one) We can improve the model by having a betting history for each player. This can help the model recognize any pattern that the opponents may display such as commonly bluffing with a weak hand. With this, the model’s action will better suit the opponent.
- It is possible to implement a risk factor for the model to use and decide whether their action should be to call or to fold. This risk factor can be implemented by tracking how many chips need to be betted and the potential winnings. If the model has a weak hand, it may still choose to not fold if the investment isn’t too high. Similarly, if the pot is small and the investment is too high the model may choose to fold despite the hand. The reason as to why we are not considering raises or all ins is due to the incalculable fact on player personality, as well as the cases in which they are bluffing their hands. 
- Moving forward, we want to strive for a model that maximizes the accuracy of our actions.

## Monte Carlo Agent : PEAS
This model will perform the same task as the previous one, building upon it to achieve improved performance. Since we are working with the same preprocessed data, the PEAS for this model remain consistent with those of the Bayesian Agent.

The goal-based agent in this study is designed to optimize decision-making in Poker by focusing on winning as much as possible in a poker game while avoiding losing. It operates in a partially observable environment, meaning that the agent only has access to limited information at any given time, such as its own cards, the community cards, and the actions of other players. This creates uncertainty, as the agent cannot directly observe its opponents’ hands or future game states, which complicates decision-making.

To make decisions, the agent’s actuators are responsible for actions like betting, folding, or raising, which are determined by analyzing the available game-state data. This includes factors such as the current betting round, the opponent’s past actions, and the distribution of cards. The agent’s sensors gather information from the environment, such as the actions taken by players, board changes, and game history, which are crucial for forming an understanding of the current state of play. This collected data helps the agent adjust its strategy, improving its chances of making the best move based on the incomplete information available.

## Data Preprocessing
The dataset "PokerBench" includes different scenarios in a poker game (pre/post flop, Turn and River), and other information like the betting history. We extract the observation, face up cards and store it in this order: Hands (2cards), Flop(3cards), Turn(1cards), River(1cards). 

Each card is stored with two characters, the 1st character representing the number and 2nd  character representing the suit, with all caps and separated with comma, and card that is not dealt yet is filled with "Null". For instance King of Heart is "KH", Ten of Spades is "TS" and two of Club is "2C". This allows easy access to the cards we want to initialize in the testing and training stage. 

## Monte Carlo Agent Setup


## CPT
Predicting the exact probabilities of certain in events in poker is generally impossible. For example, predicting the likelihood that the opponent holds a certain card combination will not be possible due to incomplete information. As such, due to the lack of perfect knowledge on opponent's hidden cards or future cards in the deck, calculating the exact conditional probability of a hidden event given observed cards is impractical.

The Markov Chain Monte Carlo (MCMC) method addresses this problem. This sampling-based approach helps us estimate probabilities through approximation from simulating many scenarios. 

In our set up, the Markov blanket is the set of all the currently known information on the poker board (our hand, the community cards, and revealed opponent cards) for any hidden card 𝑥. Given the Markov Blanket 𝐵𝑥, the conditional probability is P(𝑥∣𝐵𝑥), which is the probability of drawing the card 𝑥 from the remaining unseen cards. Under the assumption that every card that is not already revealed could equally likely be in the deck or in the opponent's hand, this suggests that each hidden card is equally probable. As such, sampling P(𝑥∣𝐵𝑥) simply translates to randomly drawing a card from the unseen deck. This provides a straightforward and more computationally efficient method for the MCMC sampling process. 

Through this, the MCMC method allows us to estimate the probabilities reflecting real-life poker scenarios, despite the inherent uncertainty.


## Model Evaluation Method
To reiterate:
True (Good) Check: The check lead to a win. (TC)
False (Bad) Check: The check lead to a lose. (FC)
True (Good) Fold : The fold avoid a lose. (TF)
False (Bad) Fold : The fold avoid a win. (FF)

The results from the MCMC Agent provided:
8 TC, 13 FP, 796 TF, 183 FF
with a winning accuracy of 4.19%, a losing accuracy of 98.39%, P(Win|Check) of 38.10% and P(Lose|Fold) of 81.31%.

While the results from the MC Agent provided:
164 TP, 5 FC, 831 TF, 0 FF
with a winning accuracy of 100%, Losing Accuracy of 99.40%, P(Win|Check) of 97.04%, and P(Los|Fold): 100%.

Ultimately the MC Agent provided better results as it had never folded to avoid a win, and has outperformed the MCMC agent by an astonishing 95.81%. Additionally, the MCMC agent misclassifies quite a large amount of false folds, leading to a high rate of avoiding potentials wins. Lastly the MC Agent shows perfect decision-making by having a 0 in false negatives while maintaining high true positive rates.

## Conclusion & Results
