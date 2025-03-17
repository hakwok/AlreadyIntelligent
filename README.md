# AlreadyIntelligent


## Collaborators
- Hayden Kwok
- Katelyn Villamin
- ChungYin Lee
- Ethan Shih
- Alan Diaz

## Abstract
Gambling isn't simply a game of chance—it’s about making the right decisions with limited information. This research aims to understand how a goal-based agent can be optimized for strategic decision-making in Poker betting games. The study will evaluate the effectiveness of two models: the Bayesian network and Monte Carlo, which enable the goal-based agent to improve game decision-making. These models will be tested on the Poker dataset from Kaggle, which contains game logs for each game, utilizing probabilistic reasoning and strategic adaptation. The agent operates in a partially observable dynamic environment, where performance is measured by win rates, expected long-term rewards, and adaptability. It senses relevant game-state information—such as player actions, tile distributions, and board changes—processing this data to select moves and optimize strategies. In this Milestone, we explore the Monte Carlo model. Insights from this research may inform the development of adaptive AI agents capable of strategic decision-making in other uncertain environments. 

## Dataset & Source
Dataset: https://huggingface.co/datasets/RZ412/PokerBench
Lib Source: https://github.com/uoftcprg/pokerkit

## Notebooks with Full Code
- [Data Extraction](data_extraction.ipynb)
- [Model_MCMC_and_MC](Model_MCMC_and_MC.ipynb)

## Monte Carlo Agent : PEAS
This model will perform the same task as the previous one, building upon it to achieve improved performance. Since we are working with the same preprocessed data, the PEAS for this model remain consistent with those of the Bayesian Agent.

The goal-based agent in this study is designed to optimize decision-making in Poker by focusing on winning as much as possible in a poker game while avoiding losing. It operates in a partially observable environment, meaning that the agent only has access to limited information at any given time, such as its own cards, the community cards, and the actions of other players. This creates uncertainty, as the agent cannot directly observe its opponents’ hands or future game states, which complicates decision-making.

To make decisions, the agent’s actuators are responsible for actions like betting, folding, or raising, which are determined by analyzing the available game-state data. This includes factors such as the current betting round, the opponent’s past actions, and the distribution of cards. The agent’s sensors gather information from the environment, such as the actions taken by players, board changes, and game history, which are crucial for forming an understanding of the current state of play. This collected data helps the agent adjust its strategy, improving its chances of making the best move based on the incomplete information available.

![image](https://github.com/user-attachments/assets/242375f3-c643-4b94-9883-b888339157e2)
Figure 1: A graph visualization of our agent

## Data Preprocessing
The dataset "PokerBench" includes different scenarios in a poker game (pre/post flop, Turn and River), and other information like the betting history. We extract the observation, face up cards and store it in this order: Hands (2cards), Flop(3cards), Turn(1cards), River(1cards). 

Each card is stored with two characters, the 1st character representing the number and 2nd  character representing the suit, with all caps and separated with comma, and card that is not dealt yet is filled with "Null". For instance King of Heart is "KH", Ten of Spades is "TS" and two of Club is "2C". This allows easy access to the cards we want to initialize in the testing and training stage. 

## Monte Carlo Agent Setup
Because of finding the exact inference is impossible with imperfect information (i.e. the face down card and unpredictable nature of human mind), we used simulation to estimate the probability of winning. We have made two assumption that any cards which is not in the evidence (on the board) is in the dealable cards deck, and the opponents' hands is dealt after the river which means it has no particular effect on what could possibly be dealt onto the board. 

In the first agent, we simulated the game from the moment the agent queries for the probability onwards.
```
def MC_agent1(number_player : int, seat : int, state : State):
  if PW_E_Monte_Carlo(number_player, seat, state) > 0.15:
    return "CHECK"
  return "FOLD"

def PW_E_Monte_Carlo(numb_player : int, seat : int, state : State):
  simulator = PokerSimulator(number_of_player=numb_player, seat = seat)

  num_of_games = 200 #We choose 200
  count = 0
  for i in (range(num_of_games)):
    count += simulator.simulate_statecopy(state)
  return count / num_of_games
```

In the second agent, we used MCMC method to do simulation, that is we initialize all the non-evidence cards to some random valid values and in each iteration we resample one of the non-evidence cards and record the result. Both agent uses the count of win divided by the number of simulation to estimate the winning probability given the evidence. 
```
def MCMC_agent(numb_player, seat, state):
  cpt = MCMC(numb_player, seat, state)
  return Action(cpt, state)

#Agent
def Action(win_probability, state : State):
  game_stage = len(state.board_cards)
  if game_stage == 0: #After dealin hands
      if win_probability > 0.10:
        return "CHECK"
      else:
        return "FOLD"
  if game_stage == 3: #Flop
      if win_probability > 0.20:
        return "CHECK"
      else:
        return "FOLD"
  if game_stage == 4: #Turn
      if win_probability > 0.20:
        return "CHECK"
      else:
        return "FOLD"
  if game_stage == 5: #River
      if win_probability > 0.30:
        return "CHECK"
      else:
        return "FOLD"

  return "WHAT"
def MCMC(number_of_player: int,
        seat: int,
        state: State,
        number_of_iteration: int = 1000):
    #======Internal Logic=====
    def initialize_values(players, board, dealable_cards, non_evidence_player_indices, non_evidence_board_indices):
        for i in non_evidence_player_indices:
            players[i][0]= random.choice(dealable_cards)
            dealable_cards.remove(players[i][0])
            players[i][1]= random.choice(dealable_cards)
            dealable_cards.remove(players[i][1])
        # Initialize the board
        for j in non_evidence_board_indices:
            board.append(random.choice(dealable_cards))
            dealable_cards.remove(board[j])

    def get_non_evidence_indices(number_of_player, players, board):
        non_evidence_player_indices = []
        for i in range(number_of_player):
            if players[i][0].unknown_status:
                non_evidence_player_indices.append(i)
        non_evidence_board_indices = [i for i in range(len(board), 5)]
        return non_evidence_player_indices,non_evidence_board_indices

    _state = deepcopy(state)
    players, board = deepcopy(_state.hole_cards), deepcopy(_state.board_cards)
    for i in range(len(board)):
        board[i] = board[i][0]

    dealable_cards = [card for card in _state.get_dealable_cards()]

    non_evidence_player_indices, non_evidence_board_indices = get_non_evidence_indices(number_of_player, players, board)
    initialize_values(players, board, dealable_cards, non_evidence_player_indices, non_evidence_board_indices)
    count_win = 0

    for _ in range(number_of_iteration):
        resample(players, board, dealable_cards, non_evidence_player_indices, non_evidence_board_indices)
        #Evaluation
        my_hand = StandardHighHand.from_game(players[seat], board)
        opp_hand = [StandardHighHand.from_game(players[i], board) for i in range(number_of_player) if i != seat]
        win = all(my_hand > opp_hand[i] for i in range(number_of_player-1))
        count_win += 1 if win else 0

    return count_win / number_of_iteration

```

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

Below is a sample of the code used to evaluate the model:
```

num_games = 1000 # Number of games to simulate per agent


def run_games(pk, agent, num_games, description):
    results = []
    for i in trange(num_games, desc=description):
        results.append(pk.start_round(agent))
    return results


pk = PokerGame(show_log=False, seed=6969)

with ProcessPoolExecutor() as executor:
    futures = {
        "mcmc": executor.submit(run_games, pk, MCMC_agent, num_games, "Simulating MCMC Agent Games"),
        "mc": executor.submit(run_games, pk, MC_agent1, num_games, "Simulating MC Agent Games")
    }

    # Get the results and process them
    for agent_name, future in futures.items():
        results = future.result()  # Get the list of game results

        true_pos, false_pos, true_neg, false_neg = 0, 0, 0, 0
        for game in results:
            if game == "TC":
                true_pos += 1
            elif game == "FC":
                false_pos += 1
            elif game == "TF":
                true_neg += 1
            elif game == "FF":
                false_neg += 1
```
These were the results from the evaluation:
| **Metric**         | **MCMC Agent**      | **MC Agent**     |
|--------------------|---------------------|------------------|
| Outcome (Win)      | TC: 8               | TP: 164          |
| Outcome (Loss)     | FP: 13              | FC: 5            |
| Correct Loss       | TF: 796             | TF: 831          |
| False Negative     | FF: 183             | FF: 0            |
| Winning Accuracy   | 4.19%               | 100%             |
| Losing Accuracy    | 98.39%              | 99.40%           |
| P(Win\|Check)       | 38.10%              | 97.04%           |
| P(Lose\|Fold)       | 81.31%              | 100%             |


Ultimately the MC Agent provided better results as it had never folded to avoid a win, and has outperformed the MCMC agent by an astonishing 95.81%. Additionally, the MCMC agent misclassifies quite a large amount of false folds, leading to a high rate of avoiding potentials wins. Lastly the MC Agent shows perfect decision-making by having a 0 in false negatives while maintaining high true positive rates.

## Conclusion & Results
