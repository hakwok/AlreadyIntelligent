# AlreadyIntelligent

## Abstract
Gambling isn't simply a game of chance—it’s about making the right decisions with limited information. This research aims to understand how a goal-based agent can be optimized for strategic decision-making in Poker betting games. The study will evaluate the effectiveness of three models: the Bayesian network, Hidden Markov models (HMM), and reinforcement learning, which enable the goal-based agent to improve game decision-making. These models will be tested on the Poker dataset from Kaggle, which contains game logs for each game, utilizing probabilistic reasoning and strategic adaptation. The agent operates in a partially observable dynamic environment, where performance is measured by win rates, expected long-term rewards, and adaptability. It senses relevant game-state information—such as player actions, tile distributions, and board changes—processing this data to select moves and optimize strategies. Insights from this research may inform the development of adaptive AI agents capable of strategic decision-making in other uncertain environments.

## Agent
The goal-based agent in this study is designed to optimize decision-making in Poker by focusing on win rates, long-term rewards, and adaptability as its performance measures. It operates in a partially observable environment, meaning that the agent only has access to limited information at any given time, such as its own cards, the community cards, and the actions of other players. This creates uncertainty, as the agent cannot directly observe its opponents’ hands or future game states, which complicates decision-making.

To make decisions, the agent’s actuators are responsible for actions like betting, folding, or raising, which are determined by analyzing the available game-state data. This includes factors such as the current betting round, the opponent’s past actions, and the distribution of cards. The agent’s sensors gather information from the environment, such as the actions taken by players, board changes, and game history, which are crucial for forming an understanding of the current state of play. This collected data helps the agent adjust its strategy, improving its chances of making the best move based on the incomplete information available.

## Dataset
https://huggingface.co/datasets/RZ412/PokerBench

## Collaborators
- Hayden Kwok
- Katelyn Villamin
- ChungYin Lee
- Ethan Shih
- Alan Diaz

