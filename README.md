# proximalpolicy-optimization-for-othello-mastery
Welcome to the development of "The Strategic Heuristic Algorithm with Zero-Human Advancement"! This project focuses on mastering the game of Othello using parallel self-playing and the Proximal Policy Optimization algorithm.

## Introduction
Othello, a popular board game that is sometimes called Reversi, was created in 1883 and is credited to Lewis Waterman and John W. Mollett. It eventually became standardized in its board configuration after evolving over time. Once Goro Hasegawa introduced Othello to the Japanese public in 1971, the game swiftly gained popularity throughout the world and, by 1977, was a mainstay in competitive tournaments.

The game's strategic intricacy poses a tremendous challenge for artificial intelligence, owing partly to its massive state space. Estimates range from 10^28 to 10^32, exceeding the number of bacteria on Earth by more than a hundredfold and grains of sand by 10^13. Creating one billion Othello positions each second would take over 10^18 years, which is comparable to 70,000 times the universe's lifetime.

Inspired by [DeepMind's AlphaZero](https://arxiv.org/abs/1712.01815) , which achieved superhuman proficiency in chess, shogi, and Go via self-play and reinforcement learning, this project adopts similar methodologies to conquer Othello. SHA-ZHA leverages parallel game simulation and [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) to empower the agent to learn and evolve through self-play autonomously, devoid of any prior human guidance.
## Methodology
PPO utilizes a loss function comprising three primary components:

1. **Policy Loss**: Measures the disparity between predicted and optimal action probabilities.

2. **Value Loss**: Evaluates the variance between predicted and observed returns.

3. **Entropy Loss**: Quantifies the level of uncertainty in the agent's policy.

### PPO Loss Function Components

#### Policy Loss

The policy loss function is defined as:

$$
\text{Policy Loss} = -\mathbb{E}_{t} \left[ \min \left( \frac{\pi_{\theta}(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)} \hat{A}_t, \, \text{clip} \left( \frac{\pi_{\theta}(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}, 1 - \epsilon, 1 + \epsilon \right) \hat{A}_t \right) \right]
$$

#### Value Loss

The value loss function is given by:

$$
\text{Value Loss} = \mathbb{E}_{t} \left[ \left( V_{\theta}(s_t) - R_t \right)^2 \right]
$$

#### Entropy Loss

The entropy loss function is expressed as:

$$
\text{Entropy Loss} = -\mathbb{E}_{t} \left[ \sum_{a} \pi_{\theta}(a | s_t) \log \pi_{\theta}(a | s_t) \right]
$$

### Total Loss

The total loss function, incorporating the policy, value, and entropy losses, is defined as:

$$
\text{Total Loss} = \text{Policy Loss} + c_v \cdot \text{Value Loss} - c_e \cdot \text{Entropy Loss}
$$

where:
- $c_v$ is the value loss coefficient.
- $c_e$ is the entropy loss coefficient.

These components collectively drive the optimization process, facilitating effective policy learning and value estimation for Othello.

## Results
![5725](https://github.com/mohammed-tech-innovator/proximalpolicy-optimization-for-othello-mastery/assets/40921388/35485ac7-ce47-4681-a65d-a65028a95d0d)

## Conclusion


