# proximalpolicy-optimization-for-othello-mastery
Welcome to the development of "The Strategic Heuristic Algorithm with Zero-Human Advancement"! This project focuses on mastering the game of Othello using parallel self-playing and the Proximal Policy Optimization algorithm.

## Introduction
Othello, a popular board game that is sometimes called Reversi, was created in 1883 and is credited to Lewis Waterman and John W. Mollett. It eventually became standardized in its board configuration after evolving over time. Once Goro Hasegawa introduced Othello to the Japanese public in 1971, the game swiftly gained popularity throughout the world and, by 1977, was a mainstay in competitive tournaments.

The game's strategic intricacy poses a tremendous challenge for artificial intelligence, owing partly to its massive state space. Estimates range from 10^28 to 10^32, exceeding the number of bacteria on Earth by more than a hundredfold and grains of sand by 10^13. Creating one billion Othello positions each second would take over 10^18 years, which is comparable to 70,000 times the universe's lifetime.

Inspired by [DeepMind's AlphaZero](https://arxiv.org/abs/1712.01815) , which achieved superhuman proficiency in chess, shogi, and Go via self-play and reinforcement learning, this project adopts similar methodologies to conquer Othello. SHA-ZHA leverages parallel game simulation and [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) to empower the agent to learn and evolve through self-play autonomously, devoid of any prior human guidance.
## Methodology
The PPO loss function consists of three components:

1. **Policy Loss**: This measures the difference between the predicted action probabilities and the action probabilities that maximize the expected return. The policy loss helps the agent to learn the optimal policy by adjusting the probabilities of taking certain actions.

![Screenshot 2024-06-09 143657](https://github.com/mohammed-tech-innovator/proximalpolicy-optimization-for-othello-mastery/assets/40921388/173aa008-31b4-452c-b896-7c73c54dee7e)

where:
- $\pi_{\theta}$ represents the current policy.
- $\pi_{\theta_{\text{old}}}$ represents the old policy before the update.
- $a_t$ is the action taken at time $t$.
- $s_t$ is the state at time $t$.
- $\hat{A}_t$ is the advantage estimate at time $t$.
- $\epsilon$ is the clipping parameter.

2. **Value Loss**: This measures the difference between the predicted value function and the observed returns. The value loss helps the agent to accurately estimate the value of different states, which is crucial for making informed decisions.


![Screenshot 2024-06-09 143715](https://github.com/mohammed-tech-innovator/proximalpolicy-optimization-for-othello-mastery/assets/40921388/8105a880-d50f-4477-8bb6-a168dbf09ae2)

where:
- $V_{\theta}(s_t)$ is the predicted value function for state $s_t$.
- $R_t$ is the observed return at time $t$.

3. **Entropy Loss**: This measures the uncertainty or randomness in the agent's policy. The entropy loss encourages exploration by preventing the policy from becoming too deterministic, thus promoting a more robust learning process.


![Screenshot 2024-06-09 143959](https://github.com/mohammed-tech-innovator/proximalpolicy-optimization-for-othello-mastery/assets/40921388/2a13670c-b2d3-4dd1-adee-2280e3740f58)

where:
- $\pi_{\theta}(a | s_t)$ is the probability of taking action $a$ in state $s_t$ under the current policy.


4. **Total Loss**: This is the combined loss function used to update the model parameters. It incorporates the policy loss, value loss, and entropy loss, balanced by their respective coefficients.

$$
\text{Total Loss} = \text{Policy Loss} + c_v \cdot \text{Value Loss} - c_e \cdot \text{Entropy Loss}
$$

where:
- $c_v$ is the value loss coefficient.
- $c_e$ is the entropy loss coefficient.

## Results
![5725](https://github.com/mohammed-tech-innovator/proximalpolicy-optimization-for-othello-mastery/assets/40921388/35485ac7-ce47-4681-a65d-a65028a95d0d)

## Conclusion


