# Proximal Policy Optimization for Othello Mastery

Welcome to the development of "The Strategic Heuristic Agent with Zero-human Advancement"! This project focuses on mastering the game of Othello using parallel self-playing and the Proximal Policy Optimization algorithm.


## Introduction
Othello, a popular board game that is sometimes called Reversi, was created in 1883 and is credited to Lewis Waterman and John W. Mollett. It eventually became standardized in its board configuration after evolving over time. Once Goro Hasegawa introduced Othello to the Japanese public in 1971, the game swiftly gained popularity throughout the world and, by 1977, was a mainstay in competitive tournaments.

The game's strategic intricacy poses a tremendous challenge for artificial intelligence, owing partly to its massive state space. Estimates range from $10^{28}$ to $10^{32}$, exceeding the number of bacteria on Earth by more than a hundredfold and grains of sand by $10^{9}$. Creating one billion Othello positions each second would take over 3.1797× $10^{11}$ years, which is comparable to 23 times the universe's lifetime.

Inspired by [DeepMind's AlphaZero](https://arxiv.org/abs/1712.01815) , which achieved superhuman proficiency in chess, shogi, and Go via self-play and reinforcement learning, this project adopts similar methodologies to conquer Othello. SHA-ZHA leverages parallel game simulation and [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) to empower the agent to learn and evolve through self-play autonomously, devoid of any prior human guidance.
## Methodology
### Training algorithm
**Proximal Policy Optimization (PPO)**: An advanced reinforcement learning algorithm developed by OpenAI. PPO balances exploration and exploitation by clipping probability ratios. This helps in stabilizing training and improving the performance of the agent.

The PPO loss function consists of three components:

1. **Policy Loss**: This measures the difference between the predicted action probabilities and the action probabilities that maximize the expected return. The policy loss helps the agent to learn the optimal policy by adjusting the probabilities of taking certain actions.
<div align="center">
  
![Screenshot 2024-06-09 143657](https://github.com/mohammed-tech-innovator/proximalpolicy-optimization-for-othello-mastery/assets/40921388/173aa008-31b4-452c-b896-7c73c54dee7e)

</div>
where:
- $\pi_{\theta}$ represents the current policy.
- $\pi_{\theta_{\text{old}}}$ represents the old policy before the update.
- $a_t$ is the action taken at time $t$.
- $s_t$ is the state at time $t$.
- $\hat{A}_t$ is the advantage estimate at time $t$.
- $\epsilon$ is the clipping parameter.

2. **Value Loss**: This measures the difference between the predicted value function and the observed returns. The value loss helps the agent to accurately estimate the value of different states, which is crucial for making informed decisions.

<div align="center">
  
![Screenshot 2024-06-09 143715](https://github.com/mohammed-tech-innovator/proximalpolicy-optimization-for-othello-mastery/assets/40921388/8105a880-d50f-4477-8bb6-a168dbf09ae2)

</div>

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

### Agent Network Architecture

The agent consists of two deep convolutional neural networks based on [ConvNeXt](https://arxiv.org/abs/2201.03545). The first network is the policy network, which maps different boards into a probability distribution over actions. The second network is the value network, which provides an evaluation of the board.

**ConvNeXt** is a modern architecture for convolutional neural networks inspired by the design principles of vision transformers (ViTs). It combines the strengths of convolutional layers and transformer-like features, achieving state-of-the-art performance in various image recognition tasks. ConvNeXt introduces innovations such as:

- **Depthwise Convolutions**: These reduce the number of parameters and computations, making the network more efficient.
- **Layer Normalization**: This stabilizes and accelerates training by normalizing the inputs across the feature map dimensions.
- **Residual Connections**: These help in training deeper networks by mitigating the vanishing gradient problem.
<div align="center">
<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTTm384qquw_hbC0UIIhF8Jnr9gHvtNokBOCQ&s" alt="ConvNeXt" hight="250" style="display: block; margin-left: auto; margin-right: auto; border-radius: 15px;"/>
</div>
Another key component that significantly speeds up the learning process is the [**Convolutional Block Attention Module (CBAM)**](https://arxiv.org/abs/1807.06521). CBAM enhances the feature representation of the neural network by focusing on important information and suppressing irrelevant details. It consists of two sequential sub-modules:

- **Channel Attention Module**: This emphasizes informative channels and suppresses less useful ones by computing channel-wise attention.
- **Spatial Attention Module**: This enhances important spatial features and suppresses irrelevant ones by computing spatial attention maps.

By applying both channel and spatial attention, CBAM improves the network's ability to focus on crucial parts of the input, leading to better performance and faster convergence.
<div align="center">
<img src="https://miro.medium.com/v2/resize:fit:1400/0*cvZx6H1aDsSgqQ1z" alt="CBAM" width="600" style="display: block; margin-left: auto; margin-right: auto; border-radius: 15px;"/>
</div>
### State (Board) Representation

Game state is represented using 3 components :
- **Legal Moves**: All legal moves an agent can take at a given position. This is also used to restrict the agent from choosing invalid moves.
- **Pieces Under Attach**: All opponent's pieces that can be captured with one move.
- **Board**: All player's pieces are represented by 1, and all opponent's pieces are represented by -1.

  
All 3 components are concatenated and used for inference.

### Training Process

The agent was trained for 6,000 steps. During each step, 16 parallel processes generated 512 games each. Every 25 steps, the agent was tested against four opponents: a random policy, a reference agent trained using the same approach, a minimax agent with a depth of 7, and the current best agent. If the agent defeated the best agent, a checkpoint was saved.
- **Hardware**: Nvidia RTX A5000

### Hyperparameters
<div align="center">
  
| Hyperparameter        | Value for steps 0-3000   | Value for steps 3000-5200 | Value for steps 5200-6000  |
|-----------------------|----------------|------------------|------------------|
| Batch Size            | 256            | 256              | 256              |
| Value Coefficient     | 0.5            | 0.5              | 0.5              |
| Entropy Coefficient   | 0.09           | 0.045            | 0.09             |
| Clip Parameter        | 0.2            | 0.2              | 0.2              |
| Learning Rate         | 0.5e-4 (step 0) | 0.25e-4 (step 3000) | 0.125e-4 (step 6000) |
| Games / Step         | 4096 | 8192 | 4096 |

</div>


## Results
After 5725 steps, the agent achieved the following win rates:

- Against random policy: **100%**.
- Against agent with minimax depth of 7: **96.748%**.
- Against the reference agent: **68.85%**.

![5725](https://github.com/mohammed-tech-innovator/proximalpolicy-optimization-for-othello-mastery/assets/40921388/35485ac7-ce47-4681-a65d-a65028a95d0d)

- Following the training phase, SHA-ZA was thoroughly evaluated in standard matches against Minimax agents, with search depths ranging from 6 to 12. These trials were conducted using SHA-ZA serving as both an initiator and a non-initiating player. SHA-ZA consistently beat the Minimax algorithm at all tested depths.
<div align="center">
  
| Match  | Minimax Depth | Initiating Player | Winner | Video Link |
|--------|---------------|-------------------|--------|------------|
| Match 1 | 6             | SHA-ZA            | SHA-ZA | [video 1](https://youtu.be/EajVC9woUSM) |
| Match 2 | 6             | Minimax           | SHA-ZA | [video 2](https://youtu.be/o-xnlFO5vpk) |
| Match 3 | 9             | SHA-ZA            | SHA-ZA | [video 3](https://youtu.be/KGrxUoulhko) |
| Match 4 | 9             | Minimax           | SHA-ZA | [video 4](https://youtu.be/_460eicf3s8) |
| Match 5 | 12            | SHA-ZA            | SHA-ZA | [video 5](https://youtu.be/9qXpWrR48Hg) |
| Match 6 | 12            | Minimax           | SHA-ZA | [video 6](https://youtu.be/g2Wx_m37Oac) |

</div>

- Other checkpoints: 1000 steps, 1950 steps, 2500 steps, 3000 steps, 4000 steps, 4150 steps, and 5025 steps.
  
## Conclusion

- SHA-ZHA, trained with Proximal Policy Optimization and self-play, achieved **superhuman Othello performance** (by being tested to defeat minimax engines up to a depth of 12) with no initial knowledge whatsoever except the basic rules of the game.
- **One interesting observation** is that from time 0:22 to 0:32 in the video, SHA-ZHA was severely outnumbered in terms of material, but the output of the value network was between 0.2 and 0.3, which is optimistic. This indicates that the model is concerned with the final outcome of the game and can strategically make smart sacrifices.


## Get in touch

- Email: [mohammed.yah.yousif@gmail.com](mailto:mohammed.yah.yousif@gmail.com)
- Profile: [tech-innovator.me](https://tech-innovator.me/)
- LinkedIn: [Mohammed Yousif](https://www.linkedin.com/in/mohammed-yousif-6b272a241/)
- X: [Mohmammed Yousif](https://x.com/Moh_yah_you)
