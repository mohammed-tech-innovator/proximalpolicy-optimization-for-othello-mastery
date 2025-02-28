# SHA-ZA: Advanced Reinforcement Learning for Othello Mastery

![SHA-ZA Othello Agent](https://github.com/user-attachments/assets/0e2db856-8447-45bc-afa7-c51ebd9c725b)

## Overview

This repository contains the official implementation for the paper ["SHA-ZA: Strategic Heuristic Agent with Zero-human Advancement for Othello Mastery Using Proximal Policy Optimization"](https://www.ijml.org/show-141-1380-1.html).

SHA-ZA masters the game of Othello through **parallel self-play** and **Proximal Policy Optimization (PPO)**, achieving superhuman gameplay without any human knowledge beyond the basic rules.

[![Paper](https://img.shields.io/badge/Paper-IJML-blue)](https://www.ijml.org/show-141-1380-1.html)
[![License](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python](https://img.shields.io/badge/Python-3.8+-yellow)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](#)

## 🎮 Introduction

Othello (sometimes called Reversi) is a classic board game created in 1883 by Lewis Waterman and John W. Mollett. The game gained worldwide popularity after Goro Hasegawa introduced it to Japan in 1971, leading to international tournaments by 1977.

The strategic complexity of Othello presents a significant challenge for AI systems due to its enormous state space:
- Estimated between 10²⁸ and 10³² possible board states
- Exceeds the number of bacteria on Earth by 100+ times
- Processing one billion positions per second would take 3.1797×10¹¹ years (23× the universe's age)

Inspired by [DeepMind's AlphaZero](https://arxiv.org/abs/1712.01815), which achieved superhuman performance in chess, shogi, and Go, this project applies similar self-play reinforcement learning methodologies to master Othello without human guidance.

## 🧠 Methodology

### Training Algorithm

SHA-ZA employs **Proximal Policy Optimization (PPO)**, an advanced reinforcement learning algorithm developed by OpenAI. PPO balances exploration and exploitation by clipping probability ratios, stabilizing training and improving performance.

The PPO loss function consists of three components:

1. **Policy Loss**: Optimizes the action selection policy

   <div align="center">
   <img src="https://github.com/mohammed-tech-innovator/proximalpolicy-optimization-for-othello-mastery/assets/40921388/173aa008-31b4-452c-b896-7c73c54dee7e" alt="Policy Loss">
   </div>

2. **Value Loss**: Improves state value estimation

   <div align="center">
   <img src="https://github.com/mohammed-tech-innovator/proximalpolicy-optimization-for-othello-mastery/assets/40921388/8105a880-d50f-4477-8bb6-a168dbf09ae2" alt="Value Loss">
   </div>

3. **Entropy Loss**: Encourages exploration

   <div align="center">
   <img src="https://github.com/mohammed-tech-innovator/proximalpolicy-optimization-for-othello-mastery/assets/40921388/2a13670c-b2d3-4dd1-adee-2280e3740f58" alt="Entropy Loss">
   </div>

4. **Total Loss**: Combined objective function

   ```
   Total Loss = Policy Loss + c_v · Value Loss - c_e · Entropy Loss
   ```

### Neural Network Architecture

SHA-ZA uses two deep convolutional neural networks based on [ConvNeXt](https://arxiv.org/abs/2201.03545):
- **Policy Network**: Maps board states to action probability distributions
- **Value Network**: Evaluates board positions

The architecture incorporates:

- **ConvNeXt building blocks**:
  - Depthwise convolutions for parameter efficiency
  - Layer normalization for training stability
  - Residual connections to mitigate gradient vanishing

  <div align="center">
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTTm384qquw_hbC0UIIhF8Jnr9gHvtNokBOCQ&s" alt="ConvNeXt" height="250" style="border-radius: 15px;">
  </div>

- **Convolutional Block Attention Module (CBAM)** to enhance feature representation:
  - Channel attention to emphasize important feature channels
  - Spatial attention to focus on relevant board regions

  <div align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1400/0*cvZx6H1aDsSgqQ1z" alt="CBAM" width="600" style="border-radius: 15px;">
  </div>

### State Representation

The game state is represented using three components:
- **Legal Moves**: Available moves at the current position
- **Pieces Under Attack**: Opponent's pieces that can be captured in one move
- **Board State**: Player's pieces (1) and opponent's pieces (-1)

These components are concatenated to form the input for the neural networks.

### Training Process

- **Training Duration**: 6,000 steps
- **Parallel Simulation**: 16 parallel processes generating 512 games each per step
- **Evaluation**: Every 25 steps against four opponents:
  - Random policy
  - Reference agent (same architecture)
  - Minimax agent (depth 7)
  - Current best agent
- **Hardware**: Nvidia RTX A5000

### Hyperparameters

| Hyperparameter      | Steps 0-3000 | Steps 3000-5200 | Steps 5200-6000 |
|---------------------|--------------|-----------------|-----------------|
| Batch Size          | 256          | 256             | 256             |
| Value Coefficient   | 0.5          | 0.5             | 0.5             |
| Entropy Coefficient | 0.09         | 0.045           | 0.09            |
| Clip Parameter      | 0.2          | 0.2             | 0.2             |
| Learning Rate       | 0.5e-4       | 0.25e-4         | 0.125e-4        |
| Games / Step        | 4096         | 8192            | 4096            |

## 📊 Results

After 5,725 training steps, SHA-ZA achieved remarkable performance:
- **vs. Random policy**: 100% win rate
- **vs. Minimax (depth 7)**: 96.75% win rate
- **vs. Reference agent**: 68.85% win rate

![Win Rate Graph](https://github.com/mohammed-tech-innovator/proximalpolicy-optimization-for-othello-mastery/assets/40921388/35485ac7-ce47-4681-a65d-a65028a95d0d)

### Matches Against Minimax

SHA-ZA consistently defeated Minimax agents at various search depths (6-12), both as first and second player:

| Match   | Minimax Depth | First Player | Winner | Video |
|---------|---------------|--------------|--------|-------|
| Match 1 | 6             | SHA-ZA       | SHA-ZA | [Watch](https://youtu.be/EajVC9woUSM) |
| Match 2 | 6             | Minimax      | SHA-ZA | [Watch](https://youtu.be/o-xnlFO5vpk) |
| Match 3 | 9             | SHA-ZA       | SHA-ZA | [Watch](https://youtu.be/KGrxUoulhko) |
| Match 4 | 9             | Minimax      | SHA-ZA | [Watch](https://youtu.be/_460eicf3s8) |
| Match 5 | 12            | SHA-ZA       | SHA-ZA | [Watch](https://youtu.be/9qXpWrR48Hg) |
| Match 6 | 12            | Minimax      | SHA-ZA | [Watch](https://youtu.be/g2Wx_m37Oac) |

### Key Checkpoints

Model checkpoints were saved at steps: 1000, 1950, 2500, 3000, 4000, 4150, 5025, and 5725.

## 🔍 Key Insights

- SHA-ZA achieved **superhuman Othello performance** through pure self-play, with no initial knowledge beyond game rules
- The agent demonstrates **deep strategic understanding**, making intelligent material sacrifices for positional advantage
- At 0:22-0:32 in the demonstration videos, SHA-ZA maintained a positive value estimate (0.2-0.3) despite being significantly behind in piece count, showing its focus on final game outcome rather than immediate material advantage

## 📦 Installation & Usage

```bash
# Clone the repository
git clone https://github.com/mohammed-tech-innovator/proximalpolicy-optimization-for-othello-mastery.git
cd proximalpolicy-optimization-for-othello-mastery

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook
```
Then open othello.ipynb to explore the training process or run your own experiments.

## 📚 Citation

If you use SHA-ZA in your research, please cite:

```bibtex
@article{yousif2025shaza,
  title={SHA-ZA: Advanced Reinforcement Learning for Othello Mastery Using Proximal Policy Optimization},
  author={Yousif, Mohammed},
  journal={International Journal of Machine Learning},
  volume={15},
  number={1},
  pages={17--22},
  year={2025}
}
```

## 👤 Contact

- **Author**: Mohammed Yousif
- **Email**: [mohammed.yah.yousif@gmail.com](mailto:mohammed.yah.yousif@gmail.com)
- **Website**: [tech-innovator.me](https://tech-innovator.me/)
- **LinkedIn**: [Mohammed Yousif](https://www.linkedin.com/in/mohammed-yousif-6b272a241/)
- **X**: [@Moh_yah_you](https://x.com/Moh_yah_you)
