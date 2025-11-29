# CSCI-166-FINAL-PROJECT
Deep Reinforcement Learning on Atari Pong 

A comparative study of Deep Q-Learning architectures trained to master the Atari 2600 game PongNoFrameskip-v4. This project implements a baseline Deep Q-Network (DQN) and compares it against an advanced variant utilizing Double DQN and Dueling Network architectures.



The goal of this project is to train an autonomous agent to play Pong from raw pixels without any prior knowledge of the game rules.

Environment: PongNoFrameskip-v4 (Gymnasium/Ale-py)

Input: Stack of 4 grayscale frames ($84 \times 84$ px)

Action Space: Discrete (6 actions)

Reward Signal: +1 (Win point), -1 (Lose point)

Implemented Architectures

Baseline DQN: Standard Mnih et al. (2015) architecture.

Double Dueling DQN:

Double DQN: Decouples action selection from evaluation to reduce overestimation bias.

Dueling Head: Splits the network into Value $V(s)$ and Advantage $A(s,a)$ streams for faster convergence.



This project is designed to run in a Jupyter Notebook environment (Google Colab recommended for GPU access).


For Baseline: Set VARIANT_MODE = False

For Advanced Variant: Set VARIANT_MODE = True

Results

The advanced architecture demonstrated significantly faster convergence and stability compared to the baseline.

Metric

Baseline DQN

Double Dueling DQN

Convergence Time

~150k frames

~80k frames

Final Mean Reward

+18.0

+19.5

Stability

High Variance

Smooth / Consistent

Learning Curve

🎥 Video Demonstrations

Early Training (Random Behavior)
Baseline
![Baseline_Standard_early-episode-0](https://github.com/user-attachments/assets/a51d6087-875b-447a-ab26-a5c0a12cb267)

DDQN


![Variant_DDQN_Dueling_early-episode-0](https://github.com/user-attachments/assets/53f80946-a317-4a14-869a-42e9fdc7a86c)



The agent moves jitterily and fails to track the ball effectively.


Late Training (Learned Behavior)

The agent tracks the ball perfectly and utilizes edge-shots to defeat the opponent.
Baseline
![Baseline_Standard_late-episode-0](https://github.com/user-attachments/assets/f056c905-78f1-4054-97ea-ff9f8911706a)


DDQN

![Variant_DDQN_Dueling_late-episode-0](https://github.com/user-attachments/assets/0677613d-4518-4d71-8024-d4190bffc98f)

Reflection

One of the key challenges in this project was the computational cost of training on high-dimensional pixel inputs. Switching from CPU to a T4 GPU reduced training time from ~12 hours to ~45 minutes. The Dueling architecture proved particularly effective for Pong, as it allowed the agent to learn the value of "safe" states without needing to explore every specific action outcome.



Starter Code: Adapted from c166f25_02b_dqn_pong.ipynb
