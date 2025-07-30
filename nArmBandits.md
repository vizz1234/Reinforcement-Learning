
# Solving N - Arm Bandits Problem Using Reinforcement Learning

Disclaimer: This article primarily follows the ideas and theory from **Reinforcement Learning: An Introduction**, book by Andrew Barto and Richard S. Sutton

Before solving multi-arm bandits problem or n-bandits problem, let us familiarize ourselves with some of the basic RL foundations. 

## The Reinforcement Learning Problem  

Reinforcement Learning (RL) in simple sense is a problem where an agent interacts with the environment and takes actions that affect the state that agent is currently in inorder to achieve an objective or goal (Maximizing reward signals). In essense, if a problem includes *sensation*, *action* and *goal* then it can be formulated as a reinforcement problem.  

**Reinforcement Learning** is a different paradigm that either *supervised* or *unsupervised* learning. Supervised Learning is learning from data that has already been categorized; Unsupervised Learning is finding hidden structures in the data that has no labels. Though both of these are important types of learning, they are not enough to learn through interation and to maximize *reward signal*. Hence, reinforcement learning is often considered as the third paradigm of *Machine Learning*.   

![RL Paradigm](./images/RL_paradigm.png)

### Exploration and Exploitation ###

One of the trade-offs and challenges in reinforcement learning is *Exploration* and *Exploitation*. As mentioned before, the agent tries to reach the goal by taking actions that provide maximum reward. But to discover such actions, it has to explore other possible actions. One might wonder, why not try all actions and choose the one that is the best. This is often computationally not feasible because of the vast number of states and actions (Backgammon approximately has 10^20^ states). Even if the agent has tried all the actions applicable in a state in the past and has the knowledge of the best action, it still has to try other actions if the environment is non stationary (environment changes with time).    

Hence, the dilemma; to explore or to exploit. In simple terms, exploit is to take decisions based on known knowledge and explore is to go down other path even when it is known the other action at this particular state might not yield the best possible reward in order to get better reward in the future. So, look out for exploration and exploitation in all the algorithms of reinforcement learning to better understand balancing of the trade-offs. One thumb rule is to have more weightage in favor of exploration if uncertainty is high and exploit otherwise.

### Elements of Reinforcement Learning ###

- **policy:** Mapping of perceived states to actions that are taken in those states.
- **reward signal:** Environment sends rewards to an RL agent upon taking action in a state at different time steps. This is usually a real number. The agent's objective is to maximize the total reward it gets in the long haul.
- **value function:** Reward is usually immediate, i.e, the agent gets the reward after taking an action in a particular state at a particular time. Value Function, however, tells what is the expected rewards an agent can achieve from the said state in the long run. Based on values, decisions are made, i.e, an agent takes actions that produce highest value.

Apart from these three, optionally, *model* can also be considered as an element. The model imitates the environment, i.e, given an action in a state, the model predicts the next state and reward. This learning that uses *planning* and *models* is called model-based learning where future states or situations are considered without even experiencing it. In contast, model-free methods are usually considered opposite of planning, where an agent learns explicitly by trial-and-error.

## Tabular Solution Methods ##

In the case where state and action spaces are relatively small, they can be represented in an array or a table. This gives an opportunity to understand the core ideas of RL algortihms, atleast in their simplest forms.

One more feature that distinguishes RL from other learnings is that it learns on training information that *evaluates* the actions rather than *instruct* the action in a particular state at a specific time. This creates the need for active exploration.

In this problem, evaluative feedback is presented in the simple setting, where the agent has to learn to act only in one situation (non-associate setting), i.e, n-armed bandits problem.

## N - Armed Bandit Problem ##

N - arm bandits refers to a *casino slot machine* where instead of one lever, there are n levers (actions). After each action selection, there is a reward according to the stationary probability distribution. The agent's goal is to maximize total reward over m steps. In this case, 1000 steps, and thereby selecting 1000 actions.

Here casino slot is referred, however, there are many applications of the same. For instance, the same problem can be formulated for online advertising; 
- **n-arm or actions:** n-ads.
- **rewards:** clicks, purchases, etc. 
- **goal:** Maximize total engagement over time.

Python is used to implement the solution. In the below image, the bandit class is initialized with *k* arms or actions. The *k* actions are assigned normally distributed rewards with mean 0 and variance 1. It also has *pull* method that takes an action and returns the reward that has some additional noise. This is due to the fact that real-world actions often have uncertainty. In this instance, pulling a casino lever may not give same payout every time.

![Bandit class initialization](./images/bandit_class.png)

There are multiple solutions to this problem. In this articles, three of them will be discussed; *ε-greedy*, *Upper Confidence Bound*, *Thompson Sampling*.

These are also known as action selection methods as the agent is choosing different actions with unknwon rewards.

### **ε-greedy**

Let *q(a)* represent true value, and *Q(a)* for estimated value. Simple way to estimate *Q(a)* is to average the rewards when that particular action is selected. If action *a* has been chosen *$N_t$(a)*, getting rewards *$R_1, R_2, R_3, ...., R_{Nt(a)}$*, then 

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?Q_{t}(a)=\frac{R_1+R_2+\cdots+R_{N_t(a)}}{N_t(a)}" alt="Q(a)">
</p>

If $N_t(a) = 0$, then $Q_t(a)$ is defined as some default value. In this problem we define it as 0. If $N_t(a)$ tends to ∞, then by law of large numbers, $Q_t(a)$ converges to its true value *q(a)*.

The obvious choice to select action is the one with the highest $Q_t(a)$, this is also known as *greedy* action selection. As it was made clear in the earlier sections, the algorithm has to balance exploration and exploitation. Hence, the agent selects one of the actions randomly with equal probability regardless of the estimate. This is done with probability ε, and with probability 1-ε, the action with the highest estimate is selected.

To have incremental update to the simple average estimate, modifying the above formula, we have

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?Q_{t+1}(a)=Q_t(a)+\alpha(R_t-Q_t(a))" alt="Incremental Update Rule">
</p>

Notice that it is *$R_t$* and not *$R_{t+1}$* as *$Q_1=0$*. Alpha(𝛼) is the step size parameter that is equal to $N_t(a)$. Below is the image of implementation details in python.

![Epsilon Greedy Agent](./images/epsilon_greedy_agent.png)

Now that we have the set-up, let us simulate the experiment. One more thing to note here is that, the experiment is carried out 2000 times, with 1000 steps or iterations each time. These are then averaged out to get the better approximate. This set up will help in visualizing the performance much better (A single run can be extremely noisy and can lead to misleading conclusions especially since the problem has inherent randomness and the agent can get lucky or unlucky based on initial estimates. So, many runs are performed and then averaged to get clear picture). Find the implementation in the picture below.

![simulate](./images/simulate.png)

Now, let's runt the algorithm.

![setup](./images/setup.png)

### ε-greedy Results ###

Now that the algorithm is familiarized, let us look into results.




