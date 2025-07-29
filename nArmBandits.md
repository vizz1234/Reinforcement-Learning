
# N - Bandits Problem

Disclaimer: This article primarily follows the ideas and theory from **Reinforcement Learning: An Introduction**, book by Andrew Barto and Richard S. Sutton

Before solving multi-arm bandits problem or n-bandits problem, let us familiarize ourselves with some of the basic RL foundations. 

## The Reinforcement Learning Problem  

Reinforcement Learning in simple sense is a problem where an agent interacts with the environment and takes actions that affect the state that agent is currently in inorder to achieve an objective or goal (Maximizing reward signals). In essense, if a problem includes *sensation*, *action* and *goal* then it can be formulated as a reinforcement problem.  

**Reinforcement Learning** is a different paradigm that either *supervised* or *unsupervised* learning. Supervised Learning is learning from data that has already been categorized; Unsupervised Learning is finding hidden structures in the data that has no labels. Though both of these are important types of learning, they are not enough to learn through interation and to maximize *reward signal*. Hence, reinforcement learning is often considered as the third paradigm of *Machine Learning*.   

### Exploration and Exploitation ###

One of the trade-offs and challenges in reinforcement learning is *Exploration* and *Exploitation*. As mentioned before, the agent tries to reach the goal by taking actions that provide maximum reward. But to discover such actions, it has to explore other possible actions. One might wonder, why not try all actions and choose the one that is the best. This is often computationally not feasible because of the vast number of states and actions (Backgammon approximately has 10^20^ states). Even if the agent has tried all the actions applicable in a state in the past and has the knowledge of the best action, it still has to try other actions if the environment is non stationary (environment changes with time).    

Hence, the dilemma; to explore or to exploit. In simple terms, exploit is to take decisions based on known knowledge and explore is to go down other path even when it is known the other action at this particular state might not yield the best possible reward in order to get better reward in the future. So, look out for exploration and exploitation in all the algorithms of reinforcement learning to better understand balancing of the trade-offs. One thumb rule is to have more weightage in favor of exploration if uncertainty is high and exploit otherwise.

### Elements of Reinforcement Learning ###

- **policy:** Mapping of perceived states to actions that are taken in those states.
- **reward signal:** Environment sends rewards to an RL agent upon taking action in a state at different time steps. This is usually a real number. The agent's objective is to maximize the total reward it gets in the long haul.
- **value function:** Reward is usually immediate, i.e, the agent gets the reward after taking an action in a particular state at a particular time. Value Function, however, tells what is the expected rewards an agent can achieve from the said state in the long run. Based on values, decisions are made, i.e, an agent takes actions that produce highest value.

Apart from these three, optionally, *model* can also be considered as an element. The model imitates the environment, i.e, given an action in a state, the model predicts the next state and reward. This learning that uses *planning* and *models* is called model-based learning where future states or situations are considered without even experiencing it. In contast, model-free methods are usually considered opposite of planning, where an agent learns explicitly by trial-and-error.




