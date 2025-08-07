def simulate(runs, steps, bandits, epsilon):

  rewardsAvg = np.zeros(steps)
  optimalActionCounts = np.zeros(steps)
  trueBestReward = []

  for i in range(runs):
    bandit = NBandit(bandits)
    agent = UCBAgent(bandits, epsilon)
    trueBestReward.append(bandit.qTrue[bandit.bestAction])

    for j in range(steps):
      action = agent.getAction()
      reward = bandit.pull(action)
      agent.update(action, reward)
      rewardsAvg[j] += reward
      if action == bandit.bestAction:
        optimalActionCounts[j] += 1
  
  rewardsAvg /= runs
  optimalActionCounts /= runs

  return rewardsAvg, optimalActionCounts, np.mean(np.array(trueBestReward))

bandits = 10
steps = 1000
runs = 2000
epsilons = [2]
rewards, optimalCounts, labels, trueReward = [], [], [], []
for eps in epsilons:
    avgR, optA, tR = simulate(runs, steps, bandits, eps)
    rewards.append(avgR)
    optimalCounts.append(optA)
    trueReward.append(tR)

plt.figure(figsize=(12, 5))
# for i in range(len(experiments)):
plt.plot(rewards[0], label='UCB with c = 2')
plt.plot(rewards[2], label = 'ε = 0.1 greedy')
plt.axhline(y=np.mean(np.array(tR)), color='red', linestyle='--', linewidth=2, label='Avg True Best Reward')
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.title("Average Reward over Time")
plt.legend()
plt.grid(True)
plt.show()

# Plot: Optimal Action %
plt.figure(figsize=(12, 5))
plt.plot(optimalCounts[0], label = 'UCB with c = 2')
plt.plot(optimalCounts[2], label= 'ε = 0.1 greedy')
plt.xlabel("Steps")
plt.ylabel("% Optimal Action Chosen")
plt.title("Optimal Action Percentage over Time")
plt.legend()
plt.grid(True)
plt.show()
