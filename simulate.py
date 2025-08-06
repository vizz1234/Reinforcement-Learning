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
