class greedyAgent:

  def __init__(self, k, epsilon = 0):
    self.k = k
    self.epsilon = epsilon
    self.qEstimate = np.zeros(k)
    self.actionCount = np.zeros(k)
  
  def getAction(self):
    if np.random.rand() < self.epsilon:
      return np.random.randint(self.k)
    else:
      return np.argmax(self.qEstimate)
  
  def update(self, action, reward):
    self.actionCount[action] += 1
    alpha = 1 / self.actionCount[action]
    self.qEstimate[action] += alpha * (reward - self.qEstimate[action]) 
