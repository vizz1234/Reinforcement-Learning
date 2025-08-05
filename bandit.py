import numpy as np

class NBandit:

  def __init__(self, k):
    self.k = k
    self.qTrue = np.random.normal(0, 4, k)
    self.bestAction = np.argmax(self.qTrue)

  def pull(self, action):
    return np.random.normal(self.qTrue[action], 1)
