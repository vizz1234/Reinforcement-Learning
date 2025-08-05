class UCBAgent:
    def __init__(self, k, c=2):
        self.k = k
        self.c = c
        self.qEstimate = np.zeros(k)
        self.actionCount = np.zeros(k)
        self.totalSteps = 0

    def getAction(self):
        self.totalSteps += 1
        ucbValues = np.zeros(self.k)
        for a in range(self.k):
            if self.actionCount[a] == 0:
                return a  # Ensure each action is tried at least once
            bonus = self.c * np.sqrt(np.log(self.totalSteps) / self.actionCount[a])
            ucbValues[a] = self.qEstimate[a] + bonus
        return np.argmax(ucbValues)

    def update(self, action, reward):
        self.actionCount[action] += 1
        alpha = 1 / self.actionCount[action]
        self.qEstimate[action] += alpha * (reward - self.qEstimate[action])
