import numpy as np

class Orchestrator():
  def __init__(self, n_arms, d, l, g, R, z):
    self.n_arms = n_arms # number of arms
    self.d = d # size of the context vector

    # lambda, array of importances over objectives; for clarity, sums to one
    # it has the same length as the number of arms
    if n_arms == 2:
      self.l = np.asarray([l, 1-l])
    else:
      self.l = l

    self.g = g # gamma, discounting rate
    self.v = R * np.sqrt(24 / z * d * np.log(1 / g)) # empirical parameter

    self.B = np.asarray([np.identity(d)]*n_arms) # n_arms covariance matrices
    self.mu = np.zeros((n_arms, d)) # n_arms means for the context features coefficients
    self.f = np.zeros((n_arms, d))


  def context(self, state):
    c1 = features_extractor.feature1(state)
    c2 = features_extractor.feature2(state)
    return np.asarray([[c1, c2]]) # size (d, 1)


  def best_arm(self, state):
    # extract n_arms vectors of d-dim vectors for context coefficients according
    # to the present beliefs
    mu_sample = np.asarray([np.random.multivariate_normal(mu[i], v * v * np.linalg.inv(B[i])) 
                            for i in range(n_arms)])

    # context vector
    c = self.context(state)

    # return the best arm
    return np.argmax(mu_sample.dot(c))


  def update_beliefs(self, old_state, rewards):
    # context in which best_arm was chosen
    c = self.context(old_state)

    # total reward; Value functions have to be added
    total_reward = rewards.dot(self.l)

    # update self.B[best_arm]
    self.B[best_arm] += B[best_arm] + np.matmul(c, c.T)

    # update self.f[best_arm]
    self.f[best_arm] += c.squeeze() * total_reward

    # update self.mu[best_arm]
    self.mu[best_arm] += np.linalg.inv(self.B[best_arm]).dot(self.f)