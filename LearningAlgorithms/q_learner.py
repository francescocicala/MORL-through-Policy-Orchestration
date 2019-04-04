import numpy as np
import abc

def L2_normalization(vec):
  return vec / np.linalg.norm(vec)

class Q_Learner(abc.ABC):
  """
  Q-learning agent for an environment where the set of possibile actions
  is finite.
  """
  def __init__(self, actions_arr, d, learning_rate, epsilon, discount_factor, theta=None):
    self.actions_arr = actions_arr
    self.d = d # number of features taken into account in the Q-function
    self.learning_rate = learning_rate # learning rate
    self.epsilon = epsilon # we use an epsilon-greedy strategy
    self.discount_factor = discount_factor # discounting factor

    if theta == None:
      self.theta = L2_normalization(np.random.normal(0, 0.1, d))

    elif len(theta) == d:
      self.theta = L2_normalization(d)

    else:
      raise ValueError("theta length ({}) is different from d ({})".format(len(self.theta), d))


  @abc.abstractmethod
  def features(self, state, action):
    pass


  def Q(self, state, action):
    return self.features(state, action).dot(self.theta)


  def best_action(self, state, training=False):
    u = np.random.uniform()
    if u > self.epsilon and training:
      return self.actions_arr[np.random.randint(len(self.actions_arr))]
    else:
      table = np.asarray([self.Q(state, action) for action in self.actions_arr])
      return self.actions_arr[np.argmax(table)]


  def difference(self, old_state, new_state, action, reward):
    return reward + self.discount_factor * self.best_action(new_state) - self.Q(old_state, action)


  def update_parameters(self, old_state, new_state, action, reward):
    diff = self.difference(old_state, new_state, action, reward)
    self.theta += self.learning_rate * diff * self.features(old_state, action)
    self.theta = L2_normalization(self.theta) # some noise is added in this step 


class My_Q_Learner(Q_Learner):
  def features(self, state, action):
    return state