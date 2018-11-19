import math
import gym
import gym_bandit.utils.dp as dp
import gym_bandit.utils.rl as rl
from gym.envs.registration import register
import numpy as np
import matplotlib.pyplot as plt


class Policy(object):
    def __call__(self, env, observation, supervised_cost, bandit_cost):
        assert False


class RandomPolicy(object):
    def __call__(self, env, observation, supervised_cost, bandit_cost):
        # Always random baseline:
        action = env.action_space.sample()
        # Check if this action is allowed
        state = observation_to_state(observation)
        if not rl.is_allowed_action(state=state, action=action, supervised_cost=supervised_cost,
                                    bandit_cost=bandit_cost):
            action = int(dp.MetaAction.none)
        return action


class NonePolicy(object):
    def __call__(self, env, observation, supervised_cost, bandit_cost):
        # Always none baseline:
        action = int(dp.MetaAction.none)
        return action


class BanditPolicy(object):
    def __call__(self, env, observation, supervised_cost, bandit_cost):
        # Always bandit baseline:
        state = observation_to_state(observation=observation)
        action = int(dp.MetaAction.bandit)
        if not rl.is_allowed_action(state=state, action=action, supervised_cost=supervised_cost,
                                    bandit_cost=bandit_cost):
            action = int(dp.MetaAction.none)
        return action


class SupervisedPolicy(object):
    def __call__(self, env, observation, supervised_cost, bandit_cost):
        # Always select supvervised action
        state = observation_to_state(observation=observation)
        action = int(dp.MetaAction.supervised)
        if not rl.is_allowed_action(state=state, action=action, supervised_cost=supervised_cost,
                                    bandit_cost=bandit_cost):
            action = int(dp.MetaAction.none)
        return action


def number_of_arms_pulled(action):
    if action == int(dp.MetaAction.none):
        return 0
    elif action == int(dp.MetaAction.bandit):
        return 1
    else:
        assert action == int(dp.MetaAction.supervised)
        return 2


def run_bandit(n_episodes, env, supervised_cost, bandit_cost, policy, max_arm_pulls):
    # Reset environment
    env.reset()
    total_reward = 0.0
    # Total number of arm pulls!
    total_arm_pulls = 0
    for episode_id in range(n_episodes):
        observation = env.reset()
        done = False
        while not done:
            action = policy(env=env, observation=observation, supervised_cost=supervised_cost, bandit_cost=bandit_cost)
            # Update the counter for the number of arms pulled
            total_arm_pulls += number_of_arms_pulled(action)
            # Clip action to none if we exceeded the budget of max arm pulls!
            if total_arm_pulls > max_arm_pulls:
                action = int(dp.MetaAction.none)
            observation, reward, done, info = env.step(action)
        total_reward += reward
        average_reward = total_reward / float(episode_id+1)
    print('average reward random baseline: ', average_reward)


def observation_to_state(observation):
    state = rl.State(budget=observation[0], horizon=observation[1], rewards_arm_0=observation[2],
                     rewards_arm_1=observation[3], counts_arm_0=observation[4], counts_arm_1=observation[5])
    return state


def random_baseline(n_episodes, env, supervised_cost, bandit_cost, max_arm_pulls):
    return run_bandit(n_episodes=n_episodes, env=env, supervised_cost=supervised_cost, bandit_cost=bandit_cost,
                      policy=RandomPolicy(), max_arm_pulls=max_arm_pulls)


def none_baseline(n_episodes, env, supervised_cost, bandit_cost, max_arm_pulls):
    return run_bandit(n_episodes=n_episodes, env=env, supervised_cost=supervised_cost, bandit_cost=bandit_cost,
                      policy=NonePolicy(), max_arm_pulls=max_arm_pulls)


# TODO use allowed actions function
def bandit_baseline(n_episodes, env, supervised_cost, bandit_cost, max_arm_pulls):
    return run_bandit(n_episodes=n_episodes, env=env, supervised_cost=supervised_cost, bandit_cost=bandit_cost,
                      policy=BanditPolicy(), max_arm_pulls=max_arm_pulls)


def supervised_baseline(n_episodes, env, supervised_cost, bandit_cost, max_arm_pulls):
    return run_bandit(n_episodes=n_episodes, env=env, supervised_cost=supervised_cost, bandit_cost=bandit_cost,
                      policy=SupervisedPolicy(), max_arm_pulls=max_arm_pulls)


# TODO print confidence intervals as well
def all_baselines():
    # TODO measure standard deviation
    n_episodes = 200
    budget = 200
    horizon = 200
    supervised_cost = 2
    bandit_cost = 0
    # Create an instance of the bandit environment
    register(id='bandit-v1', entry_point='gym_bandit.learning:BanditEnv', kwargs={'budget': budget,
                                                                                  'horizon': horizon,
                                                                                  'supervised_cost': supervised_cost,
                                                                                  'bandit_cost': bandit_cost})
    env = gym.make('bandit-v1')
    bandit_baseline(n_episodes, env, supervised_cost=supervised_cost, bandit_cost=bandit_cost, max_arm_pulls=math.inf)
    supervised_baseline(n_episodes, env, supervised_cost=supervised_cost, bandit_cost=bandit_cost,
                        max_arm_pulls=math.inf)
    none_baseline(n_episodes, env, supervised_cost=supervised_cost, bandit_cost=bandit_cost, max_arm_pulls=math.inf)
    random_baseline(n_episodes, env, supervised_cost=supervised_cost, bandit_cost=bandit_cost, max_arm_pulls=math.inf)


def main():
    # x-axis: number of arm pulls
    # TODO what is the maximum number of arms to plot?
    max_arm_pulls = 100
    arm_pulls = np.arange(max_arm_pulls)
    accuracy = np.zeros(max_arm_pulls)
    print('accuracy: ', accuracy)
    # Now we need to get the best arm identification accuracy for the y_axis for both bandit and supervised baselines
    # I think we need to update the run function to take extra input the maximum allowed arm pulls
    plt.plot(arm_pulls)
    plt.show()


if __name__ == '__main__':
    all_baselines()
#    main()
