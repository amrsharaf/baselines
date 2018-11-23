import math
import gym
import gym_bandit.utils.dp as dp
import gym_bandit.utils.rl as rl
from gym.envs.registration import register
import matplotlib.pyplot as plt
from multiprocessing import Pool
from multiprocessing import cpu_count


class Policy(object):
    def __call__(self, env, observation, costs):
        assert False


class RandomPolicy(object):
    def __call__(self, env, observation, costs):
        # Always random baseline:
        action = env.action_space.sample()
        # Check if this action is allowed
        state = observation_to_state(observation)
        if not rl.is_allowed_action(state=state, action=action, costs=costs):
            action = int(dp.MetaAction.none)
        return action


class NonePolicy(object):
    def __call__(self, env, observation, costs):
        # Always none baseline:
        action = int(dp.MetaAction.none)
        return action


class BanditPolicy(object):
    def __call__(self, env, observation, costs):
        # Always bandit baseline:
        state = observation_to_state(observation=observation)
        action = int(dp.MetaAction.bandit)
        if not rl.is_allowed_action(state=state, action=action, costs=costs):
            action = int(dp.MetaAction.none)
        return action


class SupervisedPolicy(object):
    def __call__(self, env, observation, costs):
        # Always select supvervised action
        state = observation_to_state(observation=observation)
        action = int(dp.MetaAction.supervised)
        if not rl.is_allowed_action(state=state, action=action, costs=costs):
            action = int(dp.MetaAction.none)
        return action


class SinglePolicy(object):
    def __call__(self, env, observation, costs):
        # Always single arm meta-action, a single arm selected uniformly at random
        state = observation_to_state(observation=observation)
        action = int(dp.MetaAction.single)
        if not rl.is_allowed_action(state=state, action=action, costs=costs):
            action = int(dp.MetaAction.none)
        return action


def number_of_arms_pulled(action):
    if action == int(dp.MetaAction.none):
        return 0
    elif action == int(dp.MetaAction.bandit):
        return 1
    elif action == int(dp.MetaAction.single):
        return 1
    else:
        assert action == int(dp.MetaAction.supervised)
        return 2


def run_bandit(n_episodes, env, costs, policy, max_arm_pulls):
    # Reset environment
    env.reset()
    total_reward = 0.0
    # Keep some stats about how many times each arm was pulled in all episodes
    stats = {'counts_arm_0': 0, 'counts_arm_1': 0}
    for episode_id in range(n_episodes):
        # Reset the environment
        observation = env.reset()
        # Total number of arm pulls!
        total_arm_pulls = 0
        done = False
        while not done:
            action = policy(env=env, observation=observation, costs=costs)
            # Update the counter for the number of arms pulled
            total_arm_pulls += number_of_arms_pulled(action)
            # Clip action to none if we exceeded the budget of max arm pulls!
            if total_arm_pulls > max_arm_pulls:
                action = int(dp.MetaAction.none)
            observation, reward, done, info = env.step(action)
            # TODO refactor to keep track of this information in a better way
            if observation[1] == -1:
                state = observation_to_state(observation)
                stats['counts_arm_0'] = stats['counts_arm_0'] + state.counts[0]
                stats['counts_arm_1'] = stats['counts_arm_1'] + state.counts[1]
        total_reward += reward
        average_reward = total_reward / float(episode_id+1)
    print('average reward ', policy, ' baseline: ', average_reward)
    # TODO plot these stats instead of just printing it
    print('stats: ', stats)
    print('=========================================================================================================')
    return average_reward


def observation_to_state(observation):
#    print('got observation: ', observation)
    return observation
    state = rl.State(budget=observation[0], horizon=observation[1], rewards=[observation[2], observation[3]],
                     counts=[observation[4], observation[5]])
    return state


def random_baseline(n_episodes, env, costs, max_arm_pulls):
    return run_bandit(n_episodes=n_episodes, env=env, costs=costs, policy=RandomPolicy(), max_arm_pulls=max_arm_pulls)


def none_baseline(n_episodes, env, costs, max_arm_pulls):
    return run_bandit(n_episodes=n_episodes, env=env, costs=costs, policy=NonePolicy(), max_arm_pulls=max_arm_pulls)


# TODO use allowed actions function
def bandit_baseline(n_episodes, env, costs, max_arm_pulls):
    return run_bandit(n_episodes=n_episodes, env=env, costs=costs, policy=BanditPolicy(), max_arm_pulls=max_arm_pulls)


def supervised_baseline(n_episodes, env, costs, max_arm_pulls):
    return run_bandit(n_episodes=n_episodes, env=env, costs=costs, policy=SupervisedPolicy(),
                      max_arm_pulls=max_arm_pulls)


def single_baseline(n_episodes, env, costs, max_arm_pulls):
    return run_bandit(n_episodes=n_episodes, env=env, costs=costs, policy=SinglePolicy(), max_arm_pulls=max_arm_pulls)


# Creates a muti-arm bandit environment
def register_bandit_env(budget, horizon, costs):
    # Create an instance of the bandit environment
    K = 2
    register(id='bandit-v1', entry_point='gym_bandit.learning:BanditEnv', kwargs={'budget': budget,
                                                                                  'horizon': horizon,
                                                                                  'costs': costs,
                                                                                  'K': K})
    print('registered!')


# TODO print confidence intervals as well
def all_baselines():
    # TODO measure standard deviation
    n_episodes = 200
    budget = 200
    horizon = 200
    supervised_cost = 2
    bandit_cost = 0
    costs = {}
    costs[dp.MetaAction.bandit] = bandit_cost
    costs[dp.MetaAction.supervised] = supervised_cost
    register_bandit_env(budget=budget, horizon=horizon, costs=costs)
    env = gym.make('bandit-v1')
    bandit_baseline(n_episodes, env, supervised_cost=supervised_cost, bandit_cost=bandit_cost, max_arm_pulls=math.inf)
    supervised_baseline(n_episodes, env, supervised_cost=supervised_cost, bandit_cost=bandit_cost,
                        max_arm_pulls=math.inf)
    none_baseline(n_episodes, env, supervised_cost=supervised_cost, bandit_cost=bandit_cost, max_arm_pulls=math.inf)
    random_baseline(n_episodes, env, supervised_cost=supervised_cost, bandit_cost=bandit_cost, max_arm_pulls=math.inf)


def bandit_accuracy_for_threhsold(threshold):
    # TODO can we use one env instance instead of creating a new one each time?
    print('threshold: ', threshold)
    n_episodes = 100
    max_arm_pulls = threshold
    costs = get_costs()
    env = gym.make('bandit-v1')
    bandit_accuracy = bandit_baseline(n_episodes=n_episodes, env=env, costs=costs, max_arm_pulls=max_arm_pulls)
    return bandit_accuracy


def supervised_accuracy_for_threshold(threshold):
    print('threshold: ', threshold)
    n_episodes = 100
    max_arm_pulls = threshold
    costs = get_costs()
    env = gym.make('bandit-v1')
    supervised_accuracy = supervised_baseline(n_episodes=n_episodes, env=env, costs=costs, max_arm_pulls=max_arm_pulls)
    return supervised_accuracy


def single_accuracy_for_threshold(threshold):
    print('threshold: ', threshold)
    n_episodes = 100
    max_arm_pulls = threshold
    costs = get_costs()
    env = gym.make('bandit-v1')
    single_accuracy = single_baseline(n_episodes=n_episodes, env=env, costs=costs, max_arm_pulls=max_arm_pulls)
    return single_accuracy


def get_costs():
    supervised_cost = 2
    bandit_cost = 1
    single_cost = 1
    costs = {}
    costs[dp.MetaAction.bandit] = bandit_cost
    costs[dp.MetaAction.supervised] = supervised_cost
    costs[dp.MetaAction.single] = single_cost
    return costs


def main():
    # TODO abstract away the common parameters with the all_baselines method
    # Create an instance of the bandit environment
    # x-axis: number of arm pulls
    # TODO what is the maximum number of arms to plot?
    max_arm_pulls = 200
    bandit_accuracy = []
    supervised_accuracy = []
    # TODO this is so slow
    step = 10
    thresholds = []
    # Run every threshold in parallel
    n_cpus = cpu_count()
    pool = Pool(n_cpus)
    print('pool: ', pool)
    print('number of cpus: ', n_cpus)
    thresholds = range(0, max_arm_pulls, step)
    budget = 400000
    horizon = 400
    costs = get_costs()
    register_bandit_env(budget=budget, horizon=horizon, costs=costs)
    single_accuracy = list(map(single_accuracy_for_threshold, thresholds))
    bandit_accuracy = list(map(bandit_accuracy_for_threhsold, thresholds))
    supervised_accuracy = list(map(supervised_accuracy_for_threshold, thresholds))
    # Now we need to get the best arm identification accuracy for the y_axis for both bandit and supervised baselines
    # I think we need to update the run function to take extra input the maximum allowed arm pulls
    print('thresholds: ', thresholds)
    print('bandit_accuracy: ', bandit_accuracy)
    print('supervised_accuracy: ', supervised_accuracy)
    plt.plot(thresholds, bandit_accuracy, label='bandit')
    plt.plot(thresholds, supervised_accuracy, label='supervised')
    plt.plot(thresholds, single_accuracy, label='single')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
