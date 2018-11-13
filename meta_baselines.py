import gym
import gym_bandit.utils.dp as dp
import gym_bandit.utils.rl as rl
from gym.envs.registration import register


def observation_to_state(observation):
    state = rl.State(budget=observation[0], horizon=observation[1], rewards_arm_0=observation[2],
                     rewards_arm_1=observation[3], counts_arm_0=observation[4], counts_arm_1=observation[5])
    return state


def random_baseline(n_episodes, env, supervised_cost, bandit_cost):
    # Reset environment
    env.reset()
    total_reward = 0.0
    for episode_id in range(n_episodes):
        observation = env.reset()
        done = False
        while not done:
            # Always random baseline:
            action = env.action_space.sample()
            # Check if this action is allowed
            state = observation_to_state(observation)
            if not rl.is_allowed_action(state=state, action=action, supervised_cost=supervised_cost,
                                        bandit_cost=bandit_cost):
                action = int(dp.MetaAction.none)
            observation, reward, done, info = env.step(action)
        total_reward += reward
        average_reward = total_reward / float(episode_id+1)
    print('average reward random baseline: ', average_reward)


def none_baseline(n_episodes, env, supervised_cost, bandit_cost):
    # Reset environment
    env.reset()
    total_reward = 0.0
    for episode_id in range(n_episodes):
        observation = env.reset()
        done = False
        while not done:
            # Always none baseline:
            action = int(dp.MetaAction.none)
            observation, reward, done, info = env.step(action)
        total_reward += reward
        average_reward = total_reward / float(episode_id+1)
    print('average reward none baseline: ', average_reward)


# TODO use allowed actions function
def bandit_baseline(n_episodes, env, supervised_cost, bandit_cost):
    # Reset environment
    env.reset()
    total_reward = 0.0
    for episode_id in range(n_episodes):
        observation = env.reset()
        done = False
        while not done:
            # Always bandit baseline:
            state = observation_to_state(observation=observation)
            action = int(dp.MetaAction.bandit)
            if not rl.is_allowed_action(state=state, action=action, supervised_cost=supervised_cost,
                                        bandit_cost=bandit_cost):
                action = int(dp.MetaAction.none)
            observation, reward, done, info = env.step(action)
        total_reward += reward
        average_reward = total_reward / float(episode_id+1)
    print('average reward bandit baseline: ', average_reward)


def supervised_baseline(n_episodes, env, supervised_cost, bandit_cost):
    env.reset()
    total_reward = 0.0
    for episode_id in range(n_episodes):
        observation = env.reset()
        done = False
        while not done:
            # Always select supvervised action
            state = observation_to_state(observation=observation)
            action = int(dp.MetaAction.supervised)
            if not rl.is_allowed_action(state=state, action=action, supervised_cost=supervised_cost,
                                        bandit_cost=bandit_cost):
                action = int(dp.MetaAction.none)
            observation, reward, done, info = env.step(action)
        total_reward += reward
        average_reward = total_reward / float(episode_id+1)
    print('average reward so far supervised baseline: ', average_reward)


def main():
    # TODO measure standard deviation
    n_episodes = 200
    budget = 200000
    horizon = 200
    supervised_cost = 2
    bandit_cost = 1
    # Create an instance of the bandit environment
    register(id='bandit-v1', entry_point='gym_bandit.learning:BanditEnv', kwargs={'budget': budget,
                                                                                  'horizon': horizon,
                                                                                  'supervised_cost': supervised_cost,
                                                                                  'bandit_cost': bandit_cost})
    env = gym.make('bandit-v1')
    bandit_baseline(n_episodes, env, supervised_cost=supervised_cost, bandit_cost=bandit_cost)
    supervised_baseline(n_episodes, env, supervised_cost=supervised_cost, bandit_cost=bandit_cost)
    none_baseline(n_episodes, env, supervised_cost=supervised_cost, bandit_cost=bandit_cost)
    random_baseline(n_episodes, env, supervised_cost=supervised_cost, bandit_cost=bandit_cost)


if __name__ == '__main__':
    main()
