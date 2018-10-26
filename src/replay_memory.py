import numpy as np
import tensorflow as tf
import gym

from model.actor import Actor
from model.critic import Critic

env = gym.make("MountainCarContinuous-v0")
# env._max_episode_steps = 10000

input_size = env.observation_space.shape[0]
output_size = env.action_space.shape[0]

dis = 0.99
REPLAY_MEMORY = 10000
tao = 1e-3
printed = False

def replay_train(critic : Critic, critic_copy : Critic, actor : Actor, actor_copy : Actor, train_batch):
    state_stack = np.empty(input_size).reshape(1, input_size)
    action_stack = np.empty(output_size).reshape(1, output_size)
    sampled_action_stack = np.empty(output_size).reshape(1, output_size)
    y_stack = np.empty(output_size).reshape(1, output_size)

    for state, action, reward, next_state, done in train_batch:
        a = np.empty(output_size).reshape(1, output_size)
        s_a = np.empty(output_size).reshape(1, output_size)
        y = np.empty(output_size).reshape(1, output_size)

        sampled_action_copy = actor_copy.action(next_state)
        sampled_action = actor.action(state)
        sampled_q_value = critic_copy.q_value(next_state, sampled_action_copy)
        state = np.reshape(state, newshape=(1, input_size))

        if done:
            y[0, output_size - 1] = reward
        else:
            y[0, output_size - 1] = reward + dis * sampled_q_value[0][0]

        a[0, output_size - 1] = action
        s_a[0, output_size - 1] = sampled_action

        state_stack = np.vstack([state_stack, state])
        action_stack = np.vstack([action_stack, a])
        sampled_action_stack = np.vstack([sampled_action_stack, s_a])
        y_stack = np.vstack([y_stack, y])

    state_stack = np.delete(state_stack, 0, 0)
    action_stack = np.delete(action_stack, 0, 0)
    sampled_action_stack = np.delete(sampled_action_stack, 0, 0)
    y_stack = np.delete(y_stack, 0, 0)

    loss, _ = critic.update(state_stack, action_stack, y_stack)
    gradient = critic.get_gradient(state_stack, sampled_action_stack)
    actor.update(state_stack, gradient)

    return loss

def get_copy_var_ops(dest_vars, src_vars, op_name="init"):
    # Copy Variables src_scope to dest_scope
    op_holder = []

    #src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    #dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    print(src_vars)

    if op_name == "init":
        for src_var, dest_var in zip(src_vars, dest_vars):
            op_holder.append(dest_var.assign(src_var.value()))
        print(len(op_holder))
    elif op_name == "soft":
        for src_var, dest_var in zip(src_vars, dest_vars):
            op_holder.append(dest_var.assign((1.0 - tao) * dest_var.value() + tao * src_var.value()))
        print(len(op_holder))
    return op_holder

def bot_play(actor : Actor):
    # See our trained network in action
    s = env.reset()
    reward_sum = 0

    while True:
        env.render()
        a = actor.action(s)
        s, reward, done, _ = env.step(a)
        reward_sum += reward

        if done:
            print("Total score: {}".format(reward_sum))
            break
