import tensorflow as tf
import numpy as np
from collections import deque
import random

from model.critic import Critic
from model.actor import Actor
from src.replay_memory import bot_play, get_copy_var_ops, replay_train, env
from src.replay_memory import input_size, output_size, REPLAY_MEMORY
from src.Uhlenbeck import Uhlenbeck

def Main():
    max_episodes = 50000
    replay_buffer = deque()

    with tf.name_scope("network"):
        actor = Actor(n_state=input_size, n_action=output_size, n_layers=3, n_units=1000, scope="actor")
        actor_copy = Actor(n_state=input_size, n_action=output_size, n_layers=3, n_units=1000, scope="actor_copy")
        critic = Critic(n_state=input_size, n_action=output_size, n_layers=3, n_units=1000, scope="critic")
        critic_copy = Critic(n_state=input_size, n_action=output_size, n_layers=3, n_units=1000, scope="critic_copy")

    with tf.name_scope("train"):
        actor_copy_ops = get_copy_var_ops(dest_scope_name="actor_copy", src_scope_name="actor")
        critic_copy_ops = get_copy_var_ops(dest_scope_name="critic_copy", src_scope_name="critic")
        actor_soft_copy_ops = get_copy_var_ops(dest_scope_name="actor_copy", src_scope_name="actor", op_name="soft")
        critic_soft_copy_ops = get_copy_var_ops(dest_scope_name="critic_copy", src_scope_name="critic", op_name="soft")

    with tf.name_scope("miscellaneous"):
        init = tf.global_variables_initializer()
        noise_generator = Uhlenbeck(action_dimension=output_size)

    with tf.Session() as sess:
        # initialize variables
        sess.run(init)
        # copy the variables
        sess.run([actor_copy_ops,critic_copy_ops])
        # set the current session to models
        actor.set_session(sess)
        actor_copy.set_session(sess)
        critic.set_session(sess)
        critic_copy.set_session(sess)
        # iterate through the episodes
        for episode in range(max_episodes):
            done = False
            step_count = 0
            state = env.reset()
            noise_generator.reset()
            while not done:
                action = actor.action(state) + noise_generator.noise()
                action = np.clip(action, -1.0, 1.0)

                next_state, reward, done, _ = env.step(action)

                if done:
                    reward = 100

                replay_buffer.append((state, action, reward, next_state, done))
                if len(replay_buffer) > REPLAY_MEMORY:
                    replay_buffer.popleft()

                state = next_state
                step_count += 1

                # if step_count > 10000:
                    # pass
            print(step_count)
            if episode % 10 == 1:
                for _ in range(50):
                    mini_batch = random.sample(replay_buffer, 10)
                    loss = replay_train(critic, critic_copy, actor, actor_copy, mini_batch)
                    sess.run([actor_soft_copy_ops, critic_soft_copy_ops])
                print("Episode: {} steps: {}".format(episode, step_count))
                print("Loss : {}".format(loss))

            # bot_play(actor)

if __name__ == "__main__":
    Main()