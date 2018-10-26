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
        actor = Actor(n_state=input_size, n_action=output_size, n_layers=1, n_units=400, scope="actor")
        actor_copy = Actor(n_state=input_size, n_action=output_size, n_layers=1, n_units=400, scope="a_copy")
        critic = Critic(n_state=input_size, n_action=output_size, n_layers=1, n_units=400, scope="critic")
        critic_copy = Critic(n_state=input_size, n_action=output_size, n_layers=1, n_units=400, scope="c_copy")

    with tf.name_scope("train"):
        actor_copy_ops = get_copy_var_ops(actor_copy.get_variables(), actor.get_variables())
        # get_copy_var_ops(dest_scope_name="actor_copy", src_scope_name="actor")
        critic_copy_ops = get_copy_var_ops(critic_copy.get_variables(), critic.get_variables())
        #get_copy_var_ops(dest_scope_name="critic_copy", src_scope_name="critic")
        actor_soft_copy_ops = get_copy_var_ops(actor_copy.get_variables(), actor.get_variables(), "soft")
        #get_copy_var_ops(dest_scope_name="actor_copy", src_scope_name="actor", op_name="soft")
        critic_soft_copy_ops = get_copy_var_ops(critic_copy.get_variables(), critic.get_variables(), "soft")
        #get_copy_var_ops(dest_scope_name="critic_copy", src_scope_name="critic", op_name="soft")

    with tf.name_scope("miscellaneous"):
        init = tf.global_variables_initializer()
        noise_generator = Uhlenbeck(action_dimension=output_size, mu=0.6)
        saver = tf.train.Saver()

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
            loss = 0.0
            while not done:
                env.render()
                action = actor.action(state) + noise_generator.noise()
                next_state, reward, done, _ = env.step(action)

                replay_buffer.append((state, action, reward, next_state, done))
                if len(replay_buffer) > REPLAY_MEMORY:
                    replay_buffer.popleft()

                state = next_state
                step_count += 1
                if step_count % 100 == 1:
                    print("Step {}, chosed action {}, reward {}".format(step_count, action, reward))

                if len(replay_buffer) < 64:
                    continue

                mini_batch = random.sample(replay_buffer, 64)
                loss = replay_train(critic, critic_copy, actor, actor_copy, mini_batch)
                sess.run([actor_soft_copy_ops, critic_soft_copy_ops])

                if done:
                    print("Loss : {}".format(loss))

            if episode % 10 == 1:
                print("Episode: {} steps: {}".format(episode, step_count))
                print("Loss : {}".format(loss))
                save_path = saver.save(sess, "./model.ckpt")


            # bot_play(actor)

if __name__ == "__main__":
    Main()