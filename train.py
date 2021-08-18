import argparse
import numpy as np
import os
import time
import pickle
import torch
import sys
from algorithm import ATOC_trainer
import matplotlib.pyplot as plt
import xlwt
import math
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_spread", help="name of the scenario script")
    parser.add_argument("--max_episode_len", type=int, default=50, help="maximum episode length")
    parser.add_argument("--num_episodes", type=int, default=100, help="number of episodes")
    parser.add_argument("--T", type=int, default=15, help="number of step to initiate a communicagtion group")
    parser.add_argument("--m", type=int, default=2, help="number agents in a communicagtion group")
    # Core training parameters
    parser.add_argument("--actor_lr", type=float, default=3e-4, help="learning rate for actor")
    parser.add_argument("--critic_lr", type=float, default=1e-3, help="learning rate for critic")
    parser.add_argument("--gamma", type=float, default=0.96, help="discount factor")
    parser.add_argument("--actor_hidden_size", type=int, default=128, help="number of units in the actor network")
    parser.add_argument("--critic_hidden_size", type=int, default=128, help="number of units in the critic network")
    parser.add_argument("--tau", type=float, default=0.001, metavar='G', help='discount factor for model (default: 0.001)')
    parser.add_argument("--memory_size", type=int, default=2000, help='size of the replay memory')
    parser.add_argument("--warmup_size", type=int, default=300, help='number of steps before training, must larger than batch_size')
    parser.add_argument("--batch_size", type=int, default=64, help="number of steps to optimize at the same time")
    # Random process
    parser.add_argument("--ou_theta", type=float, default=0.15, help="noise theta")
    parser.add_argument("--ou_sigma", type=float, default=0.2, help="noise sigma")
    parser.add_argument("--ou_mu", type=float, default=0.0, help="noise mu")
    # Checkpointing
    parser.add_argument("--exp_name", type=str, default='test', help="name of the experiment")
    parser.add_argument("--save_path", type=str, default="", help="directory in which training state and model should be saved")
    parser.add_argument("--save_rate", type=int, default=100, help="save model once every time this many episodes are completed")
    # Evaluation
    parser.add_argument("--load", type=str, default="", help="which model to load")
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots_dir", type=str, default="./results/", help="directory where plot data is saved")
    return parser.parse_args()


def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world()
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)  # reset, reward, obs are callbacks
    return env


def train(arglist):
    arglist.benchmark = True
    env = make_env(arglist.scenario, arglist, arglist.benchmark)
    trainer = ATOC_trainer(arglist.gamma, arglist.tau, arglist.actor_hidden_size, arglist.critic_hidden_size, env.observation_space[0], env.action_space[0], arglist)

    # if arglist.display or arglist.restore or arglist.benchmark:
    #     trainer.load_model(arglist.exp_name, suffix=arglist.load)

    action_noise = False if arglist.display else True
    episode_step = 0
    train_step = 0
    agent_rewards = [[0.0] for _ in range(env.n)]
    episode_rewards = [0.0]
    final_save_rewards = []                         # sum of rewards for training curve
    time_start = time.time()
    obs_n = env.reset()
    C = None                                        # Communication group
    reward = []
    delay = [0.0]
    energy = [0.0]
    record_C = []
    occupy = [0.0]
    nagents = len(obs_n)
    if int(nagents/2)==2:
        max_num = 3
    else:
        max_num = int(nagents/2)
    arglist.m = np.random.randint(2, max_num)
    listC = []
    for i in range(nagents):
        listC.append(0)
    record_C.append(listC)
    print("arglist.m",arglist.m)

    print('Starting iterations...')
    while True:
        env.render()
        thoughts = trainer.get_thoughts(obs_n)                  # tensor(nagents, actor_hidden_size)
        if (episode_step % arglist.T == 0) or (C == None):
            C,total_dist = trainer.initiate_group(obs_n, arglist.m, thoughts)
        # TODO
        inter_thoughts = trainer.update_thoughts(thoughts, C)   # (nagents, actor_hidden_size)
        action_n = trainer.select_action(thoughts, inter_thoughts, C)

        # TODO calc delta Q for the training of the attention unit
        trainer.calc_delta_Q(obs_n, action_n, thoughts, C)


        # if energy_n!= 0:
        #     print(energy_n)

        listC = []

        for c_index in range(C.shape[0]):

            if C[c_index][c_index]:
                # print(record_C[-1])
                # print(record_C[-1][c_index])
                listC.append(1+int(record_C[-1][c_index]))
            else:
                listC.append(0+int(record_C[-1][c_index]))
            # listC.append(C[c_index][c_index])

        record_C[-1] = listC


        # add noise to the action for exploring
        # action_n += action_noise * trainer.random_process.sample()
        # action_n = np.clip(action_n, 0.0, 1.0)

        new_obs_n, reward_n, done_n, info_n = env.step(action_n)

        occupy[-1] += info_n[0]
        # print(info_n[0])
        # print("reward_n", reward_n)
        
        episode_step += 1
        train_step += 1
        done = all(done_n)
        terminal = (episode_step >= arglist.max_episode_len)

        # collect experience
        trainer.memory.push(obs_n, action_n, reward_n, new_obs_n, C.data.numpy())
        obs_n = new_obs_n
        # print("reward_n",reward_n)
        delay_n, energy_n = trainer.calc_delayAndenergy(C, total_dist)
        delay[-1] += delay_n
        energy[-1] += energy_n
        for i, rew in enumerate(reward_n):
            episode_rewards[-1] += rew
            agent_rewards[i][-1] += rew

        # print(episode_rewards)
        # print(len(episode_rewards))
        if done or terminal:
            # TODO: not to train attention unit
            if len(episode_rewards) % 10 == 0:
                trainer.update_attention_unit()
            reward.append(episode_rewards[-1]/(arglist.max_episode_len*4))
            print("steps: {}, episodes: {}, mean_episode_reward: {}, time: {}".format(
                train_step, len(episode_rewards), episode_rewards[-1]/(arglist.max_episode_len*nagents), round(time.time() - time_start, 3)))
            time_start = time.time()
            obs_n = env.reset(mode = 1) #
            episode_step = 0
            episode_rewards.append(0)
            delay.append(0)
            print(occupy)
            occupy.append(0)
            energy.append(0)
            listC = []
            for z in range(nagents):
                listC.append(0)
            record_C.append(listC)
            # print(len(record_C))
            # print(record_C)
            for a in agent_rewards:
                a.append(0)

        # for displaying learned policies
        if arglist.display:
            time.sleep(0.1)
            env.render()
            continue

        # update trainer every step, if not in display mode
        loss = None
        if len(trainer.memory) >= arglist.warmup_size and (train_step % 50 == 0):
            loss = trainer.update_parameters()



        # save model and display training output
        #  and (len(episode_rewards) % arglist.save_rate == 0)
        # if terminal:
            # trainer.save_model(arglist.exp_name, suffix=str(len(episode_rewards)//arglist.save_rate))
            # print("steps: {}, episodes: {}, mean_episode_reward: {}, time: {}".format(
            #     train_step, len(episode_rewards), episode_rewards, round(time.time() - time_start, 3)))
            # time_start = time.time()
            # final_save_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
        #     np.mean(episode_rewards[-arglist.save_rate-1:-1])

        # save final episodes rewards for plotting training results
        if len(episode_rewards) > arglist.num_episodes:
            # reward_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
            # with open(reward_file_name, 'wb') as fp:
            #     pickle.dump(final_save_rewards, fp)
            plt.figure(5)
            plt.title("rewards")
            plt.plot(episode_rewards, ls="-", lw=2, label="plot figure")
            plt.legend()

            plt.show()
            # workbook = xlwt.Workbook(encoding='utf-8')
            # # 创建一个worksheet
            # worksheet = workbook.add_sheet('My Worksheet')
            # worksheet1 = workbook.add_sheet('My C')
            #
            # # 写入excel
            # # 参数对应 行, 列, 值
            # for value in range(len(episode_rewards)):
            #     worksheet.write(value, 0, label=episode_rewards[value])
            # for value in range(len(delay)):
            #     worksheet.write(value, 1, label=delay[value])
            # for value in range(len(energy)):
            #     worksheet.write(value, 2, label=energy[value])
            # worksheet.write(0, 3, label=arglist.m)
            #
            #
            #
            # for value_x in range(len(record_C)):
            #     for value_y in range(len(record_C[value_x])):
            #         worksheet1.write(value_x, value_y, label=record_C[value_x][value_y])
            # workbook.save('Excel_test.xls')



            # result.append(arglist.m)
            data1 = pd.DataFrame(episode_rewards)
            data2 = pd.DataFrame(delay)
            data3 = pd.DataFrame(energy)
            data4 = pd.DataFrame(record_C)
            data6 = pd.DataFrame(occupy)
            m = []
            m.append(arglist.m)
            data5 = pd.DataFrame(m)
            writer = pd.ExcelWriter('data.xlsx')
            data1.to_excel(writer, 'episode_rewards')
            data2.to_excel(writer, 'delay')
            data3.to_excel(writer, 'energy')
            data4.to_excel(writer, 'record_C')
            data5.to_excel(writer, 'arglist.m')
            data6.to_excel(writer,'occupy')
            writer.save()
            # writer.close()


            # print(dataframe)


            print("Finish total of {} episodes.".format(len(episode_rewards)))
            print(record_C[-10:])
            # plt.figure(6)
            # plt.title("reward")
            # plt.plot(reward, ls="-", lw=2, label="plot figure")
            # plt.legend()
            #
            # plt.show()
            break


if __name__=="__main__":
    arglist = parse_args()
    train(arglist)