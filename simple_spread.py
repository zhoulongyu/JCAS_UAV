import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import math


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 30
        num_target = 30
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.05
        # add target
        world.target = [Landmark() for i in range(num_target)]
        for i, landmark in enumerate(world.target):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world,mode = 0):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for target
        for i, landmark in enumerate(world.target):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        if mode == 1:
            return 0

        num_target = len(world.target)
        local_size = math.ceil(np.sqrt(num_target))
        local_x = np.linspace(-1, 1, local_size + 2)
        local = np.zeros((local_size * local_size, 2))
        index = 0
        for i in range(local_size):
            for j in range(local_size):
                local[index][0] = local_x[i + 1]
                local[index][1] = local_x[j + 1]
                index += 1

        # print(local)
        for i, landmark in enumerate(world.target):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            # landmark.state.p_pos = local[i]  #上面行注释，这行打开就是均匀分布
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, world):

        occupied_target = 0
        min_dists = 0
        for l in world.target:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            if min(dists) < 0.4:
                occupied_target += 1

        return occupied_target
        # rew = 0
        # collisions = 0
        # occupied_target = 0
        # min_dists = 0
        # for l in world.target:
        #     dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
        #     min_dists += min(dists)
        #     rew -= min(dists)
        #     if min(dists) < 0.05:
        #         occupied_target += 1
        # if agent.collide:
        #     for a in world.agents:
        #         if self.is_collision(a, agent):
        #             rew -= 1
        #             collisions += 1
        # return (rew, collisions, min_dists, occupied_target)


    def is_collision(self, agent1, agent2):
        if (agent1.name == agent2.name):
            return False
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        # for l in world.target:
        #     dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
        #     rew -= min(dists)
        dists = [np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos))) for l in world.target]

        rew -= min(dists)
        # rew += 1/(2*min(dists))

        # print(rew)
        # if min(dists)< 0.1:
        #     rew += 5
        #     if min(dists) < 0.05:
        #         rew += 5
            # print("rew",rew)
        # occupy = np.zeros(len(world.target))
        # agentindex = -1
        # for a in world.agents:
        #     dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for l in world.target]
        #     index = np.argsort(dists, axis=-1)
        #     if a == agent and dists[index[0]] < 0.1:
        #         agentindex = index[0]
        #     if (dists[index[0]] < 0.1):
        #         occupy[index[0]] += 1
        #
        # if (agentindex != -1):
        #     if (occupy[agentindex] == 1):
        #         rew += 10
        #     else:
        #         rew += 10 / (occupy[agentindex] * 5)
        #
        # temp = 1

        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    # rew -= 1*temp
                    # temp += 1
        # print("agent:",agent.name,"rew",rew)
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.target:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        # print("entity_pos",entity_pos)
        entity_color = []
        for entity in world.target:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        # print("entity_color", entity_color)
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        # print("comm", comm)
        # print("other_pos", other_pos)
        # print(np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm))
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + comm + other_pos)
