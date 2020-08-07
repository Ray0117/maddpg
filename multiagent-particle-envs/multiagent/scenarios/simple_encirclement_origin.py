import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import random
import pyglet

class Scenario(BaseScenario):

    def __init__(self):
        # some desirable control references
        self.desire_rho = 0.25
        self.desire_w = 2
        self.goal_bound = 0.6

        # the obseration for actor and critic
        self.angle2agent = np.array([0,0])
        self.polar_vel = np.array([0,0])

        # the error between actual and desire 
        self.pos_error = np.sqrt(2)*(1+self.goal_bound) - self.desire_rho  # max for initial
        self.phi_error = 2*np.pi
        self.w_error = self.desire_w # np.fabs(-agent.max_speed - self.desire_w)
        self.count_collision = 0
        self.is_collide_avoid = True
        self.group = []

    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 5
        world.num_agents = num_agents
        num_adversaries = 0
        num_landmarks = 1
        self.desire_angle_spacing = 2*np.pi/num_agents

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.name = 'agent %d' % i if not agent.adversary else 'adv' 
            # agent.color = np.array([0, 1, 0])
            agent.size = 0.05 
            agent.max_speed = 0.6 if agent.adversary else 1
            if not agent.adversary:
                self.group.append(agent)

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.color = np.array([0.25, 0.25, 0.25])
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85, 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])

        # set goal landmark
        goal = world.landmarks[0]
        goal.color = np.array([0.25, 0.25, 0.25])
        goal.size = 0.055        
        goal.state.p_pos = np.random.uniform(-self.goal_bound, +self.goal_bound, world.dim_p)
        for i, landmark in enumerate(world.landmarks):
            if landmark != goal:
                landmark.size = np.random.uniform(0.05,0.07)
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                while np.sqrt(np.sum(np.square(landmark.state.p_pos - goal.state.p_pos))) < landmark.size + 3*goal.size:
                    landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)            
            landmark.state.p_vel = np.zeros(world.dim_p)

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.goal_a = goal

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            return np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
        else:
            dists = []
            for l in world.landmarks:
                dists.append(np.sum(np.square(agent.state.p_pos - l.state.p_pos)))
            dists.append(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos)))
            return tuple(dists)

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        dis2goal = np.array([np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))),0])

        goal_vel = agent.goal_a.state.p_vel
        # print('dis2goal: ',np.around(dis2goal,3),'pole_vel: ',np.around(self.polar_vel,3),'angle2agent: ',np.around(self.angle2agent,3))
        # print('error:  pos: ',np.around(self.pos_error,3),'angle2agent: ',np.around(self.phi_error,3),'pole_vel: ',np.around(self.w_error,3))

        obs_all = []
        obs_all = entity_pos + other_pos
        obs_all.append(dis2goal)
        obs_all.append(goal_vel)
        obs_all.append(self.polar_vel)
        obs_all.append(self.angle2agent)

        # print("obs_len: {} concat_obs: {}  obs: {}".format(len(np.concatenate(obs_all)),np.concatenate(obs_all),obs_all))
        return np.concatenate(obs_all)

    # 得到实际值与控制指标的误差，保留2位小数
    def geError_all(self, agent):
        error_all = np.zeros([1,4])
        error_all = np.around([self.pos_error, self.phi_error,self.w_error, self.count_collision],4)
        return error_all

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        return self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)

    def adversary_reward(self, agent, world):
        # Rewarded based on proximity to the goal landmark
        shaped_reward = True
        if shaped_reward:  # distance-based reward
            return -np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
        else:  # proximity-based reward (binary)
            adv_rew = 0
            if np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))) < 2 * agent.goal_a.size:
                adv_rew += 5
            return adv_rew

    def agent_reward(self, agent, world):
        # Rewarded based on how close any good agent is to the goal landmark, and how far the adversary is from it
        shaped_reward = True
        shaped_adv_reward = True

        # Calculate negative reward for adversary
        adversary_agents = self.adversaries(world)
        if shaped_adv_reward:  # distance-based adversary reward
            adv_rew = sum([np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in adversary_agents])
        else:  # proximity-based adversary reward (binary)
            adv_rew = 0
            for a in adversary_agents:
                if np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) < 2 * a.goal_a.size:
                    adv_rew -= 5

        # Calculate positive reward for agents
        good_agents = self.good_agents(world)

        ## agents are penalized for collision
        collide_rew = self.getcollision_rew(agent, world)

        if shaped_reward:  # distance-based agent reward
            # pos_rew = -sum([np.fabs(np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) - desire_rho) for a in good_agents])
            pos_rew = self.getPos_rew(agent)

        else:  # proximity-based agent reward (binary)
            pos_rew = 0
            if min([np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in good_agents]) \
                    < 2 * agent.goal_a.size:
                pos_rew += 5
            pos_rew -= min(
                [np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in good_agents])

        # agents are penalized for too far or too close respected to the desire spacing
        spacing_rew = self.getSpacing_rew(agent,world)

        # agents are reward if they are encircle with desire angle velocity
        angle_velocity_rew = self.getAngleVelocity_rew(agent)
    
        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        out_screen_rew = self.out_screen_rew(agent, world)
        
        # if np.fabs(np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))) - self.desire_rho) > 2*self.desire_rho:
        #     k = [5, 0, 0, 7, 5]
        # else:
        #     k = [5, 7, 20, 7, 3]
        # print('pos_rew ',np.around(pos_rew,2),'spacing_rew ',np.around(spacing_rew,2),\
            # 'angVel_rew ',np.around(angle_velocity_rew,2),'collide_rew ',np.around(collide_rew,2),'out_rew ',np.around(out_screen_rew,2))
        return collide_rew + pos_rew + spacing_rew + angle_velocity_rew + 0.5*out_screen_rew

    " calculate reward for several control references. Those are actually 'Penalty' function of error. "
    # 检测是否有碰撞
    def getcollision_rew(self, agent, world):
        if agent.collide:
            self.count_collision = 0
            for a in world.entities:
                if agent is a : continue
                safe_dis = 3*agent.size + a.size
                dist = np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos)))
                if a != agent.goal_a:
                    if dist < safe_dis : 
                        # k = np.polyfit([dist_min,safe_dis],[-20,0],1)
                        # rew = k[0]*dist + k[1]
                        self.count_collision += 1 
                else:
                    safe_dis = 2*agent.size + a.size
                    if dist < safe_dis :
                        # k = np.polyfit([dist_min,safe_dis],[-20,0],1)
                        # rew = k[0]*dist + k[1]
                        self.count_collision += 1

            self.is_collide_avoid = False if self.count_collision > 1 else True
        # normalization for 'collide_count' bound(might be):（0, 2）, nomalized to:(0,1)
        # return -20*self.count_collision
        return -10*self.count_collision

    def getPos_rew(self,agent):
        self.pos_error = np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))) - self.desire_rho
        # print('pos_error:', self.pos_error)
        # normalization for 'pos_error': bound:（0,sqrt(2)*2）, nomalized to:(0,1)
        # max_pos_error = np.sqrt(2)*(1+self.goal_bound) - self.desire_rho
        # k = np.polyfit([0,2.2],[10,0],1)
        # normalized_pos_error
        return -4.55*np.fabs(self.pos_error) + 10 if self.is_collide_avoid else 0

    def getSpacing_rew(self,agent, world):
        # # 计算agent与前后两个agents之间的角度距离，应该等间距
        # good_agents = self.good_agents(world)
        # agent_pos = []
        # agent_angle = []
        # # print(agent.state.p_vel)

        # for i, a in enumerate(good_agents):
        #     agent_pos.append(a.state.p_pos - a.goal_a.state.p_pos)
        #     if np.arctan2(agent_pos[len(agent_pos)-1][1],agent_pos[len(agent_pos)-1][0]) >= 0:
        #         agent_angle.append(np.arctan2(agent_pos[len(agent_pos)-1][1],agent_pos[len(agent_pos)-1][0]))
        #     else:
        #         agent_angle.append(np.arctan2(agent_pos[len(agent_pos)-1][1],agent_pos[len(agent_pos)-1][0]) + 2*np.pi)
        # # print('angle0 ',agent_angle[0]*180/np.pi,'angle1 ',agent_angle[1]*180/np.pi,'angle2 ',agent_angle[2]*180/np.pi)
        
        # for i in range(len(agent_angle)):
        #     if agent.name == 'agent %d' % i:  # I am the agent NO.i in angle list
        #         myID = i
        #         plusID = 0 if i==len(agent_angle)-1 else i+1
        #         miusID = len(agent_angle)-1 if i == 0 else i-1
        #         myPhi = agent_angle[myID]
        #         plusPhi = agent_angle[plusID]
        #         miusPhi = agent_angle[miusID]
        #     else:
        #         continue
        # print('miusPhi ',miusPhi*180/np.pi,'myPhi ',myPhi*180/np.pi,'plusPhi ',plusPhi*180/np.pi)

         # 计算agent与前后两个agents之间的角度距离
        myIndex = self.group.index(agent) # I am the agent NO.i in angle list
        plusIndex = 0 if myIndex==len(self.group)-1 else myIndex+1
        miusIndex = len(self.group)-1 if myIndex==0 else myIndex-1
        # print('agent: ',self.group1[miusIndex].name,self.group1[myIndex].name,self.group1[plusIndex].name)
        myPos = self.group[myIndex].state.p_pos - self.group[myIndex].goal_a.state.p_pos
        plusPos = self.group[plusIndex].state.p_pos - self.group[plusIndex].goal_a.state.p_pos
        miusPos = self.group[miusIndex].state.p_pos - self.group[miusIndex].goal_a.state.p_pos

        myAngle = np.arctan2(myPos[1],myPos[0])
        plusAngle = np.arctan2(plusPos[1],plusPos[0])
        miusAngle = np.arctan2(miusPos[1],miusPos[0])

        myPhi = myAngle if myAngle >= 0 else myAngle + 2*np.pi
        plusPhi = plusAngle if plusAngle >= 0 else plusAngle + 2*np.pi
        miusPhi = miusAngle if miusAngle >= 0 else miusAngle + 2*np.pi

        self.myPhi = np.array([myPhi,0])
        self.otherPhi = np.array([miusPhi,plusPhi])
        # print('miusPhi ',miusPhi*180/np.pi,'myPhi ',myPhi*180/np.pi,'plusPhi ',plusPhi*180/np.pi)

        # 等间距 分布:a =0.5,b=0.5
        a = 0.5
        b = 0.5
        if myPhi > plusPhi :  # means I'm the last agent
            phi_bar = a * miusPhi + b * plusPhi + b * 2 * np.pi
            self.angle2agent = np.array([myPhi - miusPhi, plusPhi + 2*np.pi - myPhi])
        elif myPhi < miusPhi : # it means I am the first robot
            phi_bar = a * miusPhi + b * plusPhi - a * 2 * np.pi
            self.angle2agent = np.array([myPhi + 2*np.pi  - miusPhi, plusPhi - myPhi])
        else:
            phi_bar = a * miusPhi + b * plusPhi
            self.angle2agent = np.array([myPhi - miusPhi, plusPhi - myPhi])
        
        # phi_bar = myID*self.desire_angle_spacing + episode_step*world.dt*self.desire_w
        # print("{} step: {} phi_bar: {} myPhi {} error {}".format(agent.name, episode_step, phi_bar, myPhi,myPhi-phi_bar))
        # normalization for 'abs(phi_bar - myPhi)': bound:（0,2*np.pi）, nomalized to:(0,1)
        self.phi_error = phi_bar - myPhi
        # k = np.polyfit([0,2*np.pi],[15,0],1)
        return -2.38*np.fabs(self.phi_error) + 15 if np.fabs(self.pos_error) < 0.04*self.desire_rho else 0

    # 计算agent的角速度，得到与期望角速度的差
    def getAngleVelocity_rew(self, agent):
        vec_agent2goal = agent.goal_a.state.p_pos - agent.state.p_pos
        vec_vel = agent.state.p_vel
        normal_vel = np.dot(vec_agent2goal,vec_vel)/np.sqrt(np.dot(vec_agent2goal,vec_agent2goal))  #法向速度大小,期望=0
        # print('vec_cel: ',vec_vel,'normal_vel: ',normal_vel)
        if np.dot(vec_vel,vec_vel) - np.square(normal_vel) < 0:
            print('error!!! vec_agent2goal ',np.around(vec_agent2goal,2), \
            'vec_vel ',np.around(vec_vel,2),'normal_vel ',np.around(normal_vel,2))
            # error!!! vec_agent2goal  [0.01 0.1 ] vec_vel  [-0.03 -0.18] normal_vel  -0.18
            tangen_vel = 0 #切向速度大小,期望=desire_w
        else:
            tangen_vel = np.sqrt(np.dot(vec_vel,vec_vel) - np.square(normal_vel)) #切向速度大小,期望=desire_w

        self.polar_vel = np.array([tangen_vel,normal_vel])

        vec_goal2velStart = -1*vec_agent2goal  # 目标到agent的向量
        vec_goal2velEnd = agent.state.p_vel+agent.state.p_pos - agent.goal_a.state.p_pos # 目标到agent速度末端的向量
        flip = 1 if np.cross(vec_goal2velStart,vec_goal2velEnd) >= 0 else -1
        # alpha1 = np.arctan2(vec_goal2velStart[1],vec_goal2velStart[0]) 
        # alpha2 = np.arctan2(vec_goal2velEnd[1],vec_goal2velEnd[0])

        # print('agent_pos ',agent.state.p_pos,'agent_vel ',agent.state.p_vel,'goal_pos ',agent.goal_a.state.p_pos)
        # print('cos ',cos_theta,'sin ',sin_theta,'tan_v ',tangen_vel,'nom_v ',normal_vel)
        # print('alpha1 ',alpha1,'alpha2 ',alpha2,'flip ',flip)

        self.w_error = flip*tangen_vel / self.desire_rho - self.desire_w
        # print(agent.name,'omega', flip*tangen_vel/ self.desire_rho)

        # # normalization for 'w_error': bound:（0,1.5）, nomalized to:(0,1)
        # max_tangen_v_error = np.fabs(-agent.max_speed - self.desire_w)
        # max w_error = self.desire_w = 1 
        # k = np.polyfit([0,2],[30,0],1)
        return -15*np.fabs(self.w_error) + 30 if np.fabs(self.phi_error) < np.pi/12 else 0

    # 检测agent是否运动脱离视野
    def out_screen_rew(self,agent, world):
        rtvl = 0
        for p in range(world.dim_p):
            z = abs(agent.state.p_pos[p])
            if z < 0.5:
                rtvl += 20
            else:
                rtvl += -40*z+40
            # else:
            #     rtvl += np.exp(z*3)
                # rtvl += min(np.exp(z*4), 50)
        
        # print(rtvl)
        # normalization: max(rtvl) = 10, nomalized to:(0,1)
        return rtvl
