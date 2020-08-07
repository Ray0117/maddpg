import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
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
        self.episode_num = 0
        self.episode_rnd = 0
        self.agent_rnd   = 0
        self.vel_rnd     = [0., 0.]

    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_good_agents = 5 # UAV Team
        num_adversaries = 1 # Target
        num_agents = num_adversaries + num_good_agents
        num_landmarks = 0
        desire_angle_form = 2 * np.pi / num_good_agents
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.055 if agent.adversary else 0.05
            # agent.accel = 3.0 if agent.adversary else 4.0
            #agent.accel = 20.0 if agent.adversary else 25.0
            # agent.max_speed = 1.0 if agent.adversary else 1.3
            agent.max_speed = 1.3 if agent.adversary else 1 # adv is the target
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.075
            landmark.boundary = False
        # make initial conditions
        self.reset_world(world)
        return world


    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85, 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p) if not agent.adversary\
                else np.random.uniform(-self.goal_bound, +self.goal_bound, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.goal = world.agents[0] # 0 is the index of adv
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)
        self.episode_num = 0
        self.episode_rnd = np.random.randint(0, 30)
        self.agent_rnd   = np.random.randint(0, 5)
        self.vel_rnd     = np.random.uniform(-2, 2, world.dim_p)


    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            return 0
        else:
            dist = np.sum(np.square(agent.state.p_pos - agent.goal.state.p_pos))
            return dist

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        other_pos = []
        for other in world.agents:
            if other is agent: continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        dist2goal = np.array([np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal.state.p_pos))),0])
        goal_vel = agent.goal.state.p_vel

        obs_all = []
        obs_all = entity_pos + other_pos
        obs_all.append(dist2goal)
        obs_all.append(goal_vel)
        obs_all.append(self.polar_vel)
        obs_all.append(self.angle2agent)

        return np.concatenate(obs_all)
    
    # Get all error between desire and actual
    def geError_all(self, agent):
        error_all = np.zeros([1, 4])
        error_all = np.around([self.pos_error, self.phi_error, self.w_error, self.count_collision], 4)
        return error_all

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]


    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for escaping encirclement
        rew = 0
        shape = True
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)

        if shape:  # reward can optionally be shaped (increased reward for increased distance from agents)
                rew += 0.1 * sum([np.sqrt(np.sum(np.square(a.state.p_pos - adversaries[0].state.p_pos))) for a in agents])
        # agents are penalized for exiting the screen, so that they can be caught by the agents
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        if self.episode_num + self.episode_rnd < 70:
            self.episode_num += 1
            print(self.episode_num)
        else:
            print('Random Attack on agent %i' %self.agent_rnd,'vel=', self.vel_rnd)
            world.agents[self.agent_rnd+1].color -= np.array([0.25, 0.25, 0.25]) # the victim will change color
            agents[self.agent_rnd].state.p_vel = self.vel_rnd
        return rew

    def agent_reward(self, agent, world):

        # Agents are rewarded if encircle the adv
        rew = 0
        # agents are penalized for collision
        collide_rew = self.getcollision_rew(agent, world)
        # agents are rewarded for distance
        pos_rew = self.getPos_rew(agent)
        # agents are penalized for difference respected to desired spacing
        spacing_rew = self.getSpacing_rew(agent,world)
        # agents are reward for desire angle velocity
        angle_velocity_rew = self.getAngleVelocity_rew(agent)
        # agents are penalized for exiting the screen
        out_screen_rew = self.out_screen_rew(agent, world)

        rew = collide_rew + pos_rew + spacing_rew + angle_velocity_rew + 0.5*out_screen_rew

        return rew

    def getcollision_rew(self, agent, world):
        if agent.collide:
            self.count_collision = 0
            for a in world.entities: # collision avoidance with all other entities
                if agent is a: continue
                safe_dist = 3 * agent.size + a.size
                dist = np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos)))

                if a is not agent.goal:
                    if dist < safe_dist: self.count_collision += 1
                else: # decrease safe dist for encircle target
                    safe_dist = 2 * agent.size + a.size
                    if dist < safe_dist: self.count_collision += 1

            self.is_collide_avoid = False if self.count_collision > 1 else True
            
        return -10 * self.count_collision

    def getPos_rew(self, agent):

        self.pos_error = np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal.state.p_pos))) - self.desire_rho
        
        return -4.55 * np.fabs(self.pos_error) + 10 if self.is_collide_avoid else 0

    
    def getSpacing_rew(self, agent, world):

        agent_i = 0
        agents = self.good_agents(world)
        agent_num = len(agents)
        for i, a in enumerate(agents):
            if a is agent: agent_i = i            
        
        # calculate position difference
        pos_i   = agents[agent_i].state.p_pos - agents[agent_i].goal.state.p_pos
        pos_nxt = agents[(agent_i+1)%agent_num].state.p_pos - agents[(agent_i+1)%agent_num].goal.state.p_pos
        pos_lst = agents[(agent_i-1)%agent_num].state.p_pos - agents[(agent_i-1)%agent_num].goal.state.p_pos

        # calculate polar coordinator angle
        angle_i = np.arctan2(pos_i[1], pos_i[0])
        angle_nxt = np.arctan2(pos_nxt[1], pos_nxt[0])
        angle_lst = np.arctan2(pos_lst[1], pos_lst[0])

        # make sure phi is plus    
        phi_i = angle_i if angle_i >= 0 else angle_i + 2 * np.pi
        phi_nxt = angle_nxt if angle_nxt >= 0 else angle_nxt + 2 * np.pi
        phi_lst = angle_lst if angle_lst >= 0 else angle_lst + 2 * np.pi

        a = b = 0.5
        if phi_i > phi_nxt:   # i is the last one in team
            phi_bar = a * phi_lst + b * phi_nxt + b * 2 * np.pi
        elif phi_i < phi_lst: # i is the first one in team
            phi_bar = a * phi_lst + b * phi_nxt - a * 2 * np.pi
        else:
            phi_bar = a * phi_lst + b * phi_nxt
        
        self.phi_error = phi_bar - phi_i
        return -2.38*np.fabs(self.phi_error) + 15 if np.abs(self.pos_error) < 0.04 * self.desire_rho else 0

    def getAngleVelocity_rew(self, agent):

        AngleFlag = False 

        if AngleFlag is True:

            vec_agent2goal = agent.goal.state.p_pos - agent.state.p_pos
            vec_vel = agent.state.p_vel
            normal_vel = np.dot(vec_agent2goal,vec_vel)/np.sqrt(np.dot(vec_agent2goal,vec_agent2goal))  # normal speed, need to get 0
            if np.dot(vec_vel, vec_vel) - np.square(normal_vel) < 0:
                print('error!!! vec_agent2goal ',np.around(vec_agent2goal,2), \
                'vec_vel ',np.around(vec_vel,2),'normal_vel ',np.around(normal_vel,2))
                tangen_vel = 0
            else:
                tangen_vel = np.sqrt(np.dot(vec_vel,vec_vel) - np.square(normal_vel)) # need to get desire_w

            self.polar_vel = np.array([tangen_vel, normal_vel])

            vec_goal2velStart = -1*vec_agent2goal  # vec from goal to agent
            vec_goal2velEnd = agent.state.p_vel+agent.state.p_pos - agent.goal_a.state.p_pos # from goal to vec vel end
            flip =1 if np.cross(vec_goal2velStart, vec_goal2velEnd) >= 0 else -1

            self.w_error = flip * tangen_vel / self.desire_rho - self.desire_w

            return -15 * np.fabs(self.w_error) + 30 if np.fabs(self.phi_error) < np.pi/12 else 0

        else:
            return 0
    
    # make sure agents keep in screen
    def out_screen_rew(self, agent, world):
        rew = 0
        for p in range(world.dim_p):
            z = abs(agent.state.p_pos[p])
            if z < 0.5:
                rew += 20
            else:
                rew += -40*z+40
        return rew
