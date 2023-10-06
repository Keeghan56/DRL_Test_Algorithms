import numpy as np 
from utils import clip_reward

class EpisodeLoop:
    def __init__(self, agent, env, load_checkpoint=False, clip_reward=False):
        self.agent = agent
        self.env = env
        self.load_checkpoint = load_checkpoint
        self.clip_reward = clip_reward

    def run(self, n_episodes=1):
        if self.load_checkpoint:
            self.agent.load_models()
        n_steps = 0 
        best_score = self.env.reward_range[0]
        scores, steps = [],[]

        for i in range(n_episodes):
            done = False
            observation = self.env.reset()
            score = 0
            while not done:
                action = self.agent.choose_action(observation)
                observation_, reward, done, info = self.env.step(action)
                score += reward
                r = clip_reward(reward) if self.clip_reward else reward
                if not self.load_checkpoint:
                    self.agent.store_transition(observation, action, r, observation_, done)
                    self.agent.update()
                observation = observation_
                n_steps += 1
            scores.append(score)
            steps.append(n_steps)

            avg_score = np.mean(scores[-100:])
            print('episode {} ep score {:.1f} avg score {:.1f} n steps {}'.format(i,score,avg_score,n_steps))

            if avg_score > best_score:
                if not self.load_checkpoint:
                    self.agent.save_models()
                best_score = avg_score
            self.handle_extra_functionality()

        return scores, steps
    
    def handle_extra_functionality(self):
        for func in self.functions:
            continue

