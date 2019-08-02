import logging
logging.basicConfig(level=logging.DEBUG)

from gym.spaces import Box
from numpy.random import randn
from sc2Env import Sc2Env, Race
from sc2Env.player import Computer

# Bot collecting observations against an easy Zerg computer

def get_opponents():
    # return a list of opponent for our bot
    return [Computer(Race['Zerg'])]

def initializer(botAI):
    # initialise the spaces, can access the map information from botAI and more..
    botAI.action_space = Box(-1, 1, shape=(3,3))
    botAI.observation_space = Box(-1, 1, shape=(2,2))

async def observer(botAI):
    # extract information from bot ai and forward return a numpy array
    # matching the observation space and a reward
    observation = randn(2,2)
    reward = 0
    return observation, reward

async def actuator(botAI, action):
    # extract information from the action numpy array to give order through
    # the botAI instance
    # see python-sc2 docs !
    # if await self.can_cast(unit, ability_id):
    #   await botAI.do_actions(todo_actions)
    pass

env = Sc2Env('KingsCoveLE', 'Zerg', get_opponents, 
    initializer, observer, actuator, game_time_limit=60)

done = False

for i in range(3):
    observation = env.reset()
    while not done:
        observation, reward, done, _ = env.step(randn(3,3))

env.close()
print('END')
