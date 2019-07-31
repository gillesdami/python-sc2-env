import logging
from select import select

from gym.spaces import Space

from .bot_ai import BotAI

logger = logging.getLogger(__name__)

class Sc2Bot(BotAI):
    def __init__(self, runner_pipe, initializer, observer, actuator):
        '''
        Sc2 bot launched in an other thread by Sc2Env.

        params:
        - initializer: fn
            Called on start, must set BotAI.action_space and 
            BotAI.observation_space.
            The defined spaces must inherit from gym.spaces.space.Space.
            example:
                def initializer(bot):
                    bot.action_space = gym.spaces.Box(5,10)
                    bot.observation_space = gym.spaces.Box(3,11)

        - observer: async fn
            Called on step, takes a python-sc2 bot instance, must return 
            an observation and a reward.
            An observation is a numpy array matching the observation space
            and a reward is number.
        - actuator: async fn
            Called on step, takes a python-sc2 bot instance and an action.
            An action is a numpy array matching the action space.
            Used to run actions using on python-sc2.
        '''
        self.runner_pipe = runner_pipe

        self.initalizer = initializer
        self.observer = observer
        self.actuator = actuator

    def on_start(self):
        self.initalizer(self)

        assert hasattr(self, 'action_space') and hasattr(self, 'observation_space')
        assert isinstance(self.action_space, Space) and isinstance(self.observation_space, Space)

        self.runner_pipe.send([self.action_space, self.observation_space])

    async def on_step(self, iteration: int):
        self.runner_pipe.send(await self.observer(self))
        
        logger.debug('waiting Sc2Env action')
        select([self.runner_pipe], [], [], 60)
        action = self.runner_pipe.recv()

        if action is None:
            raise Exception('End of training')

        await self.actuator(self, action)
    
    def on_end(self, result):
        self.runner_pipe.send(None)
