import asyncio
import logging
from multiprocessing import Pipe
from select import select
from threading import Thread

import gym
from gym.spaces import Box

from . import run_game, maps, Race
from .player import Bot, AbstractPlayer
from .Sc2Bot import Sc2Bot

logger = logging.getLogger(__name__)

class Sc2Env(gym.Env):
    def __init__(self, map, bot_race, get_opponents,
            initializer, observer, actuator, **kwargs):
        self.map = maps.get(map)
        self.bot_race = bot_race
        self.get_opponents = get_opponents
        self.initializer = initializer
        self.observer = observer
        self.actuator = actuator
        self.kwargs = kwargs

        self.env_pipe, self.runner_pipe = None, None
        self.thread = None

        self.reset()

    def _run_game(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        players = [
            Bot(
                Race[self.bot_race], 
                Sc2Bot(self.runner_pipe, self.initializer, self.observer, self.actuator)),
            *self.get_opponents()]
        assert any(isinstance(player, AbstractPlayer) for player in players)

        run_game(self.map, players, realtime=False, **self.kwargs)

    def reset(self):
        if self.thread and self.env_pipe:
            self.env_pipe.send(None)
            self.thread.join()
        
        self.env_pipe, self.runner_pipe = Pipe(True)
        self.thread = Thread(target=self._run_game).start()
        
        logger.info("waiting Sc2Bot spaces")
        select([self.env_pipe], [], [], 60)
        self.action_space, self.observation_space = self.env_pipe.recv()

        logger.debug("waiting Sc2Bot observation")
        select([self.env_pipe], [], [], 60)
        self.observation, self.reward = self.env_pipe.recv()

        return self.observation

    def step(self, action):
        self.env_pipe.send(action)

        logger.debug("waiting Sc2Env observation")
        select([self.env_pipe], [], [])
        msg = self.env_pipe.recv()

        done = not bool(msg)
        if not done:
            self.observation, self.reward =  msg

        return self.observation, self.reward, done, {}
