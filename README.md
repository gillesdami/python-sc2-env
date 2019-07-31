# A StarCraft II bot gym env library over python-sc2

This library provide [python-sc2](https://github.com/Dentosal/python-sc2) as a [gym](https://github.com/openai/gym) environment.
So you benefits both from the sc2.BotAI and gym.Env classes to train your bot using existing algorithms.

**This library (currently) covers only the raw scripted interface.** At this time I don't intend to add support for graphics-based interfaces.

## Documentation

### class Sc2Env(gym.Env)

#### def __init__(self, map, bot_race, get_opponents, **kwargs):

- map `str` upper camel case name of an installed map (ex: KingsCoveLE)
- bot_race `str` race of the bot (ex: Zerg)
- get_opponents `Callable` the opponents of the bot
  - returns a `list` of `AbstractPlayer` (BotAI, Player...)
- initializer `Callable`
    Called on start, must set BotAI.action_space and BotAI.observation_space.
    The defined spaces must inherit from gym.spaces.space.Space.

    example:

    '''python
    def initializer(bot):
        bot.action_space = gym.spaces.Box(5,10)
        bot.observation_space = gym.spaces.Box(3,11)
    '''
- observer: `async` `Callable`
    Called on step takes a `sc2.BotAI` instance, must return an observation and a reward.
    An observation is a `numpy.array` matching the observation space and a reward is `number`.
- actuator: `async` `Callable`
    Called on step, takes a `sc2.BotAI` instance and an action.
    An action is a `numpy.array` matching the action space.
    Used to run actions using in sc2.
- **kwargs `any` sc2.run_game extra arguments

More on sc2.BotAI [here](https://github.com/Dentosal/python-sc2/wiki).

## Installation

By installing this library you agree to be bound by the terms of the [AI and Machine Learning License](http://blzdistsc2-a.akamaihd.net/AI_AND_MACHINE_LEARNING_LICENSE.html).

You'll need Python 3.6 or newer.

```bash
pip3 install --user --upgrade sc2-env
```

You'll also need an StarCraft II executable. If you are running Windows or macOS, just install the normal SC2 from blizzard app. [The free starter edition works too.](https://us.battle.net/account/sc2/starter-edition/). Linux users get the best experience by installing the Windows version of StarCraft II with [Wine](https://www.winehq.org). Linux user can also use the [Linux binary](https://github.com/Blizzard/s2client-proto#downloads), but it's headless so you cannot actually see the game.

You probably want some maps too. Official map downloads are available from [Blizzard/s2client-proto](https://github.com/Blizzard/s2client-proto#downloads). Notice: the map files are to be extracted into *subdirectories* of the `install-dir/Maps` directory.

### Running

After installing the library, a StarCraft II executable, and some maps, you're ready to get started. Look at the dummy bot example!

```bash
python3 examples/dummy_bot.py
```

If you installed StarCraft II on Linux with Wine, set the `SC2PF` environment variable to `WineLinux`:

```bash
SC2PF=WineLinux python3 examples/dummy_bot.py
```
