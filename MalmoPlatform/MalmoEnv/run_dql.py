# ------------------------------------------------------------------------------------------------
# Copyright (c) 2018 Microsoft Corporation
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ------------------------------------------------------------------------------------------------

import os
import malmoenv
import argparse
from pathlib import Path
import time
from PIL import Image
import yaml
import torch

from agent import Agent
from models.model import DQN

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='malmovnv test')
    parser.add_argument('--mission', type=str, default='missions/mobchase_single_agent.xml', help='the mission xml')
    parser.add_argument('--port', type=int, default=9000, help='the mission server port')
    parser.add_argument('--server', type=str, default='127.0.0.1', help='the mission server DNS or IP address')
    parser.add_argument('--port2', type=int, default=9000, help="(Multi-agent) role N's mission port")
    parser.add_argument('--server2', type=str, default=None, help="(Multi-agent) role N's server DNS or IP")
    parser.add_argument('--episodes', type=int, default=1, help='the number of resets to perform - default is 1')
    parser.add_argument('--episode', type=int, default=0, help='the start episode - default is 0')
    parser.add_argument('--role', type=int, default=0, help='the agent role - defaults to 0')
    parser.add_argument('--episodemaxsteps', type=int, default=0, help='max number of steps per episode')
    parser.add_argument('--saveimagesteps', type=int, default=0, help='save an image every N steps')
    parser.add_argument('--resync', type=int, default=0, help='exit and re-sync every N resets'
                                                              ' - default is 0 meaning never.')
    parser.add_argument('--experimentUniqueId', type=str, default='test1', help="the experiment's unique id.")
    args = parser.parse_args()
    if args.server2 is None:
        args.server2 = args.server

    xml = Path(args.mission).read_text()
    env = malmoenv.make()

    env.init(xml, args.port,
             server=args.server,
             server2=args.server2, port2=args.port2,
             role=args.role,
             exp_uid=args.experimentUniqueId,
             episode=args.episode, resync=args.resync)
    print(env.action_space)
    import pdb; pdb.set_trace()

    if os.path.exists("config.yaml"):
        # If config.yml is provided, always use that.
        #config = yaml.load(open("config.yaml"))
        config = yaml.safe_load(open("config.yaml"))
    elif os.path.exists("config.yml.in"):
        # If config.yml.in is provided, use it as defaults with CLI
        # overrides.
        config = yaml.load(open("config.yml.in"))
        assert isinstance(config, dict), config
        p = argparse.ArgumentParser()
        for name, default in sorted(config.items()):
            p.add_argument("--%s" % name, default=default, type=type(default))
            args = p.parse_args()
            config.update(dict(args._get_kwargs()))
    else:
        print(os.getcwd())
        print(os.listdir())
        assert False, "missing config: expected config.yaml or config.yml.in"

    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(THIS_FOLDER, 'weights')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DQN
    #load_model=True
    load_model=False
    if load_model:
        weights = torch.load(path)
        #model.load_state_dict(torch.load(path))
        model.load_state_dict(weights)
        model.eval()
    
    agent = Agent(env, config, model, device)
    agent.compile()
    agent.fit(path,verbose=True)
    #agent.play(verbose=True,sleep=0.1,max_steps=100)

    #for i in range(args.episodes):
    #    break
    #    print("reset " + str(i))
    #    obs = env.reset()

    #    steps = 0
    #    done = False
    #    while not done and (args.episodemaxsteps <= 0 or steps < args.episodemaxsteps):
    #        action = env.action_space.sample()
    #        import pdb; pdb.set_trace()

    #        obs, reward, done, info = env.step(action)
    #        steps += 1
    #        print("reward: " + str(reward))
    #        # print("done: " + str(done))
    #        print("obs: " + str(obs))
    #        # print("info" + info)
    #        if args.saveimagesteps > 0 and steps % args.saveimagesteps == 0:
    #            h, w, d = env.observation_space.shape
    #            img = Image.fromarray(obs.reshape(h, w, d))
    #            img.save('image' + str(args.role) + '_' + str(steps) + '.png')

    #        time.sleep(.05)

    env.close()
