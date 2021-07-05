import retro
import numpy as np
import cv2
import neat
import pickle
import multiprocessing
import os
import visualize


class Worker(object):
    def __init__(self, genome, config):
        self.genome = genome
        self.config = config

    def work(self):

        self.env = retro.make('SonicTheHedgehog-Genesis', 'GreenHillZone.Act1')

        self.env.reset()

        ob, _, _, _ = self.env.step(self.env.action_space.sample())

        inx = int(ob.shape[0]/8)
        iny = int(ob.shape[1]/8)
        done = False

        net = neat.nn.FeedForwardNetwork.create(self.genome, self.config)

        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        counter = 0
        xpos = 0
        xpos_max = 0
        imgarray = []

        while not done:
            # self.env.render()
            frame += 1
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))

            #imgarray = np.ndarray.flatten(ob)
            #imgarray = np.interp(imgarray, (0, 254), (-1, +1))
            #actions = net.activate(imgarray)

            for x in ob:
                for y in x:
                    imgarray.append(y)

            actions = net.activate(imgarray)

            ob, rew, done, info = self.env.step(actions)
            imgarray.clear()

            xpos = info['x']

            if xpos > xpos_max:
                xpos_max = xpos
                fitness_current += 1

            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1

            if xpos == info['screen_x_end'] and xpos > 500:
                fitness_current += 100000
                done = True

            if counter > 250:
                done = True

        print(fitness_current)
        return fitness_current


def eval_genomes(genome, config):

    worky = Worker(genome, config)
    return worky.work()


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)
    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-39')

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(10))

    pe = neat.ParallelEvaluator(6, eval_genomes)
    winner = p.run(pe.evaluate)

    with open('winner.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)

    #visualize.draw_net(config, winner, True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)
