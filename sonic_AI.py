import retro
import numpy as np
import cv2
import neat
import pickle
import visualize

env = retro.make("SonicTheHedgehog-Genesis", "GreenHillZone.Act1")

imgarray = []
xpos_end = 0


def eval_genomes(genomes, config):

    for genome_id, genome in genomes:
        #Observation is image
        ob = env.reset()
        ac = env.action_space.sample()

        inx, iny, inc = env.observation_space.shape

        inx = int(inx/8)
        iny = int(iny/8)

        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        counter = 0
        xpos = 0
        xpos_max = 0

        done = False
        #cv2.namedWindow("main", cv2.WINDOW_NORMAL)

        while not done:

            # env.render()
            frame += 1

            # Visualisation of what network sees
            scaledimg = cv2.cvtColor(ob, cv2.COLOR_BGR2RGB)
            scaledimg = cv2.resize(scaledimg, (iny, inx))

            # Downsizing screenshot and making it grayscale
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))

            # Create window for visualisation
            # cv2.imshow('main', scaledimg)
            # cv2.waitKey(1)

            # Compress 2d image into single array
            imgarray = np.ndarray.flatten(ob)

            nnOutput = net.activate(imgarray)
            ob, rew, done, info = env.step(nnOutput)

            # Activates done condition if goal not met e.g. just standing
            xpos = info["x"]
            xpos_end = info["screen_x_end"]

            if xpos > xpos_max:
                fitness_current += 1
                xpos_max = xpos

            if xpos == xpos_end and xpos > 600:
                fitness_current += 100000  # If network reaches the end of the level, set max fitness
                done = True

            # fitness_current += rew # Makes network more generic

            # If better max fitness achieved counter reset so done not satisfied
            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1

            if done or counter == 250:
                done = True
                print(genome_id, fitness_current)

            genome.fitness = fitness_current


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')


p = neat.Population(config)

# Statistics
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
# p.add_reporter(neat.Checkpointer(10))

winner = p.run(eval_genomes, 1)

# If the network is done, save network
with open("winner.pkl", "wb") as output:
    pickle.dump(winner, output, 1)

#visualize.draw_net(config, winner, True)
