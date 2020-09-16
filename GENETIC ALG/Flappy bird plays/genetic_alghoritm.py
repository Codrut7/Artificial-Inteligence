import numpy as np
from copy import deepcopy
from agent import Agent
import flappy_game
import torch
import warnings
import os
import random
warnings.filterwarnings("ignore")

POPULATION = 32
CHILDREN_SCALE = 1
ITERATIONS = 5000
CHILDREN_NUMBER = POPULATION * CHILDREN_SCALE

def crossover(p1, p2, gamma=0):
    c1 = deepcopy(p1)
    c2 = deepcopy(p2)

    with torch.no_grad():

        # Crossover first layer
        parent1_1_layer_weight = p1["model"].layer1.weight
        parent2_1_layer_weight = p2["model"].layer1.weight
        parent1_1_layer_bias = p1["model"].layer1.bias
        parent2_1_layer_bias = p2["model"].layer1.bias

        alpha_weight = torch.from_numpy(np.random.randint(2, size=parent1_1_layer_weight.shape))
        alpha_bias = torch.from_numpy(np.random.randint(2, size=parent1_1_layer_bias.shape))

        c1["model"].layer1.weight.copy_(alpha_weight * parent1_1_layer_weight + (1-alpha_weight) * parent2_1_layer_weight)
        c2["model"].layer1.weight.copy_((1-alpha_weight) * parent1_1_layer_weight + alpha_weight * parent2_1_layer_weight)
        c1["model"].layer1.bias.copy_(alpha_bias * parent1_1_layer_bias + (1-alpha_bias) * parent2_1_layer_bias)
        c2["model"].layer1.bias.copy_((1-alpha_bias) * parent1_1_layer_bias + alpha_bias * parent2_1_layer_bias)

        # Crossover second layer
        parent1_2_layer_weight = p1["model"].layer2.weight
        parent2_2_layer_weight = p2["model"].layer2.weight
        parent1_2_layer_bias = p1["model"].layer2.bias
        parent2_2_layer_bias = p2["model"].layer2.bias

        alpha_weight = torch.from_numpy(np.random.randint(2, size=parent1_2_layer_weight.shape))
        alpha_bias = torch.from_numpy(np.random.randint(2, size=parent1_2_layer_bias.shape))

        c1["model"].layer2.weight.copy_(alpha_weight * parent1_2_layer_weight + (1-alpha_weight) * parent2_2_layer_weight)
        c2["model"].layer2.weight.copy_((1-alpha_weight) * parent1_2_layer_weight + alpha_weight * parent2_2_layer_weight)
        c1["model"].layer2.bias.copy_(alpha_bias * parent1_2_layer_bias + (1-alpha_bias) * parent2_2_layer_bias)
        c2["model"].layer2.bias.copy_((1-alpha_bias) * parent1_2_layer_bias + alpha_bias * parent2_2_layer_bias)


        # Crossover third layer
        parent1_3_layer_weight = p1["model"].layer3.weight
        parent2_3_layer_weight = p2["model"].layer3.weight
        parent1_3_layer_bias = p1["model"].layer3.bias
        parent2_3_layer_bias = p2["model"].layer3.bias

        alpha_weight = torch.from_numpy(np.random.randint(2, size=parent1_3_layer_weight.shape))
        alpha_bias = torch.from_numpy(np.random.randint(2, size=parent1_3_layer_bias.shape))

        c1["model"].layer3.weight.copy_(alpha_weight * parent1_3_layer_weight + (1-alpha_weight) * parent2_3_layer_weight)
        c2["model"].layer3.weight.copy_((1-alpha_weight) * parent1_3_layer_weight + alpha_weight * parent2_3_layer_weight)
        c1["model"].layer3.bias.copy_(alpha_bias * parent1_3_layer_bias + (1-alpha_bias) * parent2_3_layer_bias)
        c2["model"].layer3.bias.copy_((1-alpha_bias) * parent1_3_layer_bias + alpha_bias * parent2_3_layer_bias)

    return c1, c2

def mutate(agent, mutation_rate=0.3, sigma=0.3):

    mutated_agent = deepcopy(agent)
    # Mutate only 20% of the genes for each agent1
    with torch.no_grad():

        # Mutate layer 1
        agent_1_layer_weight = agent["model"].layer1.weight
        flag = torch.from_numpy(np.random.rand(*agent_1_layer_weight.shape) <= mutation_rate).type(torch.ByteTensor)
        mutated_agent["model"].layer1.weight[flag] = agent_1_layer_weight[flag] + torch.from_numpy(sigma * np.random.randn(*mutated_agent["model"].layer1.weight[flag].shape)).type(torch.FloatTensor)

        # Mutate layer 2
        agent_2_layer_weight = agent["model"].layer2.weight
        flag = torch.from_numpy(np.random.rand(*agent_2_layer_weight.shape) <= mutation_rate).type(torch.ByteTensor)
        mutated_agent["model"].layer2.weight[flag] = agent_2_layer_weight[flag] + torch.from_numpy(sigma * np.random.randn(*mutated_agent["model"].layer2.weight[flag].shape)).type(torch.FloatTensor)

        # Mutate layer 2
        agent_3_layer_weight = agent["model"].layer3.weight
        flag = torch.from_numpy(np.random.rand(*agent_3_layer_weight.shape) <= mutation_rate).type(torch.ByteTensor)
        mutated_agent["model"].layer3.weight[flag] = agent_3_layer_weight[flag] + torch.from_numpy(sigma * np.random.randn(*mutated_agent["model"].layer3.weight[flag].shape)).type(torch.FloatTensor)

    return mutated_agent

def roulette_selection(costs):
    p = random.uniform(0, sum(costs))
    for i, cost in enumerate(costs):
        if p < 0:
            break
        p -= cost
    return i


def iterate_population():

    agents = []
    best_agents = []
    # Initialise the first population
    for i in range(POPULATION):
        agent = {}
        # Create a new agent
        ag = Agent()
        #ag.load_state_dict(torch.load(os.path.join(os.getcwd(), 'agent2.pth')))
        #agent["model"] = Agent()
        agent["model"] = ag
        agent["cost"] = 0
        #agent = mutate(agent)
        # Add the agent to the population
        agents.append(agent)

    # Main evolutionary loop
    for i in range(ITERATIONS):

        offsprings = []
        for j in range(CHILDREN_NUMBER):

            costs = [i["cost"] for i in agents]

            if i != 0:
                i1 = roulette_selection(costs)
                i2 = roulette_selection(costs)

            else:
                permutation = np.random.permutation(POPULATION)
                i1 = permutation[0]
                i2 = permutation[1]

            parent1 = agents[i1]
            parent2 = agents[i2]

            c1, c2 = crossover(parent1, parent2)

            c1 = mutate(c1)
            c2 = mutate(c2)

            offsprings.append(c1)
            offsprings.append(c2)

        offsprings = flappy_game.play(offsprings)
        offsp_costs = [i["cost"] for i in offsprings]
        offsp_costs = sorted(offsp_costs, reverse=True)
        print("Iteration {} : Best Offspring Cost = {}".format(i, offsp_costs[0]))
        agents += offsprings
        agents = sorted(agents, key = lambda i: i["cost"], reverse=True)
        agents = agents[0:POPULATION]
        costs = [i["cost"] for i in agents]
        print(costs[0])
        best_agents.append(agents[0])

        if i % 25 == 0:
            torch.save(best_agents[i]["model"].state_dict(), os.path.join(os.getcwd(), "agent2.pth"))
            #print('haha teapa nu am salvat nimic')



iterate_population()
#ag = Agent()
#ag.load_state_dict(torch.load(os.path.join(os.getcwd(), 'agent2.pth')))
#agent = {"model" : ag,  "cost" : 0}
#flappy_game.play([agent])
