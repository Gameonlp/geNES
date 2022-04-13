import random
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import time
import socket
import shutil
import ast
import hashlib

import uuid
import queue

from multiprocessing import Process, Manager, Lock, Queue
from parse import parse

MAGNITUDE_RATE = 0.1
FREQUENCY_RATE = 0.75
PHASE_RATE = 0.5
NEW_WAVE_RATE = 0.01
WAVE_LOSS_RATE = 0.001
CROSS_OVER_RATE = 0.05
BIG_MUTATION_RATE = 0.001
MEDIUM_MUTATION_RATE = 0.05

buttons = ["A", "B", "Down", "Left", "Right", "Select",	"Start", "Up"]

manager = Manager()

orders = Queue()
to_plot = Queue()
fitness_tuples = Queue()

generation = 0
current = 1
def generate_population(size):
    global current
    population = []
    for i in range(size):
        genomes = dict()
        for j in range(len(buttons)):
            button = buttons[j]
            genomes[button] = []
            genomes[button].append([10 * random.random(), random.random(), 2 * random.random()])
        population.append(("w" + str(generation) + "." + str(current), genomes))
        current += 1
    return population

population = []
best50 = []
def mutate(genomes):
    new_genomes = (genomes[0], dict())
    while True:
        for button, genome in genomes[1].items():
            new_genomes[1][button] = []
            for k in range(len(genomes[1][button])):
                current = list(genomes[1][button][k])
                mut_size = random.random()
                if mut_size < BIG_MUTATION_RATE:
                    if random.random() < FREQUENCY_RATE:
                        current[0] += current[0] * (3 * random.random() - 2)
                    if random.random() < MAGNITUDE_RATE:
                        current[1] += current[1] * (3 * random.random() - 2)
                    if random.random() < MAGNITUDE_RATE:
                        current[2] += current[2] * (3 * random.random() - 2)
                elif mut_size < BIG_MUTATION_RATE + MEDIUM_MUTATION_RATE:
                    if random.random() < FREQUENCY_RATE:
                        current[0] += random.random() - 1
                    if random.random() < MAGNITUDE_RATE:
                        current[1] += random.random() - 1
                    if random.random() < MAGNITUDE_RATE:
                        current[2] += random.random() - 1
                else:
                    if random.random() < FREQUENCY_RATE:
                        current[0] += 0.2 * random.random() - 0.1
                    if random.random() < MAGNITUDE_RATE:
                        current[1] += 0.2 * random.random() - 0.1
                    if random.random() < MAGNITUDE_RATE:
                        current[2] += 0.2 * random.random() - 0.1
                new_genomes[1][button].append(current)
#            if random.random() < WAVE_LOSS_RATE and new_genomes[1][button]:
#                del new_genomes[1][button][random.randrange(0, len(new_genomes[1][button]))]
            if random.random() < NEW_WAVE_RATE:
                new_genomes[1][button].append([100 * random.random(), random.random(), 2 * random.random()])
            new_genomes[1][button].sort(key=lambda x: x[0])
        if new_genomes[1] not in [x[1] for x in population] + [x[1][1] for x in best50]:
            break
    return new_genomes


def crossover(mother, father):
    daughter = dict()
    son = dict()
    for i in range(len(buttons)):
        button = buttons[i]
        daughter[button] = []
        son[button] = []
        if random.random() < CROSS_OVER_RATE and len(mother[1][button]) > 1:
            split_point = random.randrange(0, len(mother[1][button]) - 1)
            split_gene = mother[1][button][split_point]
            father_split = 0
            for gene in father[1][button]:
                if gene[0] > split_gene[0]:
                    father_split = i
                    break
            son[button] = list(father[1][button][:father_split]) + list(mother[1][button][split_point:])
            daughter[button] = list(mother[1][button][:split_point]) + list(father[1][button][father_split:])
        else:
            if 0.5 < random.random():
                daughter[button] = list(mother[1][button])
                son[button] = list(father[1][button])
            else:
                daughter[button] = list(father[1][button])
                son[button] = list(mother[1][button])
    return (mother[0], daughter), (father[0], son)


def evaluate(parameters, xs):
    ret = []
    for x in xs:
        current = 0
        for parameter in parameters:
            current += parameter[1] * np.sin(parameter[0] * x + parameter[2])
        ret.append(current)
    return ret


def evaluation_process(conn):
    random.seed(time.time_ns())
    print(conn)
    try:
        while True:
            i, order = orders.get()
            conn.send((str(i) + "\n").encode("ascii"))
            for key in order[1]:
                conn.send((str(key) + "\n").encode("ascii"))
                for params in order[1][key]:
                    conn.send(((str(params[0]) + " " + str(params[1]) + " " + str(params[2])) + "\n").encode("ascii"))
                conn.send(("End button" + "\n").encode("ascii"))
            conn.send(("End individual" + "\n").encode("ascii"))
            while True:
                reply = conn.recv(2048)
                fit_i = parse("{} {}", reply.decode("ascii"))
                if i == int(fit_i[1]):
                    fitness_tuples.put((int(fit_i[0]), order))
                    print("Genome " + order[0] + " achieved a fitness of " + str(fit_i[0]))
                    break

    except Exception as e:
        orders.put((i, order))
        print("Connection " + str(conn) + "died, exception was " + str(e))
        sys.exit()

def evaluate_population(population):
    for i, genomes in enumerate(population):
        orders.put((i, genomes))
    fitness_map = []
    while len(fitness_map) < len(population):
        if not fitness_tuples.empty():
            fit_order = fitness_tuples.get()

            fitness_map.append(fit_order)
        time.sleep(1)
    return fitness_map

def handle_workers():
    TCP_IP = "localhost"
    TCP_PORT = 12345
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((TCP_IP, TCP_PORT))
    sock.listen(1)
    workers = []
    while True:
        to_remove = []
        for worker in workers:
            if not worker.is_alive():
                to_remove.append(worker)
        workers = [worker for worker in workers if worker not in to_remove]
        conn, addr = sock.accept()
        workers.append(Process(target=evaluation_process, args=(conn,)))
        workers[-1].start()
    
def plot():
    while True:
        best50 = to_plot.get()
        index = 0
        for fitness, genomes in best50:
            index += 1
            for key in genomes[1]:
                parameters = genomes[1][key]
                plt.title("Function for " + key + " of best genome " + str(genomes[0]) + " at index " + str(index) + " (" + str(fitness) + ")")
                t1 = np.arange(0.0, 50.0, 0.1)
                t2 = np.arange(0.0, 50.0, 1.0)
                #print(button)
                #print(parameters)
                y = evaluate(parameters, t1)
                y2 = evaluate(parameters, t2)
                plt.plot(t1, y, "lightgray", label="Input function")
                plt.plot(t2, y2, "xkcd:orange", label="Concrete input")
                plt.plot(t1, [0 for _ in t1], "-", c="k", label="Border for button presses, above means button pressed")
                plt.legend()
                plt.savefig("{:03d}".format(index) + key + ".png")
                plt.clf()

handler = Process(target=handle_workers)
handler.start()
plotter = Process(target=plot)
plotter.start()
while True:
    if os.path.exists('best') and not best50:
        with open('best', 'r') as f:
            gen = False
            fit = False
            nam = False
            name = ""
            fitness = 0
            in_button = False
            parameters = []
            genome = dict()
            button = ""
            for line in f:
                #print(line)
                if "End button" in line:
                    #print(1)
                    in_button = False
                    genome[button] = parameters
                    parameters = []
                elif "End individual" in line:
                    #print(5)
                    best50.append((fitness, (name, genome)))
                    nam = False
                    genome = dict()
                    fit = False
                elif not gen:
                    #print(3)
                    gen = True
                    generation = int(line.strip())
                elif not fit:
                    #print(4)
                    fit = True
                    fitness = int(line.strip())
                elif not nam:
                    name = line.strip()
                    nam = True
                elif not in_button:
                    #print(2)
                    button = line.strip()
                    in_button = True
                else:
                    #print(6)
                    params = parse('{} {} {}', line)
                    current = []
                    if params == None:
                        break
                    for param in params:
                        current.append(float(param))
                    parameters.append(current)
    elif not best50:
        checked_file = str(uuid.uuid4())
        population = generate_population(100)
        population = list(map(mutate, population))
        fitness_map = evaluate_population(population)
        fitness_map.sort(key=lambda x: x[0], reverse=True)
        best50 = fitness_map[:50]
    generation += 1
    current = 1
    mates = [(genome[1][0], dict(genome[1][1])) for genome in best50]
    population = []
    while len(population) < 50:
        selector = random.random()
        mother = None
        i = len(mates) - 1
        while mother == None:
            if selector < 5 / 6**i:
                mother = mates[i]
            i -= 1
        selector = random.random()
        father = None
        i = len(mates) - 1
        while father == None:
            if selector < 5 / 6**i:
                father = mates[i]
            i -= 1
        daughter, son = crossover(mother, father)
        population.append(daughter)
        population.append(son)
    while mates:
        mother = mates.pop(random.randrange(0, len(mates)))
        father = mates.pop(random.randrange(0, len(mates)))
        daughter, son = crossover(mother, father)
        population.append(daughter)
        population.append(son)
    population = list(map(mutate, population))
    wildcards = generate_population(5)
    for wildcard in wildcards:
        daughter, son = crossover(best50[0][1], wildcard)
        population.append(mutate(daughter))
        population.append(mutate(son))
    print("Starting Generation ", generation)
    fitness_map = evaluate_population(population)
    fitness_map.sort(key=lambda x: x[0], reverse=True)
    
    new_best50 = []
    pop_index = 0
    best50_index = 0
    index = 0
    while index < 50:
        if fitness_map[pop_index][0] >= best50[best50_index][0]:
            if fitness_map[pop_index][1][1] not in [x[1][1] for x in new_best50]: 
                new_best50.append(fitness_map[pop_index])
                pop_index = pop_index + 1
                print("Added new element with fitness of " + str(new_best50[index][0]))
            else:
                pop_index = pop_index + 1
                print("Did not add exact clone with fitness of " + str(fitness_map[pop_index][0]))
                continue
        elif fitness_map[pop_index][0] < best50[best50_index][0]:
            if best50[best50_index][1][1] not in [x[1][1] for x in new_best50]: 
                new_best50.append(best50[best50_index])
                best50_index = best50_index + 1
                print("Retained old element with fitness of " + str(new_best50[index][0]))
            else:
                best50_index = best50_index + 1
                print("Did not add exact clone with fitness of " + str(best50[best50_index][0]))
                continue
        assert new_best50[index][1][1]["Start"] or new_best50[index][0] == 0, "Start was empty"
        index = index + 1
    best50 = new_best50
    if os.path.exists("best"):
        shutil.copy("best", "best.bak")
    with open("best", "w") as best:
        best.write(str(generation) + "\n")
        for fitness, genomes in best50:
            best.write(str(fitness) + "\n")
            best.write(genomes[0] + "\n")
            for button, genome in genomes[1].items():
                    best.write(button + "\n")
                    for gene in genome:
                            best.write(str(gene[0]) + " " + str(gene[1]) + " " + str(gene[2]) + "\n")
                    best.write("End button\n")
            best.write("End individual\n")
    to_plot.put(best50)

