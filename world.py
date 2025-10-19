from creature import Creature
import numpy as np
import random


data = np.load("dataset_64x64.npz")
X = data['X']  # shape = (4096, n_samples)
Y = data['Y']  # shape = (4096, n_samples)


def create_creature(nbr_creature:int, input_size:int, output_size:int):
    """Create a list of Creature instances with random structures."""
    creatures = []
    for i in range (nbr_creature):
        structure = [input_size]
        num_hidden_layers = random.randint(1, 3)
        for i in range (num_hidden_layers):
            structure.append(random.randint(3, 10))
        structure.append(output_size)
        creatures.append(Creature(structure, id=i))
    return creatures


def life (creature:list[Creature], input_data:np.ndarray, Y:np.ndarray):
    for i in range (len(creature)):
        activations = creature[i].forward_propagation(input_data)
        output = activations['A' + str(len(creature[i].structure) - 1)]
        creature[i].score = np.mean((output - Y) ** 2) 
    creature.sort(key=lambda c: c.score)
    newcreature = creature[:len(creature)//2]
    return newcreature
    
def mutate_creature(creatures: list[Creature], mutation_rate: float = 0.01, mutation_strength: float = 0.1):
    """Return the original creatures plus one mutated offspring per creature."""
    if len(creatures) == 0:
        return []

    new_population: list[Creature] = []
    next_available_id = max(creature.id for creature in creatures) + 1

    for original in creatures:
        new_population.append(original)

        mutated_params = {}
        for param_name, values in original.neural_network_params.items():
            mutated = values.copy()
            if mutation_rate > 0:
                mask = np.random.rand(*mutated.shape) < mutation_rate
                noise = np.random.normal(0.0, mutation_strength, size=mutated.shape)
                mutated[mask] += noise[mask]
            mutated_params[param_name] = mutated

        mutated_creature = Creature(original.structure.copy(), id=next_available_id)
        next_available_id += 1

        mutated_creature.neural_network_params = mutated_params
        mutated_creature.neural_network = mutated_params
        mutated_creature.DNA = mutated_creature.createDNA()

        new_population.append(mutated_creature)

    return new_population

creature = create_creature(20, 4096, 4096)
for generation in range (50):
    creature = life(creature, X, Y)
    print(f"Generation {generation}: Best score = {creature[0].score}")
    
