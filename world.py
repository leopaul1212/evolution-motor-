from creature import Creature
import numpy as np
import random

def create_creature(nbr_creature:int, input_size:int, output_size:int):
    """Create a list of Creature instances with random structures."""
    creatures = []
    for i in range (nbr_creature):
        structure = [input_size]
        num_hidden_layers = random.randint(1, 3)
        for i in range (len(num_hidden_layers)):
            structure.append(random.randint(3, 10))
        structure.append(output_size)
        creatures.append(Creature(structure, id=i))
    return creatures
