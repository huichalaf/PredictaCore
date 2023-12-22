import numpy as np

# Estados: Inactiva, En Operación, En Mantenimiento
states = ["Inactiva", "En Operacion", "En Mantenimiento"]

# Matriz de Transición Probabilística
# Las filas corresponden al estado actual y las columnas al estado siguiente.
transition_matrix = np.array([[0.7, 0.2, 0.1],  # Probabilidades desde Inactiva
                              [0.1, 0.8, 0.1],  # Probabilidades desde En Operación
                              [0.1, 0.1, 0.8]]) # Probabilidades desde En Mantenimiento

def next_state(current_state):
    return np.random.choice(states, p=transition_matrix[states.index(current_state)])

def simulate_markov_chain(start_state, steps):
    current_state = start_state
    print("Estado Inicial:", current_state)
    for i in range(steps):
        next_step = next_state(current_state)
        print("Paso", i+1, "->", next_step)
        current_state = next_step

# Configuración inicial
initial_state = 'Inactiva'
total_steps = 10

# Simulación de la Cadena de Markov
simulate_markov_chain(initial_state, total_steps)
