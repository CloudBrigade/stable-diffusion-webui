import torch
import habana_frameworks.torch.hpu.random as htrandom

class HPUGenerator:
    def __init__(self):
        self.state = htrandom.get_rng_state()
        self.initial_seed_value = htrandom.initial_seed()

    def get_state(self):
        # PyTorch’s Generator.get_state returns a tensor, same as htrandom.get_rng_state
        return htrandom.get_rng_state()

    def set_state(self, state):
        htrandom.set_rng_state(state)
        self.state = state

    def manual_seed(self, seed):
        htrandom.manual_seed(seed)
        self.initial_seed_value = seed
        self.state = htrandom.get_rng_state()
        return self

    def seed(self):
        # Assuming htrandom.seed generates a new seed internally and sets it
        htrandom.seed()
        self.state = htrandom.get_rng_state()
        self.initial_seed_value = htrandom.initial_seed()  # Update initial_seed based on new state

    def initial_seed(self):
        return self.initial_seed_value

# Usage Example
#generator = HPUGenerator()
#state = generator.get_state()
#print(f"Initial State: {state}")
#
#generator.set_state(state)
#print("State is set back to its initial value.")
#
#generator.manual_seed(42)
#print(f"Manual Seed: {generator.initial_seed()}")
#
#generator.seed()
#print(f"Seed is set to a new value. New Initial Seed: {generator.initial_seed()}")

