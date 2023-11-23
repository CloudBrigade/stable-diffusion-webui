import time
import torch
import habana_frameworks.torch.hpu.random as htrandom

class DeviceRunner(object):
    def __init__(self, device, use_hpu_graph = True):
        super().__init__()
        self.hpu_cache = {}
        self.device = device
        if device.type != "hpu":
            use_hpu_graph = False
        else:
            import habana_frameworks.torch.hpu.graphs as hpu_graphs
            self.copy_to = hpu_graphs.copy_to
            self.CachedParams = hpu_graphs.CachedParams
        self.use_hpu_graph = use_hpu_graph

    def run(self, func, arg):
        if self.use_hpu_graph:
            func_id = hash(func)
            if func_id in self.hpu_cache:
                self.copy_to(self.hpu_cache[func_id].graph_inputs, arg)
                self.hpu_cache[func_id].graph.replay()
                return self.hpu_cache[func_id].graph_outputs
            str = "Compiling HPU graph {:26s} ".format(func.__name__)
            print(str)
            t_start = time.time()
            import habana_frameworks.torch.hpu as hpu
            graph = hpu.HPUGraph()
            graph.capture_begin()
            out = func(arg)
            graph.capture_end()
            self.hpu_cache[func_id] = self.CachedParams(arg, out, graph)
            print("{} took {:6.2f} sec".format(str, time.time() - t_start))
            return out
        elif self.device.type == "hpu":
            import habana_frameworks.torch.core as core
            core.mark_step()
        return func(arg)

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

