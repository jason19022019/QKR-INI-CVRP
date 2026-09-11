"""CVRPLIB-format .vrp instance loading."""
import os
import numpy as np


class VRPLoader:
    """Parses a CVRPLIB-format .vrp file into node coordinates/demands
    and vehicle capacity.

    ``self.nodes`` is an (N+1, 3) array: row 0 is the depot
    ``[x, y, demand=0]``, rows 1..N are customers ``[x, y, demand]``.
    """

    def __init__(self, filepath):
        self.filepath = filepath
        self.nodes, self.capacity = self.load_vrp()

    def load_vrp(self):
        coords, demands = {}, {}
        capacity = 0
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"File not found: {self.filepath}")
        with open(self.filepath, 'r') as f:
            lines = f.readlines()
        section = None
        for line in lines:
            line = line.strip()
            if line.startswith("CAPACITY"):
                capacity = float(line.split()[-1])
            elif line.startswith("NODE_COORD_SECTION"):
                section = "coords"
                continue
            elif line.startswith("DEMAND_SECTION"):
                section = "demand"
                continue
            elif line.startswith("DEPOT_SECTION"):
                break
            if section == "coords":
                parts = line.split()
                if len(parts) == 3:
                    coords[int(parts[0])] = (float(parts[1]), float(parts[2]))
            elif section == "demand":
                parts = line.split()
                if len(parts) == 2:
                    demands[int(parts[0])] = float(parts[1])
        nodes = []
        for i in sorted(coords.keys()):
            nodes.append([coords[i][0], coords[i][1], demands[i]])
        return np.array(nodes), capacity
