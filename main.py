import sys
import math
from collections import defaultdict
from queue import Queue

# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.


SUN_PER_TREE = 3
TREE_SUN_REQ = 4
STARTING_NUTRIENT = 20
NUM_DAYS = 24

SUN_HARVAST = {1: 1, 2: 2, 3: 3}
BONUS_POINTS_HARVEST = {1: 0, 2: 2, 3: 4}
GROW_COST = {0: 1, 1: 3, 2: 7}
NUM_CELLS = int(input())  # 37

GROW = "GROW"
SEED = "SEED"
COMPLETE = "COMPLETE"
WAIT = "WAIT"

INDEX_TO_VECTOR = {
    0: (0, 0, 0),
    1: (1, -1, 0),
    2: (1, 0, -1),
    3: (0, 1, -1),
    4: (-1, 1, 0),
    5: (-1, 0, 1),
    6: (0, -1, 1),
    7: (2, -2, 0),
    8: (2, -1, -1),
    9: (2, 0, -2),
    10: (1, 1, -2),
    11: (0, 2, -1),
    12: (-1, 2, -1),
    13: (-2, 2, 0),
    14: (-2, 1, 1),
    15: (-2, 0, 2),
    16: (-1, -1, 2),
    17: (0, -2, 2),
    18: (1, -2, 1),
    19: (3, -3, 0),
    20: (3, -2, -1),
    21: (3, -1, -2),
    22: (3, 0, -3),
    23: (2, 1, -3),
    24: (1, 2, -3),
    25: (0, 3, -3),
    26: (-1, 3, -2),
    27: (-2, 3, -1),
    28: (-3, 3, 0),
    29: (-3, 2, 1),
    30: (-3, 1, 2),
    31: (-3, 0, 3),
    32: (-2, -1, 3),
    33: (-1, -2, 3),
    34: (0, -3, 3),
    35: (1, -3, 2),
    36: (2, -3, 1)}


def log(*args):
    for arg in args + ('\n',):
        print(arg, file=sys.stderr, end=' ', flush=True)


class Vector:
    # cube coordinates on hex grid
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def __floordiv__(self, r):
        return Vector(self.x // r, self.y // r, self.z // r)

    def __str__(self):
        return str((self.x, self.y, self.z))

    @staticmethod
    def from_index(index):
        x, y, z = INDEX_TO_VECTOR[index]
        return Vector(x, y, z)

    def dist(self, other):
        return max(abs(self.x - other.x), abs(self.y - other.y), abs(self.z - other.z))

    def colinear(self, other):
        delta = other - self
        dcoords = (delta.x, delta.y, delta.z)
        return any(c == 0 for c in dcoords) and sum(dcoords) == 0


class Hex:
    def __init__(self, index, richness):
        self.index = index
        self.richness = richness
        self.tree = None


class Tree:
    def __init__(self, cell_index, size, is_mine, is_dormant):
        self.cell = cell_index
        self.size = size
        self.is_mine = is_mine
        self.is_dormant = is_dormant


class Graph:
    def __init__(self, num_nodes):
        self.edges = defaultdict(set)
        self.hexes = {}

    def get_hex(self, cell):
        return self.hexes[cell]

    def add_edge(self, u, v):
        self.edges[u].add(v)
        self.edges[v].add(u)

    def get_disc(self, start, radius):
        # get set of all cells of distance at most radius away from start
        q = Queue()
        q.put((start, 0))
        visited = set()
        visited.add(start)
        while not q.empty():
            current, dist = q.get()
            for nbr in self.edges[current]:
                if nbr in visited:
                    continue
                new_dist = dist + 1
                if new_dist <= radius:
                    q.put((nbr, new_dist))
                    visited.add(nbr)

        return list(visited)

    def cleanup(self):
        for hex in self.hexes.values():
            hex.tree = None


graph = Graph(NUM_CELLS)

for i in range(NUM_CELLS):
    # index: 0 is the center cell, the next cells spiral outwards
    # richness: 0 if the cell is unusable, 1-3 for usable cells
    # neigh_0: the index of the neighbouring cell for each direction
    index, richness, neigh_0, neigh_1, neigh_2, neigh_3, neigh_4, neigh_5 = [int(j) for j in input().split()]
    graph.hexes[index] = Hex(index, richness)
    for nbr in (neigh_0, neigh_1, neigh_2, neigh_3, neigh_4, neigh_5):
        if nbr != -1:
            graph.edges[index].add(nbr)


class Action:
    def __init__(self, s):
        tokens = s.split()
        if tokens[0] == GROW:
            self.command = GROW
            self.target_cell = int(tokens[1])
        elif tokens[0] == SEED:
            self.command = SEED
            self.source_cell = int(tokens[1])
            self.target_cell = int(tokens[2])
        elif tokens[0] == COMPLETE:
            self.command = COMPLETE
            self.target_cell = int(tokens[1])
        elif tokens[0] == WAIT:
            self.command = WAIT


class Game:
    def __init__(self):
        self.day = None
        self.nutrients = None
        self.my_sun = None
        self.enemy_sun = None
        self.my_score = None
        self.enemy_score = None
        self.enemy_is_waiting = None

        self.number_of_trees = None
        self.my_trees = []
        self.enemy_trees = []

        self.possible_actions = []
        self.remaining_days = defaultdict(int)
        self.tree_size_counts = defaultdict(int)

    def preprocess(self):
        # count how many times the sun will be in each direction
        for day in range(self.day + 1, NUM_DAYS):
            sun_direction = Vector.from_index(day % 6 + 1)
            self.remaining_days[sun_direction] += 1

            # tally each tree size
        for tree in self.my_trees:
            self.tree_size_counts[tree.size] += 1

    def grow(self, cell):
        print("GROW {}".format(cell))

    def wait(self):
        print("WAIT")

    def complete(self, cell):
        print("COMPLETE {}".format(cell))

    def seed(self, source_cell, dest_cell):
        print("SEED {} {}".format(source_cell, dest_cell))

    def decide(self):

        best_grow = None

        def gain(tree):
            # given a tree compute the total expected profit from growing it
            current_num_tall_nbrs = sum(graph.get_hex(cell).tree.size >= tree.size for cell in graph.edges[tree.cell] if
                                        graph.get_hex(cell).tree is not None)
            current_sun_per_turn = tree.size * (6 - current_num_tall_nbrs) / 6
            current_sun_earned = (NUM_DAYS - 1 - self.day) * current_sun_per_turn

            grown_num_tall_nbrs = sum(
                graph.get_hex(cell).tree.size >= (tree.size + 1) for cell in graph.edges[tree.cell] if
                graph.get_hex(cell).tree is not None)
            grown_sun_per_turn = (tree.size + 1) * (6 - grown_num_tall_nbrs) / 6
            grown_sun_earned = (NUM_DAYS - 1 - self.day) * grown_sun_per_turn

            cost_to_grow = GROW_COST[tree.size] + self.tree_size_counts[tree.size + 1]
            return grown_sun_earned - current_sun_earned - cost_to_grow

        if any(tree.size < 3 for tree in self.my_trees):
            best_tree = max((tree for tree in self.my_trees if tree.size < 3),
                            key=lambda tree: (gain(tree), graph.get_hex(tree.cell).richness))
            if gain(best_tree) > 0:
                best_grow = best_tree

        # can we complete a tree COMPLETING
        # complete when no 'viable grow' strategy
        if any(action.command == COMPLETE for action in self.possible_actions):

            richness = max(graph.get_hex(action.target_cell).richness for action in self.possible_actions if
                           action.command == COMPLETE)
            # predict number of points gained from harvesting today vs tomorrow
            points_today = self.nutrients + BONUS_POINTS_HARVEST[richness]
            predicted_nutrients_tomorrow = self.nutrients
            if not self.enemy_is_waiting:
                predicted_nutrients_tomorrow -= sum(tree.size == 3 and not tree.is_dormant for tree in self.enemy_trees)
            points_tomorrow = SUN_HARVAST[3] // 3 + predicted_nutrients_tomorrow + BONUS_POINTS_HARVEST[richness]

            if day == NUM_DAYS - 1:
                points_tomorrow = 0

            advantage = points_today - points_tomorrow

            if advantage > 0:
                best_action = max((action for action in self.possible_actions if action.command == COMPLETE),
                                  key=lambda action: graph.get_hex(action.target_cell).richness)
                self.complete(best_action.target_cell)
                return

                # should we plant a seed? SEEDING
        # if we have no seeds, plant the seed at max richness, that will not often be shadowed
        # and self.tree_size_counts[0]==0 \
        if any(action.command == SEED for action in self.possible_actions) and self.tree_size_counts[0] == 0:
            def score(action):
                seed_cell = action.target_cell
                seed_vector = Vector.from_index(seed_cell)
                richness = graph.get_hex(seed_cell).richness
                unobstructed_directions = set(Vector.from_index(i) for i in range(1, 7))
                for cell in graph.get_disc(seed_cell, 3):
                    if cell == seed_cell:
                        continue
                    vector = Vector.from_index(cell)
                    if graph.get_hex(cell).tree is not None and seed_vector.colinear(vector):
                        if graph.get_hex(cell).tree.size >= seed_vector.dist(vector):
                            # this tree can block us
                            offending_direction = (seed_vector - vector) // seed_vector.dist(vector)
                            if offending_direction in unobstructed_directions:
                                unobstructed_directions.remove(offending_direction)

                num_fruitful_days = sum(self.remaining_days[d] for d in unobstructed_directions)
                parent_is_size_3 = graph.get_hex(action.source_cell).tree.size == 3
                sun_gained = (num_fruitful_days - 1) * SUN_HARVAST[1] + 3 * BONUS_POINTS_HARVEST[richness]
                return (sun_gained, parent_is_size_3)

            best_action = max((action for action in self.possible_actions if action.command == SEED), key=score)
            if self.day <= NUM_DAYS - 5:
                self.seed(best_action.source_cell, best_action.target_cell)
                return

                # GROWING
        # should we grow a tree, grow the previously decided best tree

        if any(action.command == GROW for action in self.possible_actions):
            if best_grow is not None:
                # pick a tree of best sun gain
                best_action = max((action for action in self.possible_actions if action.command == GROW),
                                  key=lambda action: gain(graph.get_hex(action.target_cell).tree))
                self.grow(best_action.target_cell)
                return
            else:
                # grow the best for victory points if it can be completed
                # which is biggest tree followed by richness point
                def score(action):
                    tree = graph.get_hex(action.target_cell).tree
                    points_gain = self.nutrients + BONUS_POINTS_HARVEST[graph.get_hex(tree.cell).richness]
                    points_profit = points_gain + gain(tree) / 3
                    can_finish = tree.size + (NUM_DAYS - 1 - self.day) >= 3
                    return (can_finish, tree.size, points_profit)

                best_action = max((action for action in self.possible_actions if action.command == GROW), key=score)
                if score(best_action)[0]:
                    self.grow(best_action.target_cell)
                    return

        # wait
        self.wait()
        return

    # game loop


while True:

    game = Game()

    day = int(input())  # the game lasts 24 days: 0-23

    game.day = day

    nutrients = int(input())  # the base score you gain from the next COMPLETE action

    game.nutrients = nutrients

    # sun: your sun points
    # score: your current score
    sun, score = [int(i) for i in input().split()]
    game.my_sun = sun
    game.my_score = score

    inputs = input().split()
    opp_sun = int(inputs[0])  # opponent's sun points
    opp_score = int(inputs[1])  # opponent's score
    game.enemy_sun = opp_sun
    game.enemy_score = opp_score

    opp_is_waiting = inputs[2] != "0"  # whether your opponent is asleep until the next day
    game.enemy_is_waiting = opp_is_waiting

    number_of_trees = int(input())  # the current amount of trees
    game.number_of_trees = number_of_trees
    for i in range(number_of_trees):
        inputs = input().split()
        cell_index = int(inputs[0])  # location of this tree
        size = int(inputs[1])  # size of this tree: 0-3
        is_mine = inputs[2] != "0"  # 1 if this is your tree
        is_dormant = inputs[3] != "0"  # 1 if this tree is dormant

        tree = Tree(cell_index, size, is_mine, is_dormant)

        if is_mine == 1:
            game.my_trees.append(tree)
        else:
            game.enemy_trees.append(tree)

        graph.hexes[cell_index].tree = tree

    number_of_possible_actions = int(input())  # all legal actions
    for i in range(number_of_possible_actions):
        possible_action = input()  # try printing something from here to start with
        game.possible_actions.append(Action(possible_action))

    game.preprocess()
    game.decide()

    # clean up
    graph.cleanup()