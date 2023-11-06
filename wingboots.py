#!/usr/bin/env python3
"""
Wingboots

In Slay the Spire, the player climbs floors one at a time, visiting
one node per floor.  Each floor has a number of nodes which are
connected via edges to the nodes of the next floor, and ordinarily the
player can only traverse the edges present in the map.  Also, with the
Wingboots relic, a player can also visit an arbitrary node on the next
floor without edge constraints, using up a Charge of the Wingboots.
The wingboots come with 3 charges but not all of them must be
consumed.

Nodes can be elite nodes or non-elite nodes, but elite nodes contain
valuable rewards and the player may wish to optimize for them.  Given
that the wingboots have k charges (able to ignore edge constraints up
to k times), what is the optimum number of elites and how is it
achieved?
"""
import sys
import unittest
from pprint import pprint

DEBUG = False


MAIN_DOCSTRING = """
Usage:
python3 wingboots.py( [num_wingboots_charges])?( [elite_description])+

num_wingboots_charges:
    <int>

elite_description:
   <floor_number>(.<floor_id>)?(all_elites_reachable)

all_elites_reachable:
   (-<floor_number>(.<floor_id>)?)*
"""

if __name__ == "__main__" and len(sys.argv) < 2:
    raise ValueError(MAIN_DOCSTRING)


MAX_WINGBOOTS = 3
MAX_INT = 99999

class Node():
    def __init__(self, id, parents=None, children=None):
        self.id = id
        self.parents = parents or set()
        self.children = children or set()

    def __repr__(self):
        return f"Node({repr(self.id)}, parents={set(parent.id for parent in self.parents)}, children={set(child.id for child in self.children)})"

    def __str__(self):
        return str(self.id)


class STSNode():
    """
    A node in slay the spire.  Note that in the max flow problem, we
    actually work with the dual graph so the STSNodes are edges with
    flow capacities that connect dummy nodes.
    """
    def __init__(self, id, parents=None, children=None, is_elite=True):
        """
        id = Id object representing the STSNode
        parent = STSNode object that can be used to reach this
        children = list(STSNode) that can be reached from this STSNode.
        """
        self.id = id
        self.parents = parents or set()
        self.children = children or set()
        self.is_elite = is_elite
        if not is_elite:
            raise NotImplementedError("Please only input the elite nodes.")


class Id():
    '''
    (Immutable) readable name for STSNode and Node.
    y = floor number, x = node-distance from the left.
    For Node, it's the STSNode that we are leaving from (if applicable).
    '''
    def __init__(self, y, x, after_k_jumps=0):
        self.y = y
        self.x = x
        self.after_k_jumps = after_k_jumps

    def __str__(self):
        return f"Floor {self.y}, Node {self.x} ({self.after_k_jumps} wingboots used)"

    def __repr__(self):
        return f"Id({self.y}, {self.x}, {self.after_k_jumps})"

    def __members(self):
        return (self.y, self.x, self.after_k_jumps)

    def __eq__(self, other):
        if type(other) is type(self):
            return self.__members() == other.__members()
        else:
            return False

    def __lt__(self, other):
        if type(other) is type(self):
            return self.__members() < other.__members()
        else:
            return False

    def __hash__(self):
        return hash(self.__members())


def _trace_back_max_paths_from(partial_max_paths, max_elites_up_to_edge, max_elites_up_to_sink):
    """
    In:

        - partial_max_paths (list(list(Node))): All of the same-length
          fragments of max paths ending in sink.  The length of each
          fragment can be determined by the max number of elites minus
          max_elites_up_to_sink (plus one for the sink connection).

        - max_elites_up_to_edge (dict(tuple(Id, Id) -> int)): maps
          edge (u, v) to the maximum elites traversed up to and
          including that edge.  (Plus one if edge ends at the sink.)

        - max_elites_up_to_sink (int): Max elites that can be found up
          to the first element of each path in partial_max_paths.

    Out:
        - max_paths
    """
    if max_elites_up_to_sink <= 0:
        return partial_max_paths

    output_max_paths = []
    for partial_max_path in partial_max_paths:
        sink = partial_max_path[0]
        for parent in sink.parents:
            if max_elites_up_to_edge.get((parent.id, sink.id), -1) == max_elites_up_to_sink:
                # This is a max path.
                output_max_paths.append([parent] + partial_max_path)
    return _trace_back_max_paths_from(output_max_paths, max_elites_up_to_edge, max_elites_up_to_sink - 1)


def trace_back_max_paths_from(sink, max_elites_up_to_edge):
    """
    In:
        - sink (Node): sink node to trace back max paths from

        - max_elites_up_to_edge (dict(tuple(Id, Id) -> int)): maps
          edge (u, v) to the maximum elites traversed up to and
          including that edge.  (Plus one if edge ends at the sink.)

    Output:
    
    """
    if DEBUG:
        print("MAX ELITES UP TO EDGE")
        pprint(max_elites_up_to_edge)
    # -1 means unreachable.
    max_elites_up_to_sink = max([max_elites_up_to_edge.get((parent.id, sink.id), -1) for parent in sink.parents])
    return _trace_back_max_paths_from([[sink]], max_elites_up_to_edge, max_elites_up_to_sink)


def wingboots(graph, source, sink, k=MAX_WINGBOOTS):
    """
    In:

        - graph (dict(Id -> Node)): where nodes represent decision
          points between STSNodes and edges represent an elite battle.

        - source (Id): source node in graph

        - sink (Id): sink node in graph

        - k (int): number of wingboots charges remaining

    Out:
        - all_paths (list(list(Node))): between source and sink of max length.

    Similar to the smugglers with hyperjumps problem.  (6-1)
    https://github.com/diego-escobedo/6.046/blob/master/PS6/pset6_questions.pdf

    Returns all paths of max length.

    Algorithm:
        DP Algorithm for finding the maximum length path in
        a dag with one source and sink: (source:
        https://stackoverflow.com/questions/10712495/longest-path-in-a-dag)

            for each node with no predecessor :
                for each of his leaving arcs, E=1.
            for each node whose predecessors have all been visited :
                for each of his leaving arcs, E=max(E(entering arcs))+1.
    """
    # set(Id): nodes which have had their outbound edges processed.
    visited = set()
    # list(Node): nodes whose predecessors have all been visited.
    frontier = [value for value in graph.values() if not value.parents]
    # dict(tuple(Id, Id) -> int): maps edge (u, v) to the maximum elites
    # traversed up to and including that edge.
    max_elites_up_to_edge = {}

    if DEBUG:
        print("GRAPH:")
        pprint(graph)

    while frontier:
        curr_node = frontier.pop()
        visited.add(curr_node.id)

        max_elites_before_edge = max(
            (max_elites_up_to_edge[(parent.id, curr_node.id)] for parent in curr_node.parents),
            default=0
        )

        for child in curr_node.children:
            max_elites_up_to_edge[(curr_node.id, child.id)] = max_elites_before_edge + 1
            if all(predecessor.id in visited for predecessor in child.parents):
                frontier.append(child)

    max_paths = trace_back_max_paths_from(graph[sink], max_elites_up_to_edge)
    # Sort by last elite number of hyperjumps, then by elite floor / id
    max_paths.sort(key=lambda path: (path[-2].id.after_k_jumps, *[x.id for x in path]))
    return max_paths


def connect_nodes(from_nodes, to_nodes):
    """
    In:
        - from_nodes (list(Node)): sources of the edges
        - to_nodes (list(Node)): destinations of the edges

    Mutates each node in 'from_nodes', 'to_nodes' so that the
    'from_nodes[i]' points to 'to_nodes[i]'.
    """
    for (from_node, to_node) in zip(from_nodes, to_nodes):
        from_node.children.add(to_node)
        to_node.parents.add(from_node)


def get_or_make_node_with_id(all_nodes, id):
    if id in all_nodes:
        return all_nodes[id]
    new_node = Node(id)
    all_nodes[id] = new_node
    return new_node


def convert_graph(all_elites, k=MAX_WINGBOOTS):
    """
    In:

        - all_elites (dict(tuple(floor, id) -> STSNode)): where nodes
          represent elite nodes and edges represent paths traveled
          between them.

    Out:

        - all_nodes (dict(Id -> Node)): where nodes represent decision
          points between STSNodes and edges represent an elite battle.

        - source_node (Node): decision point at the beginning of the
          act

        - sink_node (Node): decision point at the end of the act
    """
    source_node = Node(id=Id(-1, 0, 0))
    sink_node = Node(Id(MAX_INT, 0, k))
    all_nodes = {source_node.id: source_node, sink_node.id: sink_node}

    connect_nodes(from_nodes=[source_node], to_nodes=[sink_node])

    all_elites_sorted = list(sorted(all_elites.items(), key=lambda x: x[0]))
    for i, ((id_y, id_x), sts_node) in enumerate(all_elites_sorted):
        # By sorting by floor, we guarantee we see every elite's
        # parent before it (if it has one).
        curr_nodes = [
            get_or_make_node_with_id(all_nodes, Id(id_y, id_x, j))
            for j in range(k + 1)
        ]

        min_floor_node_after = MAX_INT
        for child_id in sts_node.children:
            min_floor_node_after = min(min_floor_node_after, child_id.y)
        if not sts_node.children:
            # Connect to sink node
            connect_nodes(from_nodes=curr_nodes, to_nodes=[sink_node]*len(curr_nodes))

        # Connect to all children elites with no hyperjumps required.
        for child_id in sts_node.children:
            child_nodes = [
                get_or_make_node_with_id(all_nodes, Id(child_id.y, child_id.x, j))
                for j in range(k + 1)
            ]
            connect_nodes(from_nodes=curr_nodes, to_nodes=child_nodes)

        # Connect to all non-children elites before min floor node
        # child with a hyperjump.
        for (child_id_y, child_id_x), child_elite in all_elites_sorted[i+1:]:
            if child_id_y <= id_y:
                # Cannot go backwards with a hyperjump.
                continue
            if child_elite.id in sts_node.children:
                # Do not hyperjump if it's a reachable elite.
                continue
            if child_id_y > min_floor_node_after:
                # Visiting this child via hyperjump is never more
                # optimal, since you could get an extra elite by going
                # through the min floor node elite too.
                break

            child_nodes = [
                get_or_make_node_with_id(all_nodes, Id(child_id_y, child_id_x, j))
                for j in range(k + 1)
            ]
            connect_nodes(from_nodes=curr_nodes[:-1], to_nodes=child_nodes[1:])

        if not sts_node.parents:
            connect_nodes(from_nodes=[source_node], to_nodes=curr_nodes[:1])

    return all_nodes, source_node.id, sink_node.id


def get_or_make_sts_node_with_id_tuple(all_elites, id):
    if id in all_elites:
        return all_elites[id]

    new_node = STSNode(Id(*id))
    all_elites[id] = new_node
    return new_node

def build_graph(elite_descriptions, k=MAX_WINGBOOTS):
    """
    In:
        - elite_descriptions (list(str)): list of elite_descriptions
        - k (int): number of wingboots charges

    elite_description:
    <floor_number>-<number-within-floor>(all_elites_reachable)

    all_elites_reachable:
    (-<floor_number>-<number-within-floor>)*
    """
    all_elites = {}
    for elite_description in elite_descriptions:
        info = elite_description.split('-')
        info = [
            x.split('.') if '.' in x
            else (x, '0')
            for x in info
        ]
        info = [(int(y), int(x)) for y, x in info]
        this_chunk, *children_chunks = info

        parent_elite = STSNode(Id(*this_chunk))
        all_elites[this_chunk] = parent_elite

        for elite in children_chunks:
            child_elite = get_or_make_sts_node_with_id_tuple(all_elites, elite)
            child_elite.parents.add(parent_elite.id)
            parent_elite.children.add(child_elite.id)

    dual_graph, source, sink = convert_graph(all_elites, k)
    return dual_graph, source, sink


def get_jumps(path):
    """
    In:
      - list(Node): path

    Out:
       - jumps set(Id): Id of elites that require a hyperjump before them.
    """
    jumps = set()
    curr_num_hyperjumps = 0
    for elite in path:
        if elite.id.after_k_jumps > curr_num_hyperjumps:
            jumps.add(elite.id)
            curr_num_hyperjumps = elite.id.after_k_jumps
    return jumps


def format_output(max_paths):
    """
    In:
        - list(list(Node)): list of optimum paths

    Out:
        - num_options: (int) number of optimum paths
        - num_elites: (int) number of elites in optimum path(s)
        - elites_to_print: list(list(tuple(is_jump, Node))) list of tuples with two items:
            (1) (bool) whether to jump prior to the elite
            (2) (Node) the elite to go to next    
    """
    
    num_options, num_elites = len(max_paths), len(max_paths[0]) - 2  # - 1 for source and sink
    elites_to_print = []
    for path in max_paths:
        path_to_print = []
        jumps = get_jumps(path)
        for elite in path[1:-1]:
            path_to_print.append(((elite.id in jumps), elite))
        elites_to_print.append(path_to_print)
    return num_options, num_elites, elites_to_print


def print_output(num_options, num_elites, elites_to_print):
    """
    Prints human-readable information given the outputs of format_output.
    """
    print(f"{num_options} options found with {num_elites} elites.")
    for i, max_path in enumerate(elites_to_print):
        print(f"Option {i+1}:") # +1 for laymen
        for is_jump, elite in max_path:
            if is_jump:
                print(f"\tJump prior to {elite}")
            else:
                print(f"\t{elite}")


def main(*args):
    """
    See MAIN_DOCSTRING.
    """
    if args[0].isdigit():
        k, *graph_inputs = args
        k = int(k)
    else:
        k, graph_inputs = MAX_WINGBOOTS, args

    graph, source, sink = build_graph(graph_inputs, k)
    max_paths = wingboots(graph, source, sink)
    formatted_paths = format_output(max_paths)
    return formatted_paths



class TestWingboots(unittest.TestCase):
    '''
    Test procedure:
    python3 -m pip install nose2
    python3 -m nose2 -v wingboots
    '''
    def _test_wingboots(self, inputs, expected):
        actual = main(*inputs)
        actual[2][:] = [
            [(is_jump, node.id) for (is_jump, node) in path]
            for path in actual[2]
        ]
        self.assertEqual(expected, actual)

    def test_no_elites(self):
        inputs = ['3']
        expected = (1, 0, [[]])
        self._test_wingboots(inputs, expected)

    def test_single_elite(self):
        inputs = ['3', '1']
        expected = (1, 1, [[(False, Id(1, 0, 0))]])
        self._test_wingboots(inputs, expected)

    def test_wingboot_jump(self):
        inputs = ['3', '1', '2']
        expected = (1, 2, [[(False, Id(1, 0, 0)), (True, Id(2, 0, 1))]])
        self._test_wingboots(inputs, expected)

    def test_wingboots_not_enough(self):
        inputs = ['0', '1', '2']
        expected = (2, 1, [[(False, Id(1, 0, 0))], [(False, Id(2, 0, 0))]])
        self._test_wingboots(inputs, expected)

    def test_double_diamond(self):
        inputs = ['0', '1.0-2', '1.1-2', '2-3.0-3.1', '3.0-4', '3.1-4', '4']
        expected = (4, 4, [
            [(False, Id(1, 0, 0)),(False, Id(2, 0, 0)),(False, Id(3, 0, 0)),(False, Id(4, 0, 0))],
            [(False, Id(1, 0, 0)),(False, Id(2, 0, 0)),(False, Id(3, 1, 0)),(False, Id(4, 0, 0))],
            [(False, Id(1, 1, 0)),(False, Id(2, 0, 0)),(False, Id(3, 0, 0)),(False, Id(4, 0, 0))],
            [(False, Id(1, 1, 0)),(False, Id(2, 0, 0)),(False, Id(3, 1, 0)),(False, Id(4, 0, 0))],
        ])
        self._test_wingboots(inputs, expected)

    def test_double_diamond_need_wingboots(self):
        inputs = ['3', '1.0', '1.1', '2', '3.0', '3.1', '4']
        expected = (4, 4, [
            [(False, Id(1, 0, 0)),(True, Id(2, 0, 1)),(True, Id(3, 0, 2)),(True, Id(4, 0, 3))],
            [(False, Id(1, 0, 0)),(True, Id(2, 0, 1)),(True, Id(3, 1, 2)),(True, Id(4, 0, 3))],
            [(False, Id(1, 1, 0)),(True, Id(2, 0, 1)),(True, Id(3, 0, 2)),(True, Id(4, 0, 3))],
            [(False, Id(1, 1, 0)),(True, Id(2, 0, 1)),(True, Id(3, 1, 2)),(True, Id(4, 0, 3))],
        ])
        self._test_wingboots(inputs, expected)

    def test_sort_by_num_charges(self):
        inputs = ['3', '1.0', '1.1-2', '2', '3.0', '3.1', '4']
        expected = (4, 4, [
            [(False, Id(1, 1, 0)),(False, Id(2, 0, 0)),(True, Id(3, 0, 1)),(True, Id(4, 0, 2))],
            [(False, Id(1, 1, 0)),(False, Id(2, 0, 0)),(True, Id(3, 1, 1)),(True, Id(4, 0, 2))],
            [(False, Id(1, 0, 0)),(True, Id(2, 0, 1)),(True, Id(3, 0, 2)),(True, Id(4, 0, 3))],
            [(False, Id(1, 0, 0)),(True, Id(2, 0, 1)),(True, Id(3, 1, 2)),(True, Id(4, 0, 3))],
        ])
        self._test_wingboots(inputs, expected)

    def test_daily_challenge_06nov2023(self):
        inputs = ['0', '5.0-7-11-12', '5.1-7-12', '7-11', '11', '12']
        expected = (2, 3, [
            [(False, Id(5, 0, 0)),(False, Id(7, 0, 0)),(False, Id(11, 0, 0))],
            [(False, Id(5, 1, 0)),(False, Id(7, 0, 0)),(False, Id(11, 0, 0))],
        ])
        self._test_wingboots(inputs, expected)


if __name__ == "__main__":
    formatted_paths = main(*sys.argv[1:])
    print_output(*formatted_paths)



