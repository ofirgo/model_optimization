# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import copy
from typing import List, Callable

from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.graph.memory_graph.cut import Cut
from model_compression_toolkit.core.common.graph.memory_graph.memory_element import MemoryElements
from model_compression_toolkit.core.common.graph.memory_graph.memory_graph import ActivationMemoryTensor


class DummpyType:
    pass


class DummyBaseNodeGenerator:
    def __init__(self):
        self.counter = 0

    def __iter__(self):
        while True:
            yield BaseNode(name="dummy_node_" + str(self.counter),
                           framework_attr={},
                           input_shape=tuple(),
                           output_shape=tuple(),
                           weights={},
                           layer_class=DummpyType,
                           has_activation=False)

            self.counter += 1


class DummyActivationMemoryTensorGenerator:
    def __init__(self):
        self.counter = 0

    def __iter__(self):
        while True:
            yield ActivationMemoryTensor(shape=tuple(),
                                         node_name="dummy_tensor_" + str(self.counter),
                                         node_output_index=0,
                                         total_size=0)

            self.counter += 1


class MaxCutAstar:

    def __init__(self, memory_graph, estimate_factor: float = None):

        self.memory_graph = memory_graph
        self.estimate_factor = self.get_init_estimate_factor(memory_graph) \
            if estimate_factor is None else estimate_factor

        # In order to run Astar search on the graph we need to define a single source and a single target.
        # Dummy nodes are used for this purpose.
        # We do it a bit different from the original scheduler implementation.
        gen_a = iter(DummyBaseNodeGenerator())
        gen_b = iter(DummyActivationMemoryTensorGenerator())

        # Source Cut
        src_dummy_a = next(gen_a)
        src_dummy_b = next(gen_b)
        edges_src_ab = [(src_dummy_a, src_dummy_b)]
        edges_src_ba = [(src_dummy_b, src_a) for src_a in memory_graph.sources_a]

        # Updated graph sinks - need to connect all sink nodes to a single sink node
        # mid_b_nodes = [next(gen_b) for _ in memory_graph.sinks_a]
        # sinks_fix_ab_edges = [(original_sink, mid_b) for original_sink, mid_b in zip(memory_graph.sinks_a, mid_b_nodes)]
        # sink_fix_node_a = next(gen_a)  # this is the new single sink node
        # sinks_fix_ba_edges = [(t, sink_fix_node_a) for s, t in sinks_fix_ab_edges]

        # Target Cut
        target_dummy_a = next(gen_a)
        target_dummy_a2 = next(gen_a)
        target_dummy_b = next(gen_b)
        target_dummy_b2 = next(gen_b)
        # edges_target_ab = [(sink_fix_node_a, target_dummy_b1), (target_dummy_a1, target_dummy_b2)]
        # edges_target_ab = [(sink_fix_node_a, target_dummy_b1)]
        # edges_target_ba = [(target_dummy_b1, target_dummy_a1), (target_dummy_b2, target_dummy_a2)]
        edges_target_fix_ba = [(t, target_dummy_a) for t in memory_graph.sinks_b]
        edges_target_ab = [(target_dummy_a, target_dummy_b), (target_dummy_a2, target_dummy_b2)]
        edges_target_ba = [(target_dummy_b, target_dummy_a2)]

        # Update memory graph
        # New nodes
        # self.memory_graph.add_nodes_to_a([src_dummy_a, sink_fix_node_a, target_dummy_a1, target_dummy_a2])
        # self.memory_graph.add_nodes_to_b([src_dummy_b, target_dummy_b1, target_dummy_b2] + mid_b_nodes)
        # self.memory_graph.add_nodes_to_a([src_dummy_a, sink_fix_node_a, target_dummy_a1])
        self.memory_graph.add_nodes_to_a([src_dummy_a, target_dummy_a, target_dummy_a2])
        # self.memory_graph.add_nodes_to_b([src_dummy_b, target_dummy_b1] + mid_b_nodes)
        self.memory_graph.add_nodes_to_b([src_dummy_b, target_dummy_b, target_dummy_b2])
        # New Edges
        # self.memory_graph.add_edges(edges_src_ab + edges_src_ba + sinks_fix_ab_edges + sinks_fix_ba_edges + edges_target_ab + edges_target_ba)
        self.memory_graph.add_edges(edges_src_ab + edges_src_ba + edges_target_fix_ba + edges_target_ab + edges_target_ba)
        self.memory_graph.update_sources_a()
        self.memory_graph.update_sinks_b()

        self.src_cut = Cut([src_dummy_a], {src_dummy_a}, MemoryElements(elements={src_dummy_b}, total_size=0))
        # self.target_cut = Cut([], set(), MemoryElements(elements={target_dummy_b1, target_dummy_b2},
        #                                                 total_size=0))
        self.target_cut = Cut([], set(), MemoryElements(elements={target_dummy_b, target_dummy_b2},
                                                        total_size=0))

    def solve(self, iter_limit: int):  # TODO: what is a reasonable default number of iterations to set?
        open_list = [self.src_cut]
        closed_list = []
        costs = {self.src_cut: self.src_cut.mem_elements.total_size}
        routes = {self.src_cut: [self.src_cut]}

        expansion_count = 0

        while expansion_count < iter_limit and len(open_list) > 0:
            # Choose next node to expand
            next_cut = self._get_cut_to_expand(open_list, costs, routes)

            cut_cost = costs[next_cut]
            cut_route = routes[next_cut]

            if next_cut == self.target_cut:
                return cut_cost, self._remove_dummys_from_path(cut_route[0].op_order)

            if self.is_pivot(next_cut):
                # Can clear all search history
                open_list = []
                closed_list = []
                routes = {}
            else:
                # Can remove only next_cut and put it in closed_list
                open_list.remove(next_cut)
                del routes[next_cut]
                closed_list.append(next_cut)

            # Expand the chosen cut
            expanded_cuts = self.expand(next_cut)
            expansion_count += 1

            # Only consider nodes that where not already visited
            expanded_cuts = list(filter(lambda _c: _c not in closed_list, expanded_cuts))
            for c in expanded_cuts:
                cost = self.accumulate(cut_cost, c.mem_elements.total_size)
                if c not in open_list:
                    self._update_expanded_node(c, cost, [c] + cut_route, open_list, costs, routes)
                elif self.ordering(cost, costs[c]):
                    # If we already saw this cut during the search with a larger cost, then we want to update the order
                    # of the schedule in the cut
                    # Remove call - removes the cut with the same memory elements but different ordering from open
                    # Then - adds the cut with the improved ordering from open
                    open_list.remove(c)
                    self._update_expanded_node(c, cost, cut_route, open_list, costs, routes)

        # Halt or No Solution
        return None

    @staticmethod
    def _update_expanded_node(cut, cost, route, open_list, costs, routes):
        open_list.append(cut)
        costs.update({cut: cost})
        routes.update({cut: [cut] + route})

    def _get_cut_to_expand(self, open_list, costs, routes):
        ordered_cuts_list = sorted(open_list,
                                   key=lambda c: (self.accumulate(costs[c], self.estimate(c)), len(routes[c])),
                                   reverse=False)

        assert len(ordered_cuts_list) > 0
        return ordered_cuts_list[0]

    def clean_memory_for_next_step(self, cut: Cut) -> Cut:
        """
         This is an auxiliary function that removes irrelevant memory elements from a cut.
         The memory elements are irrelevant if all operations depended on them have already been executed.

        Args:
            cut: A Cut to remove elements from.

        Returns: A new Cut with updated memory elements list.

        """

        cut_records_names = cut.get_record_names()
        filtered_memory_elements = set(filter(lambda elm: not all(child.name in cut_records_names for child in
                                                                  self.memory_graph.activation_tensor_children(elm)),
                                              cut.mem_elements.elements))

        return Cut(cut.op_order, cut.op_record,
                   mem_elements=MemoryElements(filtered_memory_elements,
                                               sum([elm.total_size for elm in filtered_memory_elements])))

    def can_expand(self, op_node: BaseNode, cut: Cut) -> bool:
        """
        Checks whether a cut can be expanded by adding an operation node to it, that is,
        all the required memory elements for the operation computation exist in the cut.

        Args:
            op_node: An operation node to check if it can expand the cut.
            cut: A cut that contains the op_node.

        Returns: Whether the cut can be expanded by expanding the op_node.
        """

        clean_cut = self.clean_memory_for_next_step(cut)
        return op_node not in cut.op_record and len(cut.mem_elements.elements) > 0 and \
               all([parent_mem_element in clean_cut.mem_elements.elements for parent_mem_element in self.memory_graph.operation_node_parents(op_node)])

    def expand(self, cut: Cut) -> List[Cut]:
        clean_cut = self.clean_memory_for_next_step(cut)

        # candidates for expansion are children of the memory elements from the cleaned cut that can be expanded
        candidates = []
        for mem_element in clean_cut.mem_elements.elements:
            child_ops = self.memory_graph.activation_tensor_children(mem_element)
            ops_to_expand = list(filter(lambda op: self.can_expand(op, clean_cut) and op not in candidates, child_ops))
            candidates.extend(ops_to_expand)

        # for each candidate a cut is returned with the candidate expanded
        # (operation is added to record / order and resulting memory elements added to memory elements)
        next_cuts = []
        for candidate in candidates:
            op_order = clean_cut.op_order + [candidate]

            op_record = clean_cut.op_record.copy()
            op_record.add(candidate)

            mem_elements = copy.copy(clean_cut.mem_elements)
            mem_elements.add_elements_set(set(self.memory_graph.operation_node_children(candidate)))

            expanded_cut = Cut(op_order, op_record, mem_elements)
            next_cuts.append(expanded_cut)

        return next_cuts

    def is_pivot(self, cut: Cut) -> bool:
        """
        returns true if Cut is a pivot i.e. the cut must be in the selected route.
        If all memory elements in the cut have the same parent it is a pivot.

        Args:
            cut: A Cut to check whether it is a pivot.

        Returns: True if the given cut is a pivot.
        """

        clean_cut = self.clean_memory_for_next_step(cut)

        mem_elements_parents = list(map(lambda mem_elm: self.memory_graph.activation_tensor_parents(mem_elm), clean_cut.mem_elements.elements))
        unique_parents = set(sum(mem_elements_parents, []))

        return len(unique_parents) == 1

    @staticmethod
    def accumulate(cost_1: float, cost_2: float) -> float:
        """
        A function defining the accumulation method of costs for Astar search.
        We take the maximum memory requirement of the schedule, in order to find the minimal maximum out of all possible schedules.

        Args:
            cost_1: The first schedule option's cost.
            cost_2: The second schedule option's cost.

        Returns: The maximum out of the two costs.

        """
        return max(cost_1, cost_2)

    @staticmethod
    def ordering(cost_1, cost_2) -> bool:
        return cost_1 < cost_2

    def estimate(self, cut: Cut) -> float:
        return self.estimate_factor * self.memory_graph.memory_lbound_single_op

    @staticmethod
    def get_init_estimate_factor(memory_graph):
        l_bound = memory_graph.memory_lbound_single_op
        u_bound = 2 * sum([t.total_size for t in memory_graph.b_nodes]) - l_bound
        return (u_bound + l_bound) / 2

    @staticmethod
    def _remove_dummys_from_path(path):
        # TODO: make "dummy_node" a constant and use it here and in the DummyBaseNodeGenerator class
        return list(filter(lambda n: 'dummy_node' not in n.name, path))

