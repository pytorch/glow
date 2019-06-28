#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
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

import argparse
import re
from typing import (
    List,
    Dict,
    Tuple,
)
import typing

# Maintaining all nodes
NODES_MAP: Dict['NodeNameAndKind', 'Node'] = {}

# Scope stack
SCOPE_STACK: List[str] = []


class NodeNameAndKind(typing.NamedTuple):
    """A class that represents the named tuple of node name and kind."""

    name: str
    kind: str


class NodeValue(typing.NamedTuple):
    """A class that represents the named tuple of node and result number."""

    node: 'Node'
    resNo: int


class Node:
    """A class that represents a node in the compute graph

    Public attributes:
        kindName_: str. The kind name.
        name_: str. The node name.
        inputs_: List[NodeValue]. Input node values.
        users_: Dict['Node', int]. The users of this node.
    """

    def __init__(self, kindName: str, name: str):
        self.kindName_: str = kindName
        self.name_: str = name
        self.inputs_: List[NodeValue] = []
        self.users_: Dict['Node', int] = {}

    def get_kind_name(self) -> str:
        """Gets the kind name. """

        return self.kindName_

    def get_name(self) -> str:
        """Gets the node name. """

        return self.name_

    def get_inputs(self) -> List[NodeValue]:
        """Gets the input node. """

        return self.inputs_

    def get_users(self) -> Dict['Node', int]:
        """Gets the user of this node. """

        return self.users_

    def add_user(self, u: 'Node') -> None:
        """Adds one user of this node. Increment the number of uses of the user by 1. """

        if u not in self.users_:
            self.users_[u] = 0
        self.users_[u] += 1

    def remove_user(self, u: 'Node') -> None:
        """Removes one use from the given user."""

        assert u in self.users_
        self.users_[u] -= 1
        if self.users_[u] == 0:
            del self.users_[u]

    def has_no_uses(self) -> bool:
        """Returns True if the node has no uses."""

        return len(self.users_) == 0

    def set_input(self, nodeVal: NodeValue) -> None:
        """Adds one input node value."""

        self.inputs_.append(nodeVal)

    def replace_input(
            self,
            oldNodeVal: NodeValue,
            newNodeVal: NodeValue) -> None:
        """Replace one operand with another one.

        Args:
            oldNode: Node. Old operand node.
            oldResNo: int. Old operand result number.
            newNode: Node. New operand node.
            newResNo: int. New operand result number.
        """

        try:
            self.inputs_.remove(oldNodeVal)
        except ValueError:
            print("Removed input value must already exist in the node's input list. ")
        self.inputs_.append(newNodeVal)

    def set_scope_of_creation(self, creationScopeName: str) -> None:
        self.creationScopeName_ = creationScopeName


class DottyPrinter:
    """A class for generating the dotty graph file

    Public attributes:
        vertices_: List[str]. Vertices in the dotty file.
        edges_: List[str]. Edges in the dotty file.
        uniqueVertexMap_: Dict[Node, int]. A map for node with their unique index.
        uniqueVertexNo_: int. A incrementing number that represents the number of unique nodes in the graph.
        colors_: List[str]. A list for colors for nodes in the dotty graph.
    """

    def __init__(self):
        self.vertices_: List[str] = []
        self.edges_: List[str] = []
        self.uniqueVertexMap_: Dict[Node, int] = {}
        self.uniqueVertexNo_: int = 0
        self.colors_: List[str] = [
            "AliceBlue",
            "CadetBlue1",
            "Coral",
            "DarkOliveGreen1",
            "DarkSeaGreen1",
            "GhostWhite",
            "Khaki1",
            "LavenderBlush1",
            "LemonChiffon1",
            "LightSkyBlue",
            "MistyRose1",
            "MistyRose2",
            "PaleTurquoise2",
            "PeachPuff1",
            "PowderBlue",
            "Salmon",
            "Thistle1",
            "Thistle3",
            "Wheat1",
            "Yellow2",
        ]

    def get_unique_vertex_name(self, node: Node) -> str:
        """Get the unique vertex name given a Node object. """

        if node not in self.uniqueVertexMap_:
            self.uniqueVertexMap_[node] = self.uniqueVertexNo_
            self.uniqueVertexNo_ += 1

        return f"v{self.uniqueVertexMap_[node]}"

    def dump_label(self, node: Node) -> str:
        """Returns the string for the label of the given node. """

        labelStr = f"""{{ {{<Inputs>Inputs}}|
                    {{ {node.get_kind_name()}\lname: {node.get_name()} }}|
                    {{<Outputs>Outputs}} }}"""
        return labelStr

    def get_color(self, node: Node) -> str:
        """Returns the color for the given node. """

        idx = hash(node.get_kind_name()) % len(self.colors_)
        return self.colors_[idx]

    def dump_node(self, node: Node) -> None:
        """Generates the dotty information for the given node. """

        if not node:
            return

        nodeStr = f"""{self.get_unique_vertex_name(node)}[\n
                    \tlabel = \"{self.dump_label(node)}\"\n
                    \tshape = \"record\"\n
                    \tstyle=\"filled,rounded\"\n
                    \tfillcolor={self.get_color(node)}\n
                    penwidth = 2];\n"""
        self.vertices_.append(nodeStr)

    def visitNodes(self) -> None:
        """Visits all nodes in NODES_MAP and dump the dotty information for each node. """

        for node in NODES_MAP.values():
            self.dump_node(node)

    def visitEdges(self) -> None:
        """Visits all edges and dump the dotty information for each edge. """

        for node in NODES_MAP.values():
            for nodeInput in node.get_inputs():
                i = nodeInput[0]
                if (i.get_kind_name(), i.get_name()) not in NODES_MAP:
                    print(i.get_kind_name(), i.get_name())
                edgeStr = self.get_unique_vertex_name(i) + ":Outputs -> "
                edgeStr += self.get_unique_vertex_name(node) + ":Inputs"
                self.edges_.append(edgeStr)

    def dump_graph(self, dagName: str) -> None:
        """Visits the node graph and generates the dotty information. """

        self.visitNodes()
        self.visitEdges()
        with open(f"{dagName}_dotty.dot", "w") as f:
            f.write("digraph DAG {\n\trankdir=TB;\n")
            for v in self.vertices_:
                f.write(f"{v}\n")
            for e in self.edges_:
                f.write(f"{e};\n")
            f.write("}")


def parse_args() -> Tuple[str, List[str]]:
    """Parse the arguments of this script. """

    parser = argparse.ArgumentParser(description="Parse compilation log")
    parser.add_argument("-f", "--log-file")
    parser.add_argument("--dump-phases", nargs="+")
    options = parser.parse_args()

    if options.dump_phases:
        dumpPhases = options.dump_phases
    else:
        dumpPhases = []

    return options.log_file, dumpPhases


def dump_dag(dagName: str) -> None:
    dotty = DottyPrinter()
    dotty.dump_graph(dagName)


def process(logLines: List[str], dumpPhases: List[str]) -> None:
    """Process all the log lines.

    Extract their information and reconstruct the node graph. And dump DAGs at given compilation phases.

    Args:
        logLines: List[str]. All lines of compilation log.
        dumpPhases: List[str]. The phase at which to dump the DAG.
    """

    phaseNo = 0
    pCreate = re.compile(
        '^.*\[FULL.*SCOPE:(.*)\].*CREATE.*\{.*\(Kind:(.*),.*Name:(.*)\).*<==(.*)\}$')
    pInputs = re.compile(r'\([^\(\)]*\)')
    pOneInput = re.compile(r'^\(Kind:(.*),.*Name:(.*),.*ResNo:(.*)\)$')
    pChange = re.compile(
        '^.*\[FULL.*SCOPE:(.*)\].*NODE_INPUT_CHANGE.*\{.*User\(Kind:(.*),.*Name:(.*)\).*::.*PrevOprValue(.*)->.*NewOprValue(.*)\}$')
    pOpr = re.compile(r'^\(.*Kind:(.*),.*Name:(.*),.*ResNo:(.*)\)$')
    pDelete = re.compile(
        '^.*\[FULL.*SCOPE:(.*)\].*DELETE.*\(Kind:(.*),.*Name:(.*)\).*}$')
    pEnter = re.compile('^ENTERSCOPE:(.*)$')
    pExit = re.compile('^EXITSCOPE:(.*)$')

    for ln in logLines:
        # Process CREATE statement
        if "CREATE" in ln:
            m = re.match(pCreate, ln)
            if m:
                g = m.groups()
                # Create the node with given kind and name
                scopeName = g[0]
                kindName = g[1].replace(" ", "")
                nodeName = g[2].replace(" ", "")
                createdNode = Node(kindName, nodeName)
                createdNode.set_scope_of_creation(scopeName)
                NODES_MAP[NodeNameAndKind(kindName, nodeName)] = createdNode

                # Set the inputs of the created node
                inputs = re.findall(pInputs, g[3])
                if len(inputs) == 0:
                    # there's no node input for Splat
                    assert kindName in (
                        "Splat", "Constant", "Placeholder"), "This node kind shouldn't have any inputs."
                for i in inputs:
                    mi = re.match(pOneInput, i)
                    if mi:
                        gi = mi.groups()
                        inputNodeKind, inputNodeName, inputNodeResno = [
                            x.replace(" ", "") for x in gi[:3]]
                        assert (
                            inputNodeKind, inputNodeName) in NODES_MAP, "Input nodes must already exist in node graph."
                        # (node, resno)
                        inputNode = NODES_MAP[NodeNameAndKind(
                            inputNodeKind, inputNodeName)]
                        createdNode.set_input(
                            NodeValue(inputNode, inputNodeResno))
                        inputNode.add_user(createdNode)

        # Process NODE_INPUT_CHANGE statement
        elif "NODE_INPUT_CHANGE" in ln:
            m = re.match(pChange, ln)
            if m:
                g = m.groups()
                scopeName, kindName, nodeName = [
                    x.replace(" ", "") for x in g[:3]]
                assert (
                    kindName, nodeName) in NODES_MAP, "This node must already exist in node graph."
                changedNode = NODES_MAP[NodeNameAndKind(kindName, nodeName)]

                # Don't touch the line of node input changing into null, it only happened
                # in module destructor.
                if "null" in g[4]:
                    continue

                mPrev = re.match(pOpr, g[3].replace(" ", ""))
                mNew = re.match(pOpr, g[4].replace(" ", ""))

                # Previous operand
                prevNodeKind, prevNodeName, prevNodeResno = mPrev.groups()[:3]
                assert (
                    prevNodeKind, prevNodeName) in NODES_MAP, "Node's operand must already exist in node graph."
                prevNode = NODES_MAP[NodeNameAndKind(
                    prevNodeKind, prevNodeName)]

                # New operand
                newNodeKind, newNodeName, newNodeResno = mNew.groups()[:3]
                assert (
                    newNodeKind, newNodeName) in NODES_MAP, "Node's operand must already exist in node graph."
                newNode = NODES_MAP[NodeNameAndKind(newNodeKind, newNodeName)]

                # change the input of changedNode
                changedNode.replace_input(
                    NodeValue(
                        prevNode, prevNodeResno), NodeValue(
                        newNode, newNodeResno))
                prevNode.remove_user(changedNode)
                newNode.add_user(changedNode)

        # Process DELETE statement
        elif "DELETE" in ln:
            m = re.match(pDelete, ln)
            # Deleted node
            if m:
                g = m.groups()
                scopeName, kindName, nodeName = [
                    x.replace(" ", "") for x in g[:3]]
                assert (
                    kindName, nodeName) in NODES_MAP, "Deleted node must already exist in node graph."
                deletedNode = NODES_MAP[NodeNameAndKind(kindName, nodeName)]
                if not deletedNode.has_no_uses():
                    assert "Destructor" in ln or scopeName == ""
                for inputNode in deletedNode.inputs_:
                    i = inputNode[0]
                    i.remove_user(deletedNode)
                del NODES_MAP[NodeNameAndKind(kindName, nodeName)]

        # Process ENTER SCOPE statement
        elif "ENTER SCOPE:" in ln:
            ln = ln.replace("=", '').replace(" ", "")
            m = re.match(pEnter, ln)
            if m:
                phaseNo += 1
                fullScopeName = m.groups()[0]
                lastScopeName = fullScopeName.split("->")[-1]
                if "::" in lastScopeName:
                    lastScopeName = lastScopeName.split("::")[-1]
                if (lastScopeName in dumpPhases):
                    dump_dag("before_" + lastScopeName + "_" + str(phaseNo))
                SCOPE_STACK.append(lastScopeName)

        # Process EXIT SCOPE statement
        elif "EXIT SCOPE:" in ln:
            ln = ln.replace("=", '').replace(" ", "")
            m = re.match(pExit, ln)
            if m:
                fullScopeName = m.groups()[0]
                lastScopeName = fullScopeName.split("->")[-1]
                if "::" in lastScopeName:
                    lastScopeName = lastScopeName.split("::")[-1]
                assert lastScopeName == SCOPE_STACK[-1], "Exited scope must be same as the top of scope stack."
                if (lastScopeName in dumpPhases):
                    dump_dag("after_" + lastScopeName + "_" + str(phaseNo))
                SCOPE_STACK.pop()


def main():
    logFile, dumpPhases = parse_args()
    lines = filter(lambda x: len(x) > 0, open(logFile).readlines())
    process(lines, dumpPhases)
    return


if __name__ == "__main__":
    main()
