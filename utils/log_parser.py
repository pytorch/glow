#!/usr/bin/env python3
# Copyright (c) Glow Contributors. See CONTRIBUTORS file.
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
from typing import List, Dict, Tuple
import typing
import sqlite3
import os

# Maintaining all nodes
NODES_MAP: Dict["NodeNameAndKind", "Node"] = {}

# Scope stack
SCOPE_STACK: List[str] = []


class NodeNameAndKind(typing.NamedTuple):
    """A class that represents the named tuple of node name and kind."""

    name: str
    kind: str


class NodeValue(typing.NamedTuple):
    """A class that represents the named tuple of node and result number."""

    node: "Node"
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
        self.users_: Dict["Node", int] = {}

    def get_kind_name(self) -> str:
        """Gets the kind name. """

        return self.kindName_

    def get_name(self) -> str:
        """Gets the node name. """

        return self.name_

    def get_inputs(self) -> List[NodeValue]:
        """Gets the input node. """

        return self.inputs_

    def get_users(self) -> Dict["Node", int]:
        """Gets the user of this node. """

        return self.users_

    def add_user(self, u: "Node") -> None:
        """Adds one user of this node. Increment the number of uses of the user by 1. """

        if u not in self.users_:
            self.users_[u] = 0
        self.users_[u] += 1

    def remove_user(self, u: "Node") -> None:
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

    def replace_input(self, oldNodeVal: NodeValue,
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

    def __init__(self, nodesMap: Dict[NodeNameAndKind, Node]):
        self.nodesMap_ = nodesMap
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
        """Visits all nodes in nodesMap_ and dump the dotty information for each node. """

        for node in self.nodesMap_.values():
            self.dump_node(node)

    def visitEdges(self) -> None:
        """Visits all edges and dump the dotty information for each edge. """

        for node in self.nodesMap_.values():
            for nodeInput in node.get_inputs():
                i = nodeInput[0]
                if (i.get_kind_name(), i.get_name()) not in self.nodesMap_:
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


def parse_args() -> Tuple[str, str, List[str]]:
    """Parse the arguments of this script. """

    parser = argparse.ArgumentParser(description="Parse compilation log")
    parser.add_argument("-f", "--log-file")
    parser.add_argument("-d", "--db-file")
    parser.add_argument("--dump-phases", nargs="+")
    options = parser.parse_args()

    if options.dump_phases:
        dumpPhases = options.dump_phases
    else:
        dumpPhases = []

    if options.db_file:
        dbFile = options.db_file
    else:
        dbFile = "compilation_log_db.sqlite"

    return dbFile, options.log_file, dumpPhases


def dump_dag(dagName: str) -> None:
    """A helper function to dump the DAG."""

    dotty = DottyPrinter(NODES_MAP)
    dotty.dump_graph(dagName)


def store_transformation_into_DB(
    transID: int,
    baseNode: Node,
    addedNodes: List[Node],
    replacedNodes: List[Node],
    cursor: sqlite3.Cursor,
    fullScopeName: str,
) -> None:
    """A helper function to store nodes transformations into database.

    Args:
        transID: int. The ID for this stored transformation.
        baseNode: Node. The base node that changes its operands.
        addedNodes: List[Node]. A list of added nodes in this transformation.
        replacedNodes: List[Node]. A list of replaced nodes in this transformation.
        cursor: sqlite3.Cursor. Cursor of the sqlite3 database.
        fullScopeName: str. The full scope name of this transformation.
    """

    cursor.execute(
        """INSERT INTO Log_Transformation VALUES (
                        ?,
                        'OPERATOR_BASE',
                        ?,
                        ?,
                        ?
                        )""",
        (transID, baseNode.get_name(), baseNode.get_kind_name(), fullScopeName),
    )
    for an in addedNodes:
        cursor.execute(
            """INSERT INTO Log_Transformation VALUES (
                        ?,
                        'ADD_OPERAND',
                        ?,
                        ?,
                        ?
                        )""",
            (transID, an.get_name(), an.get_kind_name(), fullScopeName),
        )

    for rn in replacedNodes:
        cursor.execute(
            """INSERT INTO Log_Transformation VALUES (
                        ?,
                        'REMOVE_OPERAND',
                        ?,
                        ?,
                        ?
                        )""",
            (transID, rn.get_name(), rn.get_kind_name(), fullScopeName),
        )


def find_all_replaced_nodes(replacedNode: Node) -> List[Node]:
    """Find all nodes that will lose user after the given node is removed.

    After one node lost all its uses (e.g. after replaceAllUsesOfWith()), we go through
    all of its parents to collect all nodes that will consequently lose all their uses.

    Args:
        replacedNode: Node. The node that just lost all uses.
    """

    replacedNodeList = []
    activeDCEList = [replacedNode]
    while len(activeDCEList):
        DCEnode = activeDCEList.pop()
        replacedNodeList.append(DCEnode)
        for nv in DCEnode.inputs_:
            n = nv.node
            if len(n.users_) <= 1:
                activeDCEList.append(n)

    return replacedNodeList


def init_db(sqliteFile: str) -> sqlite3.Connection:
    """Initialize a sqlite3 database connection."""

    if os.path.isfile(sqliteFile):
        os.remove(sqliteFile)

    # Connect to database file.
    conn = sqlite3.connect(sqliteFile)
    cursor = conn.cursor()
    cursor.execute(
        """CREATE TABLE Log_Transformation (
                    trans_id INTEGER,
                    operation_type VARCHAR(200),
                    node_name VARCHAR(200),
                    node_kind VARCHAR(200),
                    full_scope VARCHAR(200)
                    )"""
    )

    cursor.execute(
        """CREATE TABLE Log_Scope (
                    scope_id INTEGER,
                    scope_str VARCHAR(200),
                    full_scope_str VARCHAR(200)
                    )"""
    )

    cursor.execute(
        """CREATE TABLE Log_Node (
                    node_name VARCHAR(200),
                    node_kind VARCHAR(200),
                    create_scope_id INTEGER,
                    delete_scope_id INTEGER
                    )"""
    )

    cursor.execute(
        """CREATE TABLE Log_Node_Operation (
                    scope_id INTEGER,
                    operation VARCHAR(200),
                    node_name VARCHAR(200),
                    node_kind VARCHAR(200)
                    )"""
    )

    return conn


def process(
    logLines: List[str], dumpPhases: List[str], conn: sqlite3.Connection
) -> None:
    """Process all the log lines.

    Extract their information and reconstruct the node graph. And dump DAGs at given compilation phases.

    Args:
        logLines: List[str]. All lines of compilation log.
        dumpPhases: List[str]. The phase at which to dump the DAG.
        conn: sqlite3.Connection. The connection to a sqlite3 database that will store all the transformation in the compilation lop.
    """

    # Regular expression patterns
    pCreate = re.compile(
        r"^.*\[FULL.*SCOPE:(.*)\].*CREATE.*\{.*\(Kind:(.*),.*Name:(.*)\).*<==(.*)\}$"
    )
    pInputs = re.compile(r"\([^\(\)]*\)")
    pOneInput = re.compile(r"^\(Kind:(.*),.*Name:(.*),.*ResNo:(.*)\)$")
    pChange = re.compile(
        r"^.*\[FULL.*SCOPE:(.*)\].*NODE_INPUT_CHANGE.*\{.*User\(Kind:(.*),.*Name:(.*)\).*::.*PrevOprValue(.*)->.*NewOprValue(.*)\}$"
    )
    pOpr = re.compile(r"^\(.*Kind:(.*),.*Name:(.*),.*ResNo:(.*)\)$")
    pDelete = re.compile(
        r"^.*\[FULL.*SCOPE:(.*)\].*DELETE.*\(Kind:(.*),.*Name:(.*)\).*}$"
    )
    pEnter = re.compile(r"^ENTERSCOPE:(.*)$")
    pExit = re.compile(r"^EXITSCOPE:(.*)$")

    # DB related vars
    cursor = conn.cursor()

    # Record nodes transformation
    replacedNodes: List[Node] = []
    addedNodes: List[Node] = []
    recordTransformation = False
    stopRecordTranformationNames = {
        "optimizeFunctionBeforeLowering",
        "optimizeFunction",
    }
    transID = 0

    # Scope related information
    scopeID = 0

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

                # Update node creation in database
                cursor.execute(
                    """INSERT INTO Log_Node VALUES (
                        ?,
                        ?,
                        ?,
                        ?
                        )""",
                    (nodeName, kindName, scopeID, -1),
                )
                cursor.execute(
                    """INSERT INTO Log_Node_Operation VALUES (
                        ?,
                        'CREATE',
                        ?,
                        ?
                        )""",
                    (scopeID, nodeName, kindName),
                )

                # Set the inputs of the created node
                inputs = re.findall(pInputs, g[3])
                if len(inputs) == 0:
                    # there's no node input for Splat
                    assert kindName in (
                        "Splat",
                        "Constant",
                        "Placeholder",
                    ), "This node kind shouldn't have any inputs."
                for i in inputs:
                    mi = re.match(pOneInput, i)
                    if mi:
                        gi = mi.groups()
                        inputNodeKind, inputNodeName, inputNodeResno = [
                            x.replace(" ", "") for x in gi[:3]
                        ]
                        assert (
                            inputNodeKind,
                            inputNodeName,
                        ) in NODES_MAP, "Input nodes must already exist in node graph."
                        # (node, resno)
                        inputNode = NODES_MAP[
                            NodeNameAndKind(inputNodeKind, inputNodeName)
                        ]
                        createdNode.set_input(
                            NodeValue(inputNode, inputNodeResno))
                        inputNode.add_user(createdNode)

                # Record nodes transformation
                if recordTransformation:
                    addedNodes.append(createdNode)

        # Process NODE_INPUT_CHANGE statement
        elif "NODE_INPUT_CHANGE" in ln:
            m = re.match(pChange, ln)
            if m:
                g = m.groups()
                scopeName, kindName, nodeName = [
                    x.replace(" ", "") for x in g[:3]]
                assert (
                    kindName,
                    nodeName,
                ) in NODES_MAP, "This node must already exist in node graph."
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
                    prevNodeKind,
                    prevNodeName,
                ) in NODES_MAP, "Node's operand must already exist in node graph."
                prevNode = NODES_MAP[NodeNameAndKind(
                    prevNodeKind, prevNodeName)]

                # New operand
                newNodeKind, newNodeName, newNodeResno = mNew.groups()[:3]
                assert (
                    newNodeKind,
                    newNodeName,
                ) in NODES_MAP, "Node's operand must already exist in node graph."
                newNode = NODES_MAP[NodeNameAndKind(newNodeKind, newNodeName)]

                # change the input of changedNode
                changedNode.replace_input(
                    NodeValue(
                        prevNode, prevNodeResno), NodeValue(
                        newNode, newNodeResno)
                )
                prevNode.remove_user(changedNode)
                newNode.add_user(changedNode)

                # Record nodes transformation
                if recordTransformation:
                    if prevNode.has_no_uses():
                        replacedNodes = find_all_replaced_nodes(prevNode)
                        store_transformation_into_DB(
                            transID,
                            changedNode,
                            addedNodes,
                            replacedNodes,
                            cursor,
                            scopeName,
                        )

                        transID += 1
                        addedNodes = []
                        replacedNodes = []

        # Process DELETE statement
        elif "DELETE" in ln:
            m = re.match(pDelete, ln)
            # Deleted node
            if m:
                g = m.groups()
                scopeName, kindName, nodeName = [
                    x.replace(" ", "") for x in g[:3]]
                assert (
                    kindName,
                    nodeName,
                ) in NODES_MAP, "Deleted node must already exist in node graph."
                deletedNode = NODES_MAP[NodeNameAndKind(kindName, nodeName)]
                if not deletedNode.has_no_uses():
                    assert "Destructor" in ln or scopeName == ""
                for inputNode in deletedNode.inputs_:
                    i = inputNode[0]
                    i.remove_user(deletedNode)
                del NODES_MAP[NodeNameAndKind(kindName, nodeName)]

                # Update node deletion in database
                cursor.execute(
                    """UPDATE Log_Node
                        SET delete_scope_id=?
                        WHERE node_name=?
                        """,
                    (scopeID, nodeName),
                )
                cursor.execute(
                    """INSERT INTO Log_Node_Operation VALUES (
                        ?,
                        'DELETE',
                        ?,
                        ?
                        )""",
                    (scopeID, nodeName, kindName),
                )

        # Process ENTER SCOPE statement
        elif "ENTER SCOPE:" in ln:
            ln = ln.replace("=", "").replace(" ", "")
            m = re.match(pEnter, ln)
            if m:
                scopeID += 1
                fullScopeName = m.groups()[0]
                lastScopeName = fullScopeName.split("->")[-1]
                if "::" in lastScopeName:
                    lastScopeName = lastScopeName.split("::")[-1]
                if lastScopeName in dumpPhases:
                    dump_dag(f"before_{lastScopeName}_{scopeID}")
                if str(scopeID) in dumpPhases:
                    dump_dag(f"phase_{scopeID}")
                SCOPE_STACK.append(lastScopeName)

                # Start recording transformations.
                if (
                    lastScopeName in stopRecordTranformationNames
                    and len(SCOPE_STACK) == 1
                ):
                    recordTransformation = True

                # Update scope entrance in database
                cursor.execute(
                    """INSERT INTO Log_Scope VALUES (
                        ?,
                        ?,
                        ?
                        )""",
                    (scopeID,
                     "ENTER " + lastScopeName,
                     "ENTER " + fullScopeName),
                )

        # Process EXIT SCOPE statement
        elif "EXIT SCOPE:" in ln:
            ln = ln.replace("=", "").replace(" ", "")
            m = re.match(pExit, ln)
            if m:
                scopeID += 1
                fullScopeName = m.groups()[0]
                lastScopeName = fullScopeName.split("->")[-1]
                if "::" in lastScopeName:
                    lastScopeName = lastScopeName.split("::")[-1]
                assert (
                    lastScopeName == SCOPE_STACK[-1]
                ), "Exited scope must be same as the top of scope stack."
                if lastScopeName in dumpPhases:
                    dump_dag(f"after_{lastScopeName}_{scopeID}")
                if str(scopeID) in dumpPhases:
                    dump_dag(f"phase_{scopeID}")
                SCOPE_STACK.pop()

                # Stop recording transformations.
                if (
                    lastScopeName in stopRecordTranformationNames
                    and len(SCOPE_STACK) == 0
                ):
                    recordTransformation = False

                # Update scope exit in database
                cursor.execute(
                    """INSERT INTO Log_Scope VALUES (
                        ?,
                        ?,
                        ?
                        )""",
                    (scopeID, "EXIT " + lastScopeName, "EXIT " + fullScopeName),
                )

    # Update scope db
    cursor.execute(
        """INSERT INTO Log_Scope VALUES (
                        0,
                        'MODULE LOADER',
                        'MODULE LOADER'
                        )"""
    )

    conn.commit()


def main():
    dbFile, logFile, dumpPhases = parse_args()
    lines = filter(lambda x: len(x) > 0, open(logFile).readlines())
    with init_db(dbFile) as conn:
        process(lines, dumpPhases, conn)
    return


if __name__ == "__main__":
    main()
