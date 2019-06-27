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

# Maintaining all nodes
NODES_MAP = {}

# Maintaining all edges
EDGES_MAP = {}

# Scope stack
SCOPE_STACK = []


class Node:
    def __init__(self, kindName, name):
        self.kindName_ = kindName
        self.name_ = name
        self.inputs_ = []
        self.users_ = {}

    def get_kind_name(self):
        return self.kindName_

    def get_name(self):
        return self.name_

    def get_inputs(self):
        return self.inputs_

    def get_users(self):
        return self.users_

    def add_user(self, u):
        if u not in self.users_:
            self.users_[u] = 0
        self.users_[u] += 1

    def remove_user(self, u):
        assert u in self.users_
        self.users_[u] -= 1
        if self.users_[u] == 0:
            del self.users_[u]

    def has_no_uses(self):
        return len(self.users_) == 0

    def set_input(self, node, resNo):
        self.inputs_.append((node, resNo))

    def replace_input(self, oldNode, oldResNo, newNode, newResNo):
        assert (oldNode, oldResNo) in self.inputs_
        tmp_inputs = [(newNode, newResNo)]
        for n, rn in self.inputs_:
            if n != oldNode or rn != oldResNo:
                tmp_inputs.append((n, rn))
        self.inputs_ = tmp_inputs

    def set_scope_of_creation(self, creationScopeName):
        self.creationScopeName_ = creationScopeName


class DottyPrinter:
    def __init__(self):
        self.vertices_ = []
        self.edges_ = []
        self.uniqueVertexMap = {}
        self.uniqueVertexNo = 0
        self.colors = ["AliceBlue", "CadetBlue1", "Coral", "DarkOliveGreen1", "DarkSeaGreen1", "GhostWhite", "Khaki1", "LavenderBlush1", "LemonChiffon1",
                       "LightSkyBlue", "MistyRose1", "MistyRose2", "PaleTurquoise2", "PeachPuff1", "PowderBlue", "Salmon", "Thistle1", "Thistle3", "Wheat1", "Yellow2"]

    def get_unique_vertex_name(self, node):
        if node not in self.uniqueVertexMap:
            self.uniqueVertexMap[node] = self.uniqueVertexNo
            self.uniqueVertexNo += 1
        return "v" + str(self.uniqueVertexMap[node])

    def dump_label(self, node):
        labelStr = "{"
        labelStr += " {<Inputs>Inputs}|"
        labelStr += "{" + node.get_kind_name() + "\lname: " + \
            node.get_name() + "}|"
        labelStr += " {<Outputs>Outputs}"
        labelStr += "}"
        return labelStr

    def get_color(self, node):
        idx = hash(node.get_kind_name()) % len(self.colors)
        return self.colors[idx]

    def dump_node(self, node):
        if not node:
            return
        nodeStr = self.get_unique_vertex_name(node) + "[\n"
        nodeStr += "\tlabel = \""
        nodeStr += self.dump_label(node)
        nodeStr += "\"\n"
        nodeStr += "\tshape = \"record\"\n"
        nodeStr += "\tstyle=\"filled,rounded\"\n"
        nodeStr += "\tfillcolor="
        nodeStr += self.get_color(node)
        nodeStr += "\npenwidth = 2];\n"
        self.vertices_.append(nodeStr)

    def visitNodes(self):
        for k, n in NODES_MAP:
            self.dump_node(NODES_MAP[(k, n)])

    def visitEdges(self):
        for k, n in NODES_MAP:
            node = NODES_MAP[(k, n)]
            for nodeInput in node.get_inputs():
                i = nodeInput[0]
                if (i.get_kind_name(), i.get_name()) not in NODES_MAP:
                    print(i.get_kind_name(), i.get_name())
                edgeStr = self.get_unique_vertex_name(i)+":Outputs -> "
                edgeStr += self.get_unique_vertex_name(node)+":Inputs"
                self.edges_.append(edgeStr)

    def dump_graph(self, dagName):
        self.visitNodes()
        self.visitEdges()
        f = open(dagName + "_dotty.dot", "w")
        f.write("digraph DAG {\n\trankdir=TB;\n")
        for v in self.vertices_:
            f.write(v + "\n")
        for e in self.edges_:
            f.write(e + ";\n")
        f.write("}")
        f.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Parse compilation log")
    parser.add_argument("-f", "--log-file")
    parser.add_argument("--dump-phases", nargs="+")
    options = parser.parse_args()

    if options.dump_phases:
        dumpPhases = options.dump_phases
    else:
        dumpPhases = []

    return options.log_file, dumpPhases


def dump_dag(dagName):
    dotty = DottyPrinter()
    dotty.dump_graph(dagName)


def process(logLines, dumpPhases):
    phaseNo = 0
    pCreate = re.compile(
        '^.*\[FULL.*SCOPE:(.*)\].*CREATE.*\{.*\(Kind:(.*),.*Name:(.*)\).*<==(.*)\}$')
    pInputs = re.compile('\([^\(\)]*\)')
    pOneInput = re.compile('^\(Kind:(.*),.*Name:(.*),.*ResNo:(.*)\)$')
    pChange = re.compile(
        '^.*\[FULL.*SCOPE:(.*)\].*NODE_INPUT_CHANGE.*\{.*User\(Kind:(.*),.*Name:(.*)\).*::.*PrevOprValue(.*)->.*NewOprValue(.*)\}$')
    pOpr = re.compile('^\(.*Kind:(.*),.*Name:(.*),.*ResNo:(.*)\)$')
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
                assert len(g) == 4

                # Create the node with given kind and name
                scopeName = g[0]
                kindName = g[1].replace(" ", "")
                nodeName = g[2].replace(" ", "")
                createdNode = Node(kindName, nodeName)
                createdNode.set_scope_of_creation(scopeName)
                NODES_MAP[(kindName, nodeName)] = createdNode

                # Set the inputs of the created node
                inputs = re.findall(pInputs, g[3])
                if len(inputs) == 0:
                    # there's no node input for Splat
                    assert kindName in ("Splat", "Constant", "Placeholder")
                for i in inputs:
                    mi = re.match(pOneInput, i)
                    if mi:
                        gi = mi.groups()
                        inputNodeKind, inputNodeName, inputNodeResno = [
                            x.replace(" ", "") for x in gi[:3]]
                        assert (inputNodeKind, inputNodeName) in NODES_MAP
                        # (node, resno)
                        inputNode = NODES_MAP[(inputNodeKind, inputNodeName)]
                        createdNode.set_input(inputNode, inputNodeResno)
                        inputNode.add_user(createdNode)

        # Process NODE_INPUT_CHANGE statement
        elif "NODE_INPUT_CHANGE" in ln:
            m = re.match(pChange, ln)
            if m:
                g = m.groups()
                scopeName, kindName, nodeName = [
                    x.replace(" ", "") for x in g[:3]]
                assert (kindName, nodeName) in NODES_MAP
                changedNode = NODES_MAP[(kindName, nodeName)]

                # Don't touch the line of node input changing into null, it only happened
                # in module destructor.
                if "null" in g[4]:
                    continue

                mPrev = re.match(pOpr, g[3].replace(" ", ""))
                mNew = re.match(pOpr, g[4].replace(" ", ""))

                # Previous operand
                assert mPrev
                prevNodeKind, prevNodeName, prevNodeResno = mPrev.groups()[:3]
                assert (prevNodeKind, prevNodeName) in NODES_MAP
                prevNode = NODES_MAP[(prevNodeKind, prevNodeName)]

                # New operand
                assert mNew
                newNodeKind, newNodeName, newNodeResno = mNew.groups()[:3]
                assert (newNodeKind, newNodeName) in NODES_MAP
                newNode = NODES_MAP[(newNodeKind, newNodeName)]

                # change the input of changedNode
                changedNode.replace_input(
                    prevNode, prevNodeResno, newNode, newNodeResno)
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
                assert (kindName, nodeName) in NODES_MAP
                deletedNode = NODES_MAP[(kindName, nodeName)]
                if not deletedNode.has_no_uses():
                    assert "Destructor" in ln or scopeName == ""
                for inputNode in deletedNode.inputs_:
                    i = inputNode[0]
                    i.remove_user(deletedNode)
                del NODES_MAP[(kindName, nodeName)]

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
                assert lastScopeName == SCOPE_STACK[-1]
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
