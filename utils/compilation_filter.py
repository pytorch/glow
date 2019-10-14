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
import sqlite3
import os
from typing import List, Dict

# A list of all filtered transformations.
TRANS_LIST: List["Transformation"] = []

# Mapping between added nodes and the transformation that adds these nodes.
NODES_ADDING_MAP: Dict[str, "Transformation"] = {}


class Transformation:
    """A class that represents the nodes transformation, e.g. lower,fold etc.

    Public attributes:
        addedNodes_: List[str]. Nodes added by this transformation.
        removedNodes_: List[str]. Nodes removed by this transformation.
        ancestors_: List['Transformation']. The ancestor transformation of current transformation.
        scopeName_: str. The scope of current transformation.
        transID_: str. The internal transformation ID in the database.
        isDirectTrans_ :bool. Whether this transformation directly created/replaced the given nodeName that is passed to this script file.
    """

    def __init__(self, transID: str):
        self.addedNodes_: List[str] = []
        self.removedNodes_: List[str] = []
        self.ancestors_: List["Transformation"] = []
        self.scopeName_: str = ""
        self.transID_: str = transID
        self.isDirectTrans_: bool = False

    def appendAddedNode(self, nodeName: str) -> None:
        """Append the added nodes of this transformation."""

        self.addedNodes_.append(nodeName)

    def appendRemovedNode(self, nodeName: str) -> None:
        """Append the removed nodes of this transformation."""

        self.removedNodes_.append(nodeName)

    def addAncestor(self, ancestor: "Transformation") -> None:
        """Add ancestors of this transformation."""

        self.ancestors_.append(ancestor)

    def setBase(self, baseName: str) -> None:
        """Set the operator base of this node."""

        self.baseNode_ = baseName


class DottyPrinter:
    """A class for generating the dotty graph file"""

    def __init__(self, transList: List[Transformation]):
        self.transList_ = transList
        self.vertices_ = []
        self.edges_ = []

    def get_color(self, isDirectTrans: bool) -> str:
        """Returns the color for the given node. """

        if isDirectTrans:
            return "Yellow2"
        else:
            return "AliceBlue"

    def dump_label(self, tran: Transformation) -> str:
        """Returns the string for the label of the given transformation. """

        labelStr = (
            rf"""{{ {{SCOPE:\l{tran.scopeName_} }}|{{ORIGINAL OPERAND CHAIN:\l\l"""
        )
        for rstr in tran.removedNodes_:
            labelStr += rf"""{rstr}\l\l"""
        labelStr += rf"}}| {{NEW OPERAND CHAIN:\l\l"
        for astr in tran.addedNodes_:
            labelStr += rf"""{astr}\l\l"""
        labelStr += rf"}} |{{USER NODE: \l\l {tran.baseNode_}}} }}"
        return labelStr

    def dump_node(self, tran: Transformation) -> None:
        """Generates the dotty information for the given transformation. """

        if not tran:
            return

        tranStr = f"""v{tran.transID_}[\n
                    \tlabel = \"{self.dump_label(tran)}\"\n
                    \tshape = \"record\"\n
                    \tstyle=\"filled,rounded\"\n
                    \tfillcolor={self.get_color(tran.isDirectTrans_)}\n
                    penwidth = 2];\n"""
        self.vertices_.append(tranStr)

    def visit_nodes(self) -> None:
        """Visits all transformation and dump the dotty information for each transformation. """

        for tran in self.transList_:
            self.dump_node(tran)

    def visit_edges(self) -> None:
        """Visits all edges and dump the dotty information for each edge. """

        for tran in self.transList_:
            for anc in tran.ancestors_:
                edgeStr = f"v{anc.transID_} -> v{tran.transID_}"
                self.edges_.append(edgeStr)

    def dump_graph(self, dottyFile: str) -> None:
        """Visits the graph and generates the dotty information. """

        self.visit_nodes()
        self.visit_edges()
        with open(f"transformations_{dottyFile}.dot", "w") as f:
            print(
                f"\nWriting DAG info into dotty file transformations_{dottyFile}.dot ..."
            )
            f.write("digraph DAG {\n\trankdir=TB;\n")
            for v in self.vertices_:
                f.write(f"{v}\n")
            for e in self.edges_:
                f.write(f"{e};\n")
            f.write("}")


def dump_dotty_DAG(dottyFile: str) -> None:
    """A helper function to dump the dotty file."""

    dotty = DottyPrinter(TRANS_LIST)
    dotty.dump_graph(dottyFile)


def init_db(sqliteFile: str) -> sqlite3.Connection:
    """Initialize a sqlite3 database connection."""

    assert os.path.isfile(sqliteFile)

    # Connect to database file.
    return sqlite3.connect(sqliteFile)


def find_all_related_transformation(
        cursor: sqlite3.Cursor, transIDs: List[str]):
    """A recursive function that find all related transformations given a list of transformation IDs in the database.

    Args:
        cursor: sqlite3.Cursor. Cursor of current sqlite3 database connection.
        transIDs: List[str]. A list of transformation IDs.
    """

    transQueryStr = "(" + ", ".join(transIDs) + ")"
    cursor.execute(
        f"""
            SELECT node_name
            FROM Log_Transformation
            WHERE trans_id in {transQueryStr} and operation_type in ('ADD_OPERAND', 'REMOVE_OPERAND')
            GROUP BY node_name
        """
    )
    rows = cursor.fetchall()
    nodesList = ["'" + r[0] + "'" for r in rows]

    transQueryStr = "(" + ", ".join(nodesList) + ")"
    cursor.execute(
        f"""
            SELECT trans_id
            FROM Log_Transformation
            WHERE node_name in {transQueryStr} and operation_type in ('ADD_OPERAND', 'REMOVE_OPERAND')
            GROUP BY trans_id
        """
    )
    rows = cursor.fetchall()
    newTransIDs = [str(r[0]) for r in rows]

    if sorted(newTransIDs) != sorted(transIDs):
        transIDs = find_all_related_transformation(cursor, newTransIDs)
    return transIDs


def filter_node_transformation(
    nodeName: str, conn: sqlite3.Connection, verbose: bool, dottyFile: str
):
    """Filter out all node transformation that is related to the given node.

    Args:
        nodeName: str. The node name that is passed to this script.
        conn: sqlite3.Connection. A sqlite3 database connection.
        verbose: bool. Verbosity of the output.
        dottyFile: str. Dotty file name.
    """

    cursor = conn.cursor()
    cursor.execute(
        """
            SELECT trans_id
            FROM Log_Transformation
            WHERE node_name = ?
            GROUP BY trans_id
        """,
        (nodeName,),
    )
    rows = cursor.fetchall()

    directTransIDs = [str(r[0]) for r in rows]

    transIDs = find_all_related_transformation(cursor, directTransIDs)

    for tid in transIDs:
        cursor.execute(
            """
            SELECT *
            FROM Log_Transformation
            WHERE trans_id = ?
        """,
            (tid,),
        )
        rows = cursor.fetchall()
        if len(rows):
            tran = Transformation(tid)
            if tid in directTransIDs:
                tran.isDirectTrans_ = True
            TRANS_LIST.append(tran)
            tran.scopeName_ = rows[0][4].replace(
                "glow::", "").replace(
                "->", r" --\> ")
            for r in rows:
                opr_type, name, kind = r[1:4]
                if opr_type == "ADD_OPERAND":
                    nodeKindAndName = kind + r" \l" + name
                    tran.appendAddedNode(nodeKindAndName)
                    NODES_ADDING_MAP[nodeKindAndName] = tran
                elif opr_type == "REMOVE_OPERAND":
                    nodeKindAndName = kind + r" \l" + name
                    tran.appendRemovedNode(nodeKindAndName)
                    if nodeKindAndName in NODES_ADDING_MAP:
                        tran.addAncestor(NODES_ADDING_MAP[nodeKindAndName])
                elif opr_type == "OPERATOR_BASE":
                    nodeKindAndName = kind + r" \l" + name
                    tran.setBase(nodeKindAndName)

    def processOutDottyName(dottyStyleName):
        return dottyStyleName.split(r"\l")[1]

    def checkNodeInIt(tran, nodeName):
        if nodeName == processOutDottyName(tran.baseNode_):
            return True

        for rn in tran.removedNodes_:
            if nodeName == processOutDottyName(rn):
                return True

        for an in tran.addedNodes_:
            if nodeName == processOutDottyName(an):
                return True

        return False

    for tran in TRANS_LIST:
        if not verbose:
            if not checkNodeInIt(tran, nodeName):
                continue

        print(
            f"\n===============Transformation ID: {tran.transID_} ================")
        print("Scope:  " + tran.scopeName_.replace(r"\>", ">"))
        if nodeName == processOutDottyName(tran.baseNode_):
            print("USER NODE: \n(*)" + tran.baseNode_.replace(r"\l", " "))
        else:
            print("USER NODE: \n" + tran.baseNode_.replace(r"\l", " "))
        print("------ Previous operands set:")

        for rn in tran.removedNodes_:
            if nodeName == processOutDottyName(rn):
                print("\t(*)" + rn.replace(r"\l", " "))
            else:
                print("\t" + rn.replace(r"\l", " "))

        print("------ New operands set:")
        for an in tran.addedNodes_:
            if nodeName == processOutDottyName(an):
                print("\t(*)" + an.replace(r"\l", " "))
            else:
                print("\t" + an.replace(r"\l", " "))

    dump_dotty_DAG(dottyFile)
    conn.commit()


def stat_list_phases(conn, depth=0):
    cursor = conn.cursor()
    cursor.execute(
        """
            SELECT *
            FROM Log_Scope
            ORDER BY scope_id
        """
    )
    rows = cursor.fetchall()

    currDepth = 0
    print("Phase ID \tPhase Name\n-------------------------\n")
    for r in rows:
        if "ENTER" in r[1]:
            currDepth += 1
        if currDepth <= depth or depth == 0:
            print(r[0], "\t" * currDepth + r[1])

        if "EXIT" in r[1]:
            currDepth -= 1
            assert currDepth >= 0


def stat_phases_summary(conn: sqlite3.Connection,
                        startPhase: int, endPhase: int):
    cursor = conn.cursor()
    cursor.execute(
        """
            SELECT lng.scope_id, ls.full_scope_str, lng.operation, lng.node_kind, COUNT(node_kind)
            FROM Log_Node_Operation lng
            LEFT JOIN Log_Scope ls
            ON lng.scope_id = ls.scope_id
            WHERE lng.scope_id >= ? AND lng.scope_id < ?
            GROUP By lng.node_kind
            ORDER BY lng.scope_id
        """,
        (startPhase, endPhase),
    )
    rows = cursor.fetchall()
    print(f"---- Between phase {startPhase} and phase {endPhase}:\n")
    summaryStrs = {}
    for r in rows:
        scope_id, scope, opr, kind, num = r
        if scope_id not in summaryStrs:
            summaryStrs[scope_id] = f"Phase {scope_id}: \n    [{scope}]\n"

        summaryStrs[scope_id] += f"\t {opr}D {num} {kind} nodes.\n"

    for sid in summaryStrs:
        print(summaryStrs[sid])


def stat_phase(conn: sqlite3.Connection, phaseId: int):
    cursor = conn.cursor()
    cursor.execute(
        """SELECT full_scope_str FROM Log_Scope WHERE scope_id=?""", (phaseId,)
    )
    rows = cursor.fetchall()
    fullScope = rows[0][0]
    cursor.execute(
        """
            SELECT node_kind, COUNT(node_kind), COUNT(node_kind)*100.0/ (SELECT Count(*) FROM  Log_Node WHERE create_scope_id < ? AND delete_scope_id >= ?)
            FROM Log_Node
            WHERE create_scope_id < ? AND delete_scope_id >= ?
            GROUP By node_kind
            ORDER BY COUNT(node_kind) DESC
        """,
        (phaseId, phaseId, phaseId, phaseId),
    )
    rows = cursor.fetchall()
    print(f"=== At phase {phaseId} ({fullScope}): \n")
    print(
        "\t{:>4s}  \t{:>12s} \t\t{:>2s}\n--------------------------------------------------------".format(
            "Num",
            "Kind",
            "(Percentage)"))
    for r in rows:
        kind, num, perc = r
        print(
            "\t{:>4d}  \t{:>12s} \t\t({:>2f}%)".format(
                num, kind, round(
                    perc, 2)))


def process():
    """Parse args and process this script. """

    parser = argparse.ArgumentParser(
        description="Filter compilation and optimiztion.")
    parser.add_argument("--db-file")
    parser.add_argument("--filter-target")
    parser.add_argument("--filter-target-verbose")
    parser.add_argument("--dotty-file")
    parser.add_argument("--stat-list-phases", type=bool)
    parser.add_argument("--stat-list-phases-depth", type=int)
    parser.add_argument("--stat-phases-summary", type=int, nargs="+")
    parser.add_argument("--stat-phase", type=int)
    options = parser.parse_args()

    assert options.db_file, "Please specify db file."
    with init_db(options.db_file) as conn:
        dottyFile = options.dotty_file if options.dotty_file else "dotty"
        if options.filter_target:
            filter_node_transformation(
                options.filter_target, conn, False, dottyFile)

        if options.filter_target_verbose:
            filter_node_transformation(
                options.filter_target_verbose, conn, True, dottyFile
            )

        if options.stat_list_phases:
            stat_list_phases(conn)

        if options.stat_list_phases_depth:
            stat_list_phases(conn, options.stat_list_phases_depth)

        if options.stat_phases_summary:
            assert len(options.stat_phases_summary) == 2
            startPhase, endPhase = options.stat_phases_summary
            stat_phases_summary(conn, startPhase, endPhase)

        if options.stat_phase:
            stat_phase(conn, options.stat_phase)


def main():
    process()


if __name__ == "__main__":
    main()
