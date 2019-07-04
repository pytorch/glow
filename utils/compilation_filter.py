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
import sqlite3
import os
from typing import (
    List,
    Dict,
)

# A list of all filtered transformations.
TRANS_LIST: List['Transformation'] = []

# Mapping between added nodes and the transformation that adds these nodes.
NODES_ADDING_MAP: Dict[str, 'Transformation'] = {}


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
        self.ancestors_: List['Transformation'] = []
        self.scopeName_: str = ''
        self.transID_: str = transID
        self.isDirectTrans_: bool = False

    def appendAddedNode(self, nodeName: str) -> None:
        """Append the added nodes of this transformation."""

        self.addedNodes_.append(nodeName)

    def appendRemovedNode(self, nodeName: str) -> None:
        """Append the removed nodes of this transformation."""

        self.removedNodes_.append(nodeName)

    def addAncestor(self, ancestor: 'Transformation') -> None:
        """Add ancestors of this transformation."""

        self.ancestors_.append(ancestor)


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

        labelStr = rf"""{{ {{SCOPE:\l{tran.scopeName_} }}|{{ORIGINAL NODES:\l\l"""
        for rstr in tran.removedNodes_:
            labelStr += rf"""{rstr}\l\l"""
        labelStr += rf"}}| {{REPLACED BY:\l\l"
        for astr in tran.addedNodes_:
            labelStr += rf"""{astr}\l\l"""
        labelStr += f"}} }}"
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

    def dump_graph(self) -> None:
        """Visits the graph and generates the dotty information. """

        self.visit_nodes()
        self.visit_edges()
        with open(f"transformations.dot", "w") as f:
            f.write("digraph DAG {\n\trankdir=TB;\n")
            for v in self.vertices_:
                f.write(f"{v}\n")
            for e in self.edges_:
                f.write(f"{e};\n")
            f.write("}")


def dump_dotty_DAG():
    """A helper function to dump the dotty file."""

    dotty = DottyPrinter(TRANS_LIST)
    dotty.dump_graph()


def parse_args():
    """Parse the arguments of this script. """

    parser = argparse.ArgumentParser(
        description="Filter compilation and optimiztion.")
    parser.add_argument("--db-file")
    parser.add_argument("--filter-target")
    options = parser.parse_args()

    assert options.db_file and options.filter_target, "Please specify db file and filter target."
    return options.db_file, options.filter_target


def init_db(sqliteFile: str) -> sqlite3.Connection:
    """Initialize a sqlite3 database connection."""

    assert os.path.isfile(sqliteFile)

    # Connect to database file.
    return sqlite3.connect(sqliteFile)


def find_all_related_transformation(
        cursor: sqlite3.Cursor,
        transIDs: List[str]):
    """A recursive function that find all related transformations given a list of transformation IDs in the database.

    Args:
        cursor: sqlite3.Cursor. Cursor of current sqlite3 database connection.
        transIDs: List[str]. A list of transformation IDs.
    """

    transQueryStr = "(" + ', '.join(transIDs) + ')'
    cursor.execute(f"""
            SELECT node_name
            FROM Log_Transformation
            WHERE trans_id in {transQueryStr}
            GROUP BY node_name
        """)
    rows = cursor.fetchall()
    nodesList = ["'" + r[0] + "'" for r in rows]

    transQueryStr = "(" + ', '.join(nodesList) + ')'
    cursor.execute(f"""
            SELECT trans_id
            FROM Log_Transformation
            WHERE node_name in {transQueryStr}
            GROUP BY trans_id
        """)
    rows = cursor.fetchall()
    newTransIDs = [str(r[0]) for r in rows]

    if sorted(newTransIDs) != sorted(transIDs):
        transIDs = find_all_related_transformation(cursor, newTransIDs)
    return transIDs


def filter_node_transformation(nodeName: str, conn: sqlite3.Connection):
    """Filter out all node transformation that is related to the given node.

    Args:
        nodeName: str. The node name that is passed to this script.
        conn: sqlite3.Connection. A sqlite3 database connection.
    """

    cursor = conn.cursor()
    cursor.execute("""
            SELECT trans_id
            FROM Log_Transformation
            WHERE node_name = ?
            GROUP BY trans_id
        """, (nodeName,))
    rows = cursor.fetchall()

    directTransIDs = [str(r[0]) for r in rows]

    transIDs = find_all_related_transformation(cursor, directTransIDs)

    for tid in transIDs:
        cursor.execute("""
            SELECT *
            FROM Log_Transformation
            WHERE trans_id = ?
        """, (tid, ))
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
                if opr_type == 'ADD':
                    nodeKindAndName = kind + r" \l" + name
                    tran.appendAddedNode(nodeKindAndName)
                    NODES_ADDING_MAP[nodeKindAndName] = tran
                elif opr_type == 'REMOVE':
                    nodeKindAndName = kind + r" \l" + name
                    tran.appendRemovedNode(nodeKindAndName)
                    if nodeKindAndName in NODES_ADDING_MAP:
                        tran.addAncestor(NODES_ADDING_MAP[nodeKindAndName])

    dump_dotty_DAG()
    conn.commit()


def main():
    dbFile, filterTarget = parse_args()
    with init_db(dbFile) as conn:
        filter_node_transformation(filterTarget, conn)


if __name__ == "__main__":
    main()
