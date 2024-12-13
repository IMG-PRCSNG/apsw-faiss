from __future__ import annotations
from typing import Any, Callable, TYPE_CHECKING, Sequence
import apsw
import apsw.ext

from pathlib import Path

import numpy as np
import faiss as Faiss

if TYPE_CHECKING:
    from apsw import VTTable, VTModule, VTCursor
else:
    VTTable = object
    VTModule = object
    VTCursor = object

import logging
import typer

app = typer.Typer()
app_state = {"verbose": True}
logger = logging.getLogger()
D = 64
N = 1000


def get_dummy_data(n, d):
    logger.info("creating dummy data 1000 x 64")
    np.random.seed(1234)
    x = np.random.random((n, d)).astype("float32")
    return x


def create_index(x: np.ndarray):
    logger.info("Creating faiss index")
    n, d = x.shape
    index = Faiss.IndexFlatL2(d)  # type: ignore
    index.add(x)

    return index


def get_or_create_index(path: Path):
    if path.is_file():
        # read index
        logger.info(f"Loading faiss index from {path}")
        return Faiss.read_index(str(path))

    # Create index
    x = get_dummy_data(n=N, d=D)
    index = create_index(x)
    logger.info(f"Writing dummy faiss index to {path}")
    Faiss.write_index(index, str(path))

    return index


@app.callback()
def base(verbose: bool = False):
    """
    APP CLI
    TODO: Fill this
    """

    import apsw.bestpractice

    app_state["verbose"] = verbose
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s (%(threadName)s): %(name)s - %(levelname)s - %(message)s",
    )
    global logger
    logger = logging.getLogger()

    # Where the extension module is on the filesystem
    print("      Using APSW file", apsw.__file__)

    # From the extension
    print("         APSW version", apsw.apsw_version())

    # From the sqlite header file at APSW compile time
    print("SQLite header version", apsw.SQLITE_VERSION_NUMBER)

    # The SQLite code running
    print("   SQLite lib version", apsw.sqlite_lib_version())

    # If True then SQLite is incorporated into the extension.
    # If False then a shared library is being used, or static linking
    print("   Using amalgamation", apsw.using_amalgamation)

    apsw.bestpractice.apply(apsw.bestpractice.recommended)
    apsw.ext.log_sqlite()


def table_range(start=1, stop=100, step=1):
    for i in range(start, stop + 1, step):
        yield (i,)


def search(q, index, top_k=5):
    logger.info("performing faiss search")
    dist, ids = index.search(q, top_k)
    yield from zip(ids[0], dist[0])


class FaissVTModule(VTModule):

    def Create(
        self,
        connection: apsw.Connection,
        modulename: str,
        databasename: str,
        tablename: str,
        *args: tuple[int | float | bytes | str | None, ...],
    ) -> tuple[str, apsw.VTTable]:
        logger.info(
            f"Create: {connection}, {modulename}, {databasename}, {tablename}, {args}"
        )
        if len(args) != 1 and not isinstance(args[0], str):
            # Empty args passed. Cannot create table
            raise RuntimeError("Cannot create table without path to index")

        self.table = FaissVTTable(Path(args[0].strip("'")))
        self.schema = "create table %s (%s);" % (
            tablename,
            ", ".join(["id INTEGER", "dist REAL", "_embedding BLOB HIDDEN"]),
        )
        return self.schema, self.table

    def Connect(
        self,
        connection: apsw.Connection,
        modulename: str,
        databasename: str,
        tablename: str,
        *args: tuple[int | float | bytes | str | None, ...],
    ) -> tuple[str, apsw.VTTable]:
        logger.info(
            f"Connect: {connection}, {modulename}, {databasename}, {tablename}, {args}"
        )
        return self.schema, self.table

    # TODO - implement a common function for create and connect


class FaissVTTable(VTTable):
    def __init__(self, index_file_path):
        self.index_file_path = index_file_path
        self.index = get_or_create_index(self.index_file_path)

    def Open(self) -> apsw.VTCursor:
        return FaissVTCursor(self.index)

    def BestIndex(
        self,
        constraints: Sequence[tuple[int, int]],
        orderbys: Sequence[tuple[int, int]],
    ) -> Any:
        logger.info(f"calling bestindex constraint: {constraints}, orderby: {orderbys}")

        out_constraints = []
        for col, op in constraints:
            if col == 2 and op == 151:
                # faiss_search
                out_constraints.append((0, True))
            else:
                out_constraints.append(None)

        return out_constraints, 0, "faiss_index", False, 1000

    def Disconnect(self) -> None:
        logger.info("Disconnect")

    def Destroy(self) -> None:
        logger.info("Destroy")

    def FindFunction(
        self, name: str, nargs: int
    ) -> None | Callable[..., Any] | tuple[int, Callable[..., Any]]:
        logger.info(f"FindFunction called name: {name}, nargs: {nargs}")
        if name == "faiss_search" and nargs == 2:
            return (
                int(
                    apsw.mapping_bestindex_constraints[
                        "SQLITE_INDEX_CONSTRAINT_FUNCTION"
                    ]
                )
                + 1,
                lambda *args: False,
            )

        return None


class FaissVTCursor(VTCursor):
    def __init__(self, index):
        self.index = index
        self.rows = enumerate(iter(()))
        self.idx = -1
        self.row = (None, None)
        self.eof = True

    def Rowid(self) -> int:
        return self.idx

    def Next(self) -> None:
        try:
            self.idx, self.row = next(self.rows)
        except StopIteration:
            self.idx = -1
            self.row = (None, None)
            self.eof = True
        return

    def Filter(
        self, indexnum: int, indexname: str, constraintargs: tuple | None
    ) -> None:
        logger.info(f"filter - {indexnum}, {indexname}, {constraintargs}")
        if indexname == "faiss_index" and constraintargs:
            logger.info("searching with faiss and initialising cursor")
            self.rows = enumerate(
                search(
                    np.frombuffer(constraintargs[0], dtype=np.float32).reshape(1, -1),
                    self.index,
                )
            )
            self.eof = False
            self.Next()
        return

    def Eof(self) -> bool:
        return self.eof

    def Column(self, number: int) -> None | int | float | bytes | str:
        x = self.row[number].item()
        logger.info(f"Requesting column - {number} (Row: {self.idx}) = {x}")
        return x

    def Close(self) -> None:
        return


@app.command()
def faiss():
    logger.info("Creating in-memory connection")
    connection = apsw.Connection(":memory:")

    def embed(*args):
        rng = np.random.RandomState(args[0] if args[0] is not None else 1234)
        q = rng.random((1, D)).astype("float32")
        return q.tobytes()

    # Switch to this for table valued functions to keep things simple
    # logger.info('Creating module')
    # apsw.ext.make_virtual_module(connection, "faiss", table_range)
    module = FaissVTModule()
    connection.create_module("faiss", module, read_only=True)
    connection.overload_function("faiss_search", 2)
    connection.create_scalar_function("embed", embed, deterministic=True)
    connection.execute(
        "CREATE TABLE metadata(id INTEGER PRIMARY KEY, description TEXT)"
    )
    strings = ["apple", "banana", "orange", "jackfruit", "cherry"]
    n_strings = len(strings)
    data = ((id, strings[id % n_strings]) for id in range(N))
    q = "INSERT INTO metadata VALUES(?, ?)"
    with connection:
        connection.executemany(q, data)
    # tell SQLite about the table
    connection.execute(
        "create VIRTUAL table faissindex USING faiss('test_index.faiss')"
    )

    query = (
        "SELECT fid.id, fid.dist, metadata.description FROM faissindex AS fid "
        "INNER JOIN metadata ON metadata.id = fid.id "
        "WHERE faiss_search(fid._embedding, embed(12312))"
    )
    # ask for all information available
    qd = apsw.ext.query_info(
        connection,
        query,
        actions=True,  # which tables/views etc and how they are accessed
        explain=True,  # shows low level VDBE
        explain_query_plan=True,  # how SQLite solves the query
    )
    import pprint

    print("query", qd.query)
    print("\nexpanded_sql", qd.expanded_sql)
    print("\nfirst_query", qd.first_query)
    print("\nis_explain", qd.is_explain)
    print("\nis_readonly", qd.is_readonly)
    print("\ndescription")
    pprint.pprint(qd.description)
    if hasattr(qd, "description_full"):
        print("\ndescription_full")
        pprint.pprint(qd.description_full)

    print("\nquery_plan")
    pprint.pprint(qd.query_plan)

    print(apsw.ext.format_query_table(connection, query))


@app.command()
def example():

    # set column names
    table_range.columns = ("value",)
    # set how to access what table_range returns
    table_range.column_access = apsw.ext.VTColumnAccess.By_Index
    connection = apsw.Connection(":memory:")

    # register it
    apsw.ext.make_virtual_module(connection, "range", table_range)

    # see it work.  we can provide both positional and keyword
    # arguments
    query = "SELECT * FROM range(90) WHERE step=2"
    print(apsw.ext.format_query_table(connection, query))

    # the parameters are hidden columns so '*' doesn't select them
    # but you can ask
    query = "SELECT *, start, stop, step FROM range(89) WHERE step=3"
    print(apsw.ext.format_query_table(connection, query))
    pass


if __name__ == "__main__":
    app()
