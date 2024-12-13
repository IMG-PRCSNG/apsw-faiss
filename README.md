# SQLite Virtual Table on top of FAISS index

Experimenting with APSW to get started on fine-tuning the requirements to represent FAISS index as a sqlite vtable

Can later consider re-implementing it in C++ and create an sqlite extension

Advantages:
1. Once a sqlite extension is created, anyone can represent the media search with SQL
2. Can piggyback on SQL as the grammar
3. UI and CLI then becomes a SQL parser / validator for the WHERE clause

Disadvantages:
1. While it removes the need to invent new grammar, the advantages seem narrow - does it add enough value that makes the effort worth it?
2. We could also do this with arbitrary functions in python. Do we need to do it in SQL?


## Pre-requisites
1. [conda](https://docs.anaconda.com/miniconda/install/) / [mamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)

Create an environment with conda / mamba
```bash
conda create -f environment.yml
```

# Try it

```bash
# Activate environment
conda activate apsw-faiss
python3 app.py faiss
```
## What does the sample do
1. Creates a dummy faiss flat l2 index with 1000 vectors of 64 dimensions and writes to disk if not present
2. Adds dummy metadata table for the 1000 vectors
3. Wraps the index search functionality as a sql virtual table implementation and uses the apsw library to register the functionality
4. A search is performed over the joined virtual table and metadata table that returns the id, distance, and metadata rows
```SQL
CREATE TABLE metadata(id INTEGER PRIMARY KEY, description TEXT)
--- INSERT INTO metadata VALUES(?, ?) is done

CREATE VIRTUAL TABLE faissindex USING faiss('test_index.faiss')

SELECT fid.id, fid.dist, metadata.description
FROM faissindex AS fid INNER JOIN metadata ON metadata.id = fid.id
WHERE faiss_search(fid._embedding, embed(12312))
```
where
- `faissindex` is the name of the table
- `faiss` is the python module implementing the table
- `test_index.faiss` path to faiss index on disk
- `faiss_search` identifier that informs sqlite that it should be treated like a constraint (equivalent to LIKE, MATCH, IN, etc)
- `embed` is a user defined function that generates a random vector of 64 dimension and the argument is the seed
- `metadata` contains id and description columns (dummy data)

## Ideas

APSW
- Map numpy array type to blob and back (type adapter)

CLI
- Add REPL interface to make it simpler to try


## References

1. [APSW Docs](https://rogerbinns.github.io/apsw/vtable.html)
2. [SQlite Docs](https://sqlite.org/vtab.html)
3. [SQLite-vss - OG Library](https://github.com/asg017/sqlite-vss)
4. [Sample for REST API based VTable](https://gist.github.com/coleifer/ad0c610b0575db71cfcd)