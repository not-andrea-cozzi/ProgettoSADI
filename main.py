import io
import json
import random
import chess
import pandas as pd
import zstandard as zstd
import torch
from torch_geometric.data import Data
from Component.ChessAnalysisPipeline import ChessAnalysisPipeline, board_to_graph_json


def read_zst_csv(path: str) -> pd.DataFrame:
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as f, dctx.stream_reader(f) as r:
        text = io.TextIOWrapper(r, encoding="utf-8")
        return pd.read_csv(text)


def extract_puzzle_graphs(puzzle_csv_zst: str, out_csv: str, mate_range=(1, 5)):
    df = read_zst_csv(puzzle_csv_zst)
    lo, hi = mate_range
    theme_map = {n: f"mateIn{n}" for n in range(lo, hi + 1)}
    rows = []
    for _, r in df.iterrows():
        themes = str(r.get("Themes", ""))
        mate_n = next((n for n in range(lo, hi + 1) if theme_map[n] in themes), None)
        if mate_n is None:
            continue
        board = chess.Board(r["FEN"])
        rows.append(
            {
                "Game_ID": r["PuzzleId"],
                "SF_Mate": mate_n,
                "Graph_JSON": board_to_graph_json(board),
                "source": "puzzle",
            }
        )
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv


def graph_json_to_pyg(graph_json_str: str, y: int) -> Data:
    g = json.loads(graph_json_str)
    node_ids = [n["id"] for n in g["nodes"]]
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    x = torch.tensor(
        [[n["has_piece"], n["piece_type"], n["color"]] for n in g["nodes"]],
        dtype=torch.float,
    )
    edge_type_map = {"legal_move": 0, "attack": 1, "pin": 2}
    src, dst, etype = ([], [], [])
    for e in g.get("links", g.get("edges", [])):
        src.append(id_to_idx[e["source"]])
        dst.append(id_to_idx[e["target"]])
        etype.append(edge_type_map.get(e.get("edge_type"), 0))
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(etype, dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([y]))


def build_train_val(
    games_csv: str,
    puzzle_csv: str,
    out_dir: str,
    n_target_per_class: dict,
    puzzle_ratio: float = 0.7,
    val_frac: float = 0.1,
    seed: int = 42,
):
    random.seed(seed)
    games_df = pd.read_csv(games_csv)
    puzzle_df = pd.read_csv(puzzle_csv)
    train, val = ([], [])
    for n, target in n_target_per_class.items():
        n_puzzle = int(target * puzzle_ratio)
        n_games = target - n_puzzle
        pool_p = puzzle_df[puzzle_df["SF_Mate"].abs() == n]
        pool_g = games_df[games_df["SF_Mate"].abs() == n]
        pool_p = pool_p.sample(n=min(n_puzzle, len(pool_p)), random_state=seed)
        pool_g = pool_g.sample(n=min(n_games, len(pool_g)), random_state=seed)
        combined = pd.concat([pool_p, pool_g]).sample(frac=1, random_state=seed)
        n_val = int(len(combined) * val_frac)
        val_rows = combined.iloc[:n_val]
        train_rows = combined.iloc[n_val:]
        for split_list, rows in [(train, train_rows), (val, val_rows)]:
            for _, r in rows.iterrows():
                split_list.append(graph_json_to_pyg(r["Graph_JSON"], y=n))
    random.shuffle(train)
    torch.save(train, f"{out_dir}/train.pt")
    torch.save(val, f"{out_dir}/val.pt")
    print(f"train={len(train)} val={len(val)}")


def build_holdout_testset(
    external_csv: str,
    out_dir: str,
    n_range=(1, 10),
    fen_col: str = "FEN",
    mate_col: str = "MateIn",
):
    df = pd.read_csv(external_csv)
    lo, hi = n_range
    test = []
    for _, r in df.iterrows():
        n = int(r[mate_col])
        if not lo <= n <= hi:
            continue
        board = chess.Board(r[fen_col])
        test.append(graph_json_to_pyg(board_to_graph_json(board), y=n))
    random.shuffle(test)
    torch.save(test, f"{out_dir}/test.pt")
    print(f"held-out test={len(test)}")


if __name__ == "__main__":
    N_TARGET = {1: 25000, 2: 25000, 3: 20000, 4: 15000, 5: 15000}


    GAMES_CSV_OUTPUT : str = "dataset/games_all_2019_06.csv"
    pipeline = ChessAnalysisPipeline(
            zst_path="rawData/lichess_db_standard_rated_2019-06.pgn.zst",
            stockfish_path="/usr/games/stockfish",
            output_csv="",
            max_games=None,  
        )
    pipeline.run()
    
    
    extract_puzzle_graphs("lichess_puzzles.csv", "dataset/puzzle_graphs.csv")

    build_train_val(
        games_csv=GAMES_CSV_OUTPUT,   
        puzzle_csv="dataset/puzzle_graphs.csv",
        out_dir="dataset",
        n_target_per_class=N_TARGET,
        puzzle_ratio=0.7,
    )

    # 3. test held-out da fonte ESTERNA (Kaggle/Chess.com)
    # build_holdout_testset(
    #     external_csv="kaggle_chess_puzzles.csv",
    #     out_dir="dataset",
    #     n_range=(1, 10),
    #     fen_col="FEN",
    #     mate_col="MateIn",
    # )