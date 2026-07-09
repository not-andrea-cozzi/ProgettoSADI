import io
import os
from Model.graph_builder import GraphBuilder
import chess
import chess.pgn
import chess.engine
import pandas as pd
import torch
import zstandard as zstd
from tqdm import tqdm
from Component.ChessAnalysisPipeline import ChessAnalysisPipeline
from Component.PuzzleGraphDataset import PuzzleGraphDataset, merge_and_split
from Component.TimeStatBuilder import TimeStatsBuilder, load_avg_time_by_rating


def decompress_zst_csv(zst_path: str, out_csv: str, chunk_size: int = 1024 * 1024) -> str:
    """Decomprime un .csv.zst in un .csv su disco, con progress bar (basata sui byte
    compressi letti, non sulla dimensione finale che zstd non conosce a priori)."""
    if os.path.exists(out_csv):
        print(f"{out_csv} gia' presente, salto decompressione.")
        return out_csv
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    total_size = os.path.getsize(zst_path)
    dctx = zstd.ZstdDecompressor()

    with open(zst_path, "rb") as f_in, open(out_csv, "wb") as f_out:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=f"Decomprimo {os.path.basename(zst_path)}") as pbar:
            reader = dctx.stream_reader(f_in)
            while True:
                chunk = reader.read(chunk_size)
                if not chunk:
                    break
                f_out.write(chunk)
                pbar.n = f_in.tell()
                pbar.refresh()
    return out_csv


def build_puzzle_pt(csv_path: str, root: str, mate_range=(1, 5), max_puzzles=None,
                     avg_time_by_rating=None):
    """Genera i tre split puzzle (train/val/test) come liste di Data PyG.
    Lo split e' fatto per PuzzleId DENTRO PuzzleGraphDataset (vedi fix in quel file):
    ogni split="train"/"val"/"test" processa lo STESSO pool di righe CSV filtrate e
    ne prende una partizione disgiunta e deterministica (stesso seed), quindi le tre
    chiamate qui sotto non si sovrappongono mai."""
    splits = {}
    for split in ("train", "val", "test"):
        ds = PuzzleGraphDataset(csv_path, root, split=split, mate_range=mate_range,
                                 max_puzzles=max_puzzles, avg_time_by_rating=avg_time_by_rating)
        splits[split] = list(ds)
    return splits


def build_external_holdout(
    external_csv: str,
    stockfish_path: str,
    out_pt: str,
    mate_range=(1, 5),
    time_limit: float = 0.2,
    pgn_col: str = "pgn",
    max_games=None,
    max_problems=None,
    require_move_match: bool = True,
):
    """Held-out ESTERNO da dataset chess.com (60k games, colonna `pgn` con partita completa).
    Non ha FEN/Moves/MateIn pronti: scandisce ogni partita mossa per mossa, usa Stockfish
    per trovare posizioni di matto forzato in mate_range mosse (stessa logica di
    ChessAnalysisPipeline, qui sequenziale dato il volume ridotto per l'eval finale).
    MAI mischiato con games/puzzle Lichess usati in train/val (richiesta del prof).

    require_move_match: se True (default), una posizione entra nell'held-out solo se la
    mossa EFFETTIVAMENTE GIOCATA in partita coincide con la prima mossa del matto forzato
    trovato da Stockfish (info[0]["pv"][0]). Senza questo controllo la label userebbe la
    mossa storica anche quando il giocatore non ha eseguito il matto individuato (mossa
    alternativa, non ottimale o un errore), producendo esempi con etichetta scorretta."""
    df = pd.read_csv(external_csv)
    if max_games:
        df = df.head(max_games)

    lo, hi = mate_range
    test_data = []
    skipped_no_match = 0

    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"Threads": 1, "Hash": 64})

    try:
        for game_idx, pgn_text in enumerate(tqdm(df[pgn_col].dropna(), desc="Held-out chess.com")):
            game = chess.pgn.read_game(io.StringIO(pgn_text))
            if game is None:
                continue

            node = game
            while node.variations:
                nxt = node.variation(0)
                board = node.board()

                try:
                    info = engine.analyse(board, chess.engine.Limit(time=time_limit, mate=hi), multipv=1)
                except Exception:
                    node = nxt
                    continue

                score = info[0].get("score") if info else None
                if score and score.relative.is_mate():
                    mate_n = score.relative.mate()
                    if lo <= abs(mate_n) <= hi:
                        pv = info[0].get("pv")
                        engine_best_move = pv[0] if pv else None

                        if require_move_match and nxt.move != engine_best_move:
                            skipped_no_match += 1
                            node = nxt
                            continue

                        legal = list(board.legal_moves)
                        try:
                            best_idx = legal.index(nxt.move)
                        except ValueError:
                            node = nxt
                            continue

                        label = {"mate_n": mate_n, "best_move_idx": best_idx}
                        d = GraphBuilder.board_to_pyg_data(board, clock_seconds=0.0, label=label)
                        d.problem_id = f"chesscom_{game_idx}_{node.ply()}"
                        test_data.append(d)

                node = nxt

            if max_problems and len(test_data) >= max_problems:
                break
    finally:
        engine.quit()

    torch.save(test_data, out_pt)
    print(f"held-out esterno: {len(test_data)} problemi -> {out_pt} "
          f"(scartati per mossa non giocata dal match Stockfish: {skipped_no_match})")
    return test_data

if __name__ == "__main__":
    MATE_RANGE = (1, 5)

    # Crea tutte le cartelle di output UNA VOLTA sola, prima di qualunque step lento,
    # cosi' un path mancante non fa perdere ore di scansione (vedi bug torch.save).
    os.makedirs("dataset", exist_ok=True)
    os.makedirs("dataset/puzzles", exist_ok=True)
    os.makedirs("dataset/merged", exist_ok=True)

    # Statistiche reali Lichess (tempo medio speso per mossa, per fascia di rating),
    # come richiesto dalla spec ("average times from similar Lichess games").
    # Calcolate dagli stessi game .pgn.zst usati allo step 1, da %clk reali
    # (tempo speso = clock_precedente - clock_attuale + increment, per colore).
    # Cache su json: se gia' presente non ricalcola (lettura .pgn.zst e' lenta).
    time_stats_path = "dataset/avg_time_by_rating.json"
    if os.path.exists(time_stats_path):
        AVG_TIME_BY_RATING = load_avg_time_by_rating(time_stats_path)
    else:
        stats_builder = TimeStatsBuilder(
            zst_path="rawData/lichess_db_standard_rated_2019-06.pgn.zst",
            max_games=50_000,  
            bucket_size=100,
        )
        AVG_TIME_BY_RATING = stats_builder.build_and_save(time_stats_path)
    print(f"avg_time_by_rating: {len(AVG_TIME_BY_RATING)} bucket -> {time_stats_path}")

    """
    games_pipeline = ChessAnalysisPipeline(
        zst_path="rawData/lichess_db_standard_rated_2019-06.pgn.zst",
        stockfish_path="/usr/games/stockfish",
        output_pt="dataset/games.pt",
        mate_range=MATE_RANGE,
        max_games=180_000,
    )
    games_splits, games_paths = games_pipeline.run()
   

    # 2. Puzzle: rawData/lichess_puzzles.csv.zst -> decompresso -> dataset/puzzles/puzzle_{train,val,test}.pt
    #    Split per PuzzleId fatto DENTRO PuzzleGraphDataset (fix leakage): tutte le mosse
    #    dello stesso puzzle finiscono nello stesso split.
    puzzle_csv = decompress_zst_csv(
        zst_path="rawData/lichess_db_puzzle.csv.zst",
        out_csv="dataset/lichess_puzzles.csv",
    )
    puzzle_splits = build_puzzle_pt(
        csv_path=puzzle_csv,
        root="dataset/puzzles",
        mate_range=MATE_RANGE,
        max_puzzles=100_000,
        avg_time_by_rating=AVG_TIME_BY_RATING,
    )

    # 3. Merge puzzle + games PER-SPLIT -> dataset/merged/merged_{train,val,test}.pt
    #    NIENTE ri-shuffle/ri-split qui: puzzle_splits e games_splits sono gia' partizionati
    #    correttamente a monte (per puzzle_id / game_id). merge_and_split unisce solo le
    #    liste corrispondenti split-per-split.
    merge_and_split(
        puzzle_splits=puzzle_splits,
        games_splits=games_splits,
        out_dir="dataset/merged",
    )   

    """

    # 4. Held-out FINALE: fonte esterna (chess.com 60k games), MAI vista in training.
    #    Questo e' il test set richiesto dal prof per la valutazione comparativa GNN vs LLM.
    build_external_holdout(
        external_csv="rawData/club_games_data.csv",
        stockfish_path="/usr/games/stockfish",
        out_pt="dataset/merged/external_holdout.pt",
        mate_range=MATE_RANGE,
        max_games=500,
    )

    print("Train/val/test pronti in dataset/merged/merged_{train,val,test}.pt")
    print("Held-out ESTERNO (eval finale) in dataset/merged/external_holdout.pt")
    print("NB: per Component/Trainer.py usa merged_train.pt / merged_val.pt.")
    print("Per la valutazione finale/comparativa GNN vs LLM usa external_holdout.pt, non merged_test.pt.")