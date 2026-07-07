from Component.ChessAnalysisPipeline import ChessAnalysisPipeline

if __name__ == "__main__":
    pipeline = ChessAnalysisPipeline(
        zst_path="lichess_db_standard_rated_2017-01.pgn.zst",
        stockfish_path="/usr/games/stockfish",
        output_pt="dataset/games.pt",
        max_games=100_000
    )

    pipeline.run()