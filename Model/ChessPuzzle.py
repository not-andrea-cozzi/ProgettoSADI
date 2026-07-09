import chess
from Model.ChessMove import ChessMove, ColorePedina, TipoPedina


class ChessPuzzle:
    """Rappresenta un puzzle Lichess (da lichess_db_puzzle.csv) con la lista di ChessMove risolutive."""

    def __init__(self, row: dict, avg_time_by_rating: dict | None = None):
        # Colonne CSV: PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags
        self.puzzle_id = row["PuzzleId"]
        self.fen = row["FEN"]
        self.rating = int(row["Rating"])
        self.themes = row["Themes"].split()
        self.mate_n = self._extract_mate_n(self.themes)

        uci_moves = row["Moves"].split()
        self.moves: list[ChessMove] = self._build_moves(self.fen, uci_moves)
        self.simulated_clock = self._simulate_clock(avg_time_by_rating)

    @staticmethod
    def _extract_mate_n(themes: list[str]) -> int | None:
        for t in themes:
            if t.startswith("mateIn"):
                try:
                    return int(t.replace("mateIn", ""))
                except ValueError:
                    return None
        return None

    def _build_moves(self, fen: str, uci_moves: list[str]) -> list[ChessMove]:
        """Ricostruisce la sequenza come game PGN sintetico per riusare ChessMove."""
        board = chess.Board(fen)
        game = chess.pgn.Game()
        game.setup(board)
        node = game
        for uci in uci_moves:
            node = node.add_variation(chess.Move.from_uci(uci))

        moves = []
        node = game
        while node.variations:
            node = node.variation(0)
            moves.append(ChessMove(node))
        return moves

    def _simulate_clock(self, avg_time_by_rating: dict | None) -> list[float]:
        """Tempo simulato per mossa: rating puzzle piu' alto -> pensiero piu' lungo.
        avg_time_by_rating: dict opzionale {rating_bucket: avg_seconds} da statistiche Lichess reali."""
        if avg_time_by_rating:
            bucket = round(self.rating / 100) * 100
            base = avg_time_by_rating.get(bucket, avg_time_by_rating.get(min(avg_time_by_rating)))
        else:
            base = 5.0 + (self.rating / 3000.0) * 55.0  # 5s..~60s lineare sul rating

        return [round(base * (1.0 + 0.1 * i), 2) for i in range(len(self.moves))]