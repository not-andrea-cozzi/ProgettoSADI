import io
import re
import json
from collections import defaultdict

import chess.pgn
import zstandard as zstd


class TimeStatsBuilder:
    """Calcola avg_time_by_rating leggendo i %clk reali dai game Lichess (.pgn.zst),
    per popolare la simulazione dei tempi puzzle richiesta dalla spec:
    'augment with ... average times from similar Lichess games'.

    Nota: %clk e' il tempo RIMASTO sull'orologio, non il tempo speso sulla mossa.
    Il tempo speso si ricava come:
        speso[i] = clock[i-1] - clock[i] + increment
    confrontando mosse consecutive dello STESSO colore (bianco con bianco, nero con nero).
    """

    CLK_RE = re.compile(r'\[%clk (\d+):(\d+):(\d+)\]')

    def __init__(self, zst_path: str, max_games: int = 20_000, bucket_size: int = 100):
        self.zst_path = zst_path
        self.max_games = max_games
        self.bucket_size = bucket_size

    @staticmethod
    def _parse_clock(comment: str) -> float | None:
        m = TimeStatsBuilder.CLK_RE.search(comment or "")
        if not m:
            return None
        h, mi, s = map(int, m.groups())
        return float(h * 3600 + mi * 60 + s)

    @staticmethod
    def _parse_increment(time_control: str) -> float:
        # TimeControl es. "300+0" -> base=300, incr=0. "-" (correspondence) -> 0.
        if not time_control or time_control == "-":
            return 0.0
        m = re.match(r'(\d+)\+(\d+)', time_control)
        return float(m.group(2)) if m else 0.0

    def _stream_games(self):
        dctx = zstd.ZstdDecompressor()
        with open(self.zst_path, "rb") as f, dctx.stream_reader(f) as r:
            text = io.TextIOWrapper(r, encoding="utf-8")
            n = 0
            while True:
                if self.max_games and n >= self.max_games:
                    break
                g = chess.pgn.read_game(text)
                if g is None:
                    break
                n += 1
                yield g

    def build(self) -> dict[int, float]:
        """Ritorna {rating_bucket: avg_seconds_spent_per_move}."""
        bucket_sum = defaultdict(float)
        bucket_count = defaultdict(int)

        for game in self._stream_games():
            headers = game.headers
            increment = self._parse_increment(headers.get("TimeControl", ""))
            try:
                white_elo = int(headers.get("WhiteElo", 0))
                black_elo = int(headers.get("BlackElo", 0))
            except ValueError:
                continue
            if not white_elo or not black_elo:
                continue

            # clock precedente per colore (None finche' non abbiamo un secondo campione)
            prev_clock = {chess.WHITE: None, chess.BLACK: None}

            node = game
            while node.variations:
                nxt = node.variation(0)
                mover_color = node.board().turn  # chi sta per muovere in `nxt`
                clk = self._parse_clock(nxt.comment)

                if clk is not None:
                    prev = prev_clock[mover_color]
                    if prev is not None:
                        spent = prev - clk + increment
                        if spent > 0:  # scarta valori negativi/rumore
                            rating = white_elo if mover_color == chess.WHITE else black_elo
                            bucket = round(rating / self.bucket_size) * self.bucket_size
                            bucket_sum[bucket] += spent
                            bucket_count[bucket] += 1
                    prev_clock[mover_color] = clk

                node = nxt

        return {
            bucket: round(bucket_sum[bucket] / bucket_count[bucket], 2)
            for bucket in bucket_sum
            if bucket_count[bucket] >= 20  # bucket con troppo pochi campioni -> scartato
        }

    def build_and_save(self, out_json: str) -> dict[int, float]:
        stats = self.build()
        with open(out_json, "w") as f:
            json.dump(stats, f, indent=2, sort_keys=True)
        return stats


def load_avg_time_by_rating(json_path: str) -> dict[int, float]:
    """Carica il json prodotto da build_and_save, con chiavi int (json le salva come str)."""
    with open(json_path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}