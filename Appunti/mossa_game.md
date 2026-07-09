| Campo | Significato | Valori possibili |
|---|---|---|
| `Event` | Tipo torneo/partita | `Rated Bullet/Blitz/Classical/Correspondence tournament\|game`, `Casual ...` |
| `Site` | URL partita | `https://lichess.org/{gameId}` |
| `Date` | Data locale | `YYYY.MM.DD` o `????.??.??` |
| `Round` | Round torneo | numero o `-` |
| `White`/`Black` | Username giocatori | stringa |
| `Result` | Esito | `1-0`, `0-1`, `1/2-1/2`, `*` |
| `UTCDate`/`UTCTime` | Timestamp UTC | `YYYY.MM.DD` / `HH:MM:SS` |
| `WhiteElo`/`BlackElo` | Rating Glicko-2 | intero |
| `WhiteRatingDiff`/`BlackRatingDiff` | Δ rating post-game | `+N`/`-N` |
| `WhiteTitle`/`BlackTitle` | Titolo scacchistico | `GM,IM,FM,CM,NM,WGM,WIM,WFM,WCM,BOT` |
| `ECO` | Codice apertura | `A00`-`E99` |
| `Opening` | Nome apertura | stringa |
| `TimeControl` | Base+incremento (sec) | `sec+incr` o `-` (correspondence) |
| `Termination` | Come è finita | `Normal, Time forfeit, Abandoned, Rules infraction` |
| `%eval` | Valutazione SF dopo mossa | centipedoni (es. `0.17`) o mate `#N`/`#-N` |
| `%clk` | Tempo rimanente | `H:MM:SS` |
| Suffisso mossa `!` | Buona mossa | annotazione NAG |
| `!!` | Mossa eccellente | idem |
| `?` | Mossa dubbia/errore | idem |
| `??` | Blunder grave | idem |
| `!?` | Interessante/rischiosa | idem |
| `?!` | Dubbia | idem |