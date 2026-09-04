import torch
data = torch.load("Dataset/Train/merged_train.pt", weights_only=False)
sources = set()
for d in data[:5]:
    sources.add(hasattr(d, 'puzzle_id'))  # True=puzzle, False=games/altro
print(sources)
clocks = [d.clock_seconds for d in data if hasattr(d, 'clock_seconds')]
print(min(clocks), max(clocks), sum(clocks)/len(clocks))
import numpy as np

is_puzzle = lambda d: hasattr(d, 'puzzle_id')

puzzle_clocks = [getattr(d, 'clock_seconds', None) for d in data if is_puzzle(d)]
games_clocks = [getattr(d, 'clock_seconds', None) for d in data if not is_puzzle(d)]

puzzle_clocks = [c for c in puzzle_clocks if c is not None]
games_clocks = [c for c in games_clocks if c is not None]

print("n puzzle con clock_seconds:", len(puzzle_clocks), "/ totale puzzle:", sum(is_puzzle(d) for d in data))
print("n games con clock_seconds:", len(games_clocks), "/ totale games:", sum(not is_puzzle(d) for d in data))

if puzzle_clocks:
    print("puzzle stats:", np.percentile(puzzle_clocks, [0,10,50,90,100]))
if games_clocks:
    print("games stats:", np.percentile(games_clocks, [0,10,50,90,100]))
    print("games frac < 1s:", (np.array(games_clocks) < 1).mean())
    print("games frac == 0:", (np.array(games_clocks) == 0).mean())