from Model.ChessMove import ChessMove


class ChessGame:
    """Rappresenta l'intero Game, contenente la lista di oggetti ChessMove"""
    def __init__(self, game, analyzer=None, engine_instance=None):
        self.headers = game.headers
        self._game = game
        self.result = self.headers.get("Result")
        self.moves: list[ChessMove] = []
        
        node = game
        while not node.is_end():
            node = node.variation(0)      
            mossa_oggetto = ChessMove(node) 
            
            # Se passiamo l'analyzer, arricchiamo la mossa in tempo reale durante il parsing
            if analyzer and engine_instance:
                analyzer.enrich_move_with_stockfish(node, mossa_oggetto, engine_instance)
                
            self.moves.append(mossa_oggetto) 
            
        self.move_strings = [mossa.uci for mossa in self.moves]