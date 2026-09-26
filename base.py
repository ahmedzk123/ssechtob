import chess 

"""

will ipdate k==0 case to return a NN that takes in board and ooutpiuts best score 
- kinda like rollout idea but willl hopefilly train NN to find better soiutojns
- 

"""

class MinMax():
    def __init__(self, eval):
        self.eval = eval

    def find_best_move(self, board, k, my_move):
        best_score = -float('inf')
        opp_score = float('inf')
        best_move = None
        opp_move = None
        for move in board.legal_moves:
            board.push(move)
            score = self.minmax_depth_k(board, k-1, not my_move)
            board.pop()
    
            if my_move:
                if score > best_score:
                    best_score = score
                    best_move = move
            else:
                if score < opp_score:
                    best_score = score
                    best_move = move 
    
        return best_move

    def minmax_depth_k(self, board, k, my_move):
        if my_move:
            if board.is_checkmate(): return -float('inf')
            if k == 0: return self.eval(board)  

            best_score = -float('inf')
            for move in board.legal_moves:
                board.push(move)
                move_k_score = self.minmax_depth_k(board, k-1, not my_move)
                board.pop()
                if move_k_score > best_score: 
                    best_score = move_k_score
            return best_score
            
                
        elif not my_move:
            if board.is_checkmate(): return float('inf')
            if k == 0: return self.eval(board)
            opp_score = float('inf')
            for move in board.legal_moves:
                board.push(move)
                move_k_score = self.minmax_depth_k(board, k-1, my_move)
                board.pop()
                if move_k_score < opp_score: 
                    opp_score = move_k_score
            return opp_score



# example: back-rank mate setup, or use a known FEN
board = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")  # rough example, verify the FEN yourself

def function(x):
    return 0

minmax = MinMax(function)
move = minmax.find_best_move(board, k=1, my_move=True)
print(move)  # should be the mating rook move