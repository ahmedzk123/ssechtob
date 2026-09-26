"""

will ipdate k==0 case to return a NN that takes in board and ooutpiuts best score 
- kinda like rollout idea but willl hopefilly train NN to find better soiutojns
- 

"""

def minmax_depth_k(board, k, my_move):
    if my_move:
        if k == 0: return 0  
        if board.is_checkmate(): return float('inf')

        best_score = -float('inf')
        for move in board.legal_moves:
            board.push(move)
            move_k_score = minmax_depth_k(board, k-1, not my_move)
            board.pop()
            if move_k_score > best_score: 
                best_score = move_k_score
        return best_score
        
            
    elif not my_move:
        if k == 0: return 0
        if board.is_checkmate(): return -float('inf')
        opp_score = float('inf')
        for move in board.legal_moves:
            board.push(move)
            move_k_score = minmax_depth_k(board, k-1, my_move)
            board.pop()
            if move_k_score < opp_score: 
                opp_score = move_k_score
        return opp_score
            
def best_move(board, k, my_move):
    best_score = -float('inf')
    opp_score = float('inf')
    best_move = None
    opp_move = None
    for move in board.legal_moves:
        board.push(move)
        score = minmax_depth_k(board, k-1, not my_move)
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
        
            