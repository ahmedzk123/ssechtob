import chess 

from base import MinMax as minmax, function as heuristic


# game below

def print_board(board, perspective=chess.WHITE):
    rows = str(board).split("\n")          # 8 lines, rank 8 first
    files = "a b c d e f g h"

    if perspective == chess.BLACK:         # flip so the player's pieces are at the bottom
        rows = [r[::-1] for r in reversed(rows)]
        files = "h g f e d c b a"
        ranks = range(1, 9)
    else:
        ranks = range(8, 0, -1)

    print()
    for rank, row in zip(ranks, rows):
        print(rank, row)
    print("  " + files)
    print()

board = chess.Board()  # rough example, verify the FEN yourself

choice = ""
while choice not in ("W", "B"):
    choice = input("W or B? ").strip().upper()

player_color = chess.WHITE if choice == "W" else chess.BLACK

bot_engine = minmax(heuristic, not player_color)
#player_engine = minmax(heuristic, player_color)


print_board(board, player_color)

while not board.is_game_over():

    if board.turn == player_color:
        try:
            player_move = input("Your move: ").strip()
            board.push_uci(player_move)

            #player_move = player_engine.find_best_move(board.copy(), 5, True)
            #board.push(player_move)
            print_board(board, player_color)

        except ValueError as error:
            print("Illegal move: ", error)
    else:
        try:
            bot_move = bot_engine.find_best_move(board.copy(), 5, True)
            board.push(bot_move)
            print("bot played", bot_move.uci())
            print_board(board, player_color)
            

        except ValueError as error:
            print("Illegal move: ", error)

outcome = board.outcome()         
