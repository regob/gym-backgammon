import numpy as np

MX = 220

# win_prob[i][j] = winning probability of player 1 if the pips needed are (i, j) and player 1 is on turn
# win prob is calculated as if the game only consisted of rolling the dice
win_prob = np.ones((MX, MX), dtype=float)

# win_prob is computed using DP, by computing the i-th row and column simultaneously
for i in range(MX):
    for j in range(i, MX):
        # compute [i][j]
        total = 0.0
        for k in range(36):
            n1, n2 = k // 6 + 1, k % 6 + 1
            col = i - n1 - n2 if n1 != n2 else i - (n1 + n2) * 2
            if col <= 0:
                total += 1.0
            else:
                total += 1.0 - win_prob[j, col]
        win_prob[i, j] = total / 36

        # compute [j, i]
        total = 0.0
        for k in range(36):
            n1, n2 = k // 6 + 1, k % 6 + 1
            col = j - n1 - n2 if n1 != n2 else j - (n1 + n2) * 2
            if col <= 0:
                total += 1.0
            else:
                total += 1.0 - win_prob[i, col]
        win_prob[j, i] = total / 36

def dumbeval(board: np.ndarray):
    computer_pip = sum((i + 1) * c for i, c in enumerate(board[1:26]) if c > 0)
    opponent_pip = sum((25 - i) * (-c) for i, c in enumerate(board[:25]) if c < 0)
    
    computer_pip = min(int(computer_pip), MX - 1)
    opponent_pip = min(int(opponent_pip), MX - 1)

    if computer_pip == 0:
        return 9999999.0

    # opponent goes first
    return 1.0 - win_prob[opponent_pip, computer_pip]
