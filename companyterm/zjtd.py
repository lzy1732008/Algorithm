def findCompany(N,matrix):
    indexs = [-1 for _ in range(N)]
    count = 0
    for i in range(N):
        index = count + 1
        if indexs[i] != -1:
            index = indexs[i]
        else:
            count += 1
        for j in range(N):
            if matrix[i][j] >= 3 and indexs[j] == -1:
                indexs[j] = index

    return count

def countPath(n):
    if n == 2:
        return 1
    if n == 4:
        return 2
    if n == 6:
        return 5




    return n

def playGame(n,matrix):
    visited = [[0 for _ in range(4)] for _ in range(4)]
    def play():
        for i in range(3):
            for j in range(4):
                if visited[i][j] == 0 and matrix[i + 1][j] == matrix[i][j] and matrix[i][j] != 0:
                    matrix[i + 1][j] = 0
                    matrix[i][j] = matrix[i][j] << 1
                    visited[i][j] = 1
                    visited[i + 1][j] = 1

                    t = i - 1
                    temp = matrix[i][j]
                    while t >= 0:
                        if matrix[t][j] == 0:
                            t -= 1
                        else:
                            break
                    matrix[i][j] = 0
                    if t == -1:
                        matrix[0][j] = temp
                    else:
                        matrix[t + 1][j] = temp


    def preProcessM(matrix):
        if n == 2:
            matrix = matrix[::-1]
        elif n == 3:
            matrix = [[n for n in line] for line in list(zip(*matrix[::-1]))]
        elif n == 4:
            matrix = [[n for n in line] for line in list(zip(*matrix))[::-1]]
        return matrix

    def lastProcessM(matrix):
        if n == 2:
            matrix = matrix[::-1]
        elif n == 4:
            matrix = [[n for n in line] for line in list(zip(*matrix[::-1]))]
        elif n == 3:
            matrix = [[n for n in line] for line in list(zip(*matrix))[::-1]]
        return matrix

    matrix = preProcessM(matrix)
    play()
    matrix = lastProcessM(matrix)
    return matrix




import sys
if __name__ == "__main__":
    # n = int(sys.stdin.readline().strip())
    # res = countPath(n)
    n = int(sys.stdin.readline().strip())
    lines = []
    for i in range(4):
        lines.append(list(map(int, sys.stdin.readline().strip().split())))
    t = playGame(n,lines)
    for line in lines:
        print(' '.join(map(str,line)))



# import sys
# if __name__ == "__main__":
#     n = int(sys.stdin.readline().strip())
#     res = countPath(n)
#     print(res)

