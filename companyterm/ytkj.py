def fun1(anum,bnum):
    station = [n for n in anum]
    sum = 0
    for i, vb in enumerate(bnum):
        temp = vb
        for j, va in enumerate(station):
            if va > 0:
               if va < temp:
                   temp -= va
                   station[j] = 0
                   sum += va * (i - j)
               else:
                   station[j] = va - temp
                   sum += temp * (i - j)
                   break
    return sum

class Node:
     def __init__(self,index,time,gain):
         self.index = index
         self.time = time
         self.gain = gain

Inf = 1000000
def fun2(Q,lines,S,E):
    #preprocess:
    matrix = [[Inf for _ in range(len(Q))] for _ in range(len(Q))]
    for line in lines:
        matrix[line[0] - 1][line[1] - 1] = line[2]
        matrix[line[1] - 1][line[0] - 1] = line[2]

    minT = [Inf for _ in range(len(Q))]
    minG = [0 for _ in Q]
    visited = [False for _ in range(len(Q))]

    minT[S - 1] = 0
    minG[S - 1] = Q[S - 1]
    count = 0
    while count < len(Q):
        minNum = Inf
        min_index = -1
        for i in range(len(Q)):
            if visited[i] == False:
               if minT[i] < minNum:
                   min_index = i
                   minNum = minT[i]
        if min_index == -1:
            break

        print(min_index)
        visited[min_index] = True
        count += 1
        for i in range(len(matrix[min_index])):
            if matrix[min_index][i] != Inf:
                if minT[i] >= minT[min_index] + matrix[min_index][i]:
                    minT[i] = minT[min_index] + matrix[min_index][i]
                    minG[i] = max(minG[min_index] + Q[i], minG[i])
        print(minG)
    return minT[E - 1], minG[E - 1]


import sys
if __name__ == "__main__":
    num1 = list(map(int, sys.stdin.readline().strip().split()))
    Q = list(map(int, sys.stdin.readline().strip().split()))
    lines = []
    for i in range(num1[1]):
        lines.append(list(map(int, sys.stdin.readline().strip().split())))
    t = fun2(Q,lines,num1[2],num1[3])
    print(t)


