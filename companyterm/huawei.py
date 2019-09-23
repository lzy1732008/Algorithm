
class Node:
    def __init__(self,x,y):
        self.x = x
        self.y = y

def Dis(node1,node2):

    return pow(pow(node1.x-node2.x, 2)+ pow(node1.y - node2.y, 2),0.5)

def MinDis(axises):
    nodes = [Node(0,0)]
    i = 0
    while i < len(axises)-1:
        node = Node(axises[i],axises[i+1])
        nodes.append(node)
        i += 2
    disMatrix = []
    for i in range(6):
        line = []
        for j in range(6):
            line.append(Dis(nodes[i],nodes[j]))
        disMatrix.append(line)
    initnode = 0
    lastsum = 0
    mindis = 100000000000000000
    nodes = [i for i in range(1,6)]
    mindis = visit(initnode, lastsum, nodes, disMatrix, mindis)
    return mindis
    #递归计算最短距离

def pathdis(path, disMatrix):
    sum = 0
    for i in range(1,5):
        sum += disMatrix(path[i-1],path[i])
    return sum

def visit(last, lastsum, nodes, disMatrix, mindis):
    if len(nodes) == 1:
        sum = lastsum + disMatrix[last][nodes[0]]
        sum += disMatrix[nodes[0]][0]
        if sum < mindis: mindis = sum
        return mindis
    for i in range(len(nodes)):
        sum = lastsum + disMatrix[last][nodes[i]]
        newnodes = nodes[:i]+nodes[i+1:]
        mindis = visit(nodes[i], sum, newnodes, disMatrix, mindis)
    return mindis

def ALine(node1,node2):
    if node1.x == node2.x or node1.y == node2.y or node1.x-node2.x == node2.y - node2.y:
        return True
    return False

def CutFruit(axises):
    if len(axises) <= 1: return len(axises)
    count = 0
    nodels = []
    for node in axises:
        flag = 0
        for n in nodels:
            if ALine(n,node):
                flag = 1
                break
        if flag == 0:
            count += 1
        nodels.append(node)
    return count


import sys
if __name__ == "__main__":
    num = int(sys.stdin.readline().strip())
    axises = []
    for i in range(num):
        xy = list(map(int,sys.stdin.readline().strip().split()))
        axises.append(Node(xy[0],xy[1]))
    n = CutFruit(axises)
    print(n)



