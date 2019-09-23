def fun(array):
    row = len(array) - 1
    col = len(array[0]) - 1
    direct = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    num_dict = {}
    for i in range(row + 1):
        for j in range(col + 1):
            if array[i][j] != '-' and int(array[i][j]) > 0:
                num_dict[array[i][j]] = (i, j)
    num_dict = sorted(num_dict.items(), key=lambda x: x[0])
    if num_dict[0][1] != (0, 0):
        num_dict.insert(('0', (0, 0)))

    def bfs(a1, b1, a2, b2):
        stack = [(a1, b1, 0)]
        visited = [(a1, b1)]
        while len(stack) > 0:
            temp = stack[0]
            for d in direct:
                if isValidate(temp[0] + d[0], temp[1] + d[1]) and (temp[0] + d[0], temp[1] + d[1]) not in visited:
                    if temp[0] + d[0] == a2 and temp[1] + d[1] == b2:
                        return temp[2] + 1
                    else:
                        stack.append((temp[0] + d[0], temp[1] + d[1], temp[2] + 1))
                        visited.append((temp[0] + d[0], temp[1] + d[1]))
            stack = stack[1:]
        return -1

    def isValidate(i, j):
        if 0 <= i <= row and 0 <= j <= col and array[i][j] != '-':
            return True
        return False

    allpath = 0
    for i in range(len(num_dict) - 1):
        temp = bfs(num_dict[i][1][0], num_dict[i][1][1], num_dict[i + 1][1][0], num_dict[i + 1][1][1])
        if temp == -1:
            return -1
        allpath += temp

    return allpath

