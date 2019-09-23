class Solution:
    def drop(self,matrix):
        for i, v1 in enumerate(matrix):
            for j, v2 in enumerate(v1):
                if v2 == 0 and i > 0:
                    matrix[i][j] = matrix[i - 1][j]
                    matrix[i - 1][j] = 0

    def remove(self,matrix, site):
        queue = [site]
        value = matrix[site[0]][site[1]]
        while len(queue) > 0:
            temp = queue[0]
            matrix[temp[0]][temp[1]] = 0
            if temp[0] >= 1 and matrix[temp[0] - 1][temp[1]] == value and (temp[0] - 1, temp[1]) not in queue:
                queue.append((temp[0] - 1, temp[1]))
            if temp[0] <= 3 and matrix[temp[0] + 1][temp[1]] == value and (temp[0] + 1, temp[1]) not in queue:
                queue.append((temp[0] + 1, temp[1]))
            if temp[1] >= 1 and matrix[temp[0]][temp[1] - 1] == value and (temp[0], temp[1] - 1) not in queue:
                queue.append((temp[0], temp[1] - 1))
            if temp[1] <= 3 and matrix[temp[0]][temp[1] + 1] == value and (temp[0], temp[1] + 1) not in queue:
                queue.append((temp[0], temp[1] + 1))
            queue = queue[1:]

    def score(self,matrix):
        score = [[1 for _ in range(5)] for _ in range(5)]
        site = (-1, -1)
        maxScore = -100000
        for i, v1 in enumerate(matrix):
            for j, v2 in enumerate(v1):
                if v2 != 0:
                    if j - 1 >= 0 and matrix[i][j - 1] == v2:
                        score[i][j] += score[i][j - 1]
                    if i - 1 >= 0 and matrix[i - 1][j] == v2:
                        score[i][j] += score[i - 1][j]

                    if score[i][j] > maxScore:
                        site = (i, j)
                        maxScore = score[i][j]

        return site

    def xxl(self,matrix):
        while 1:
            site = self.score(matrix)
            if site == (-1, -1):
                break
            self.remove(matrix, site)

        print(matrix)

S = Solution()
while 1:
    a = []
    for _ in range(5):
        s = input()
        line = []
        if s != "":
            for x in s.split():
                line.append(int(x))
        else:
            break
        a.append(line)
    if len(a) == 0:
        break
    res = S.xxl(a)
    print(res)
