def fun1(staffs, capacity):
    remain = len(staffs) % capacity
    staffs = list(reversed(staffs))
    staffs[:remain] = reversed(staffs[:remain])
    point = remain
    while point + capacity <= len(staffs):
        end = point + capacity
        staffs[point:end] = reversed(staffs[point:end])
        point = end
    return staffs


import sys

def unzipStr(zipStr):
    stack = []
    point = 0
    while point < len(zipStr):
        if zipStr[point] in ['0','1','2','3','4','5','6','7','8','9']:
            end = point + 1
            while end < len(zipStr):
                if zipStr[end] in ['0','1','2','3','4','5','6','7','8','9']:
                    end += 1
                else:
                    break

            num = int(zipStr[point: end])
            stack_top = stack[-1]
            inner_num = []
            if stack_top == ')':
                while len(stack) > 0 and stack_top != '(':
                    stack_top = stack.pop()
                    inner_num.append(stack_top)
                inner_num = list(reversed(inner_num))[1:-1]
                new_num = ''.join(inner_num * num)
                stack.append(new_num)

            else:
                new_num = stack.pop() * num
                stack.append(new_num)

            point = end

        else:
            stack.append(zipStr[point])
            point += 1
    print(''.join(stack))


def fun2(matrix,k):
    def DFS(i,j,current_k,currentLen,maxLen):
        print(i,j)
        maxLen = max(currentLen, maxLen)
        if i + 1 < len(matrix):
            if matrix[i + 1][j] > matrix[i][j]:
                maxLen = DFS(i + 1, j, current_k, currentLen+1, maxLen)
            elif current_k > 0:
                maxLen = DFS(i + 1, j, current_k - 1, currentLen + 1, maxLen)
            else:
                pass

        if j + 1 < len(matrix):
            if matrix[i][j + 1] > matrix[i][j]:
                maxLen = DFS(i, j + 1, current_k, currentLen+1, maxLen)
            elif current_k > 0:
                maxLen = DFS(i, j + 1, current_k - 1, currentLen + 1, maxLen)
            else:
                pass

        if i - 1 >= 0:
            if matrix[i - 1][j] > matrix[i][j]:
                maxLen = DFS(i - 1, j, k, currentLen+1, maxLen)
            elif current_k > 0:
                maxLen = DFS(i - 1, j, current_k - 1, currentLen + 1, maxLen)
            else:
                pass

        if j - 1 < len(matrix):
            if matrix[j - 1][j] > matrix[i][j]:
                maxLen = DFS(j - 1, j, current_k, currentLen+1, maxLen)
            elif current_k > 0:
                maxLen = DFS(j - 1, j, current_k - 1, currentLen + 1, maxLen)
            else:
                pass
        return maxLen
    maxLen = DFS(i=0,j=0,current_k=k,currentLen=1,maxLen=1)
    print(maxLen)


# maxtrix = [[1, 3, 3],[2, 4, 9],[8, 9, 2]]
# k = 1
# fun2(maxtrix,k)

def fun6(n,k):
    count = [0 for _ in range(n)]
    count[0] = 1
    t = 0
    sum = 1
    while t < k:
        newcount = []
        for i in range(n):
            newcount.append(sum - count[i])
        count = newcount
        t += 1
        sum = (n - 1)* sum
    return count[0] % 1000000007

def fun9(n,persons):
    k = (int)((n - 1)/3) + 1
    left = 0
    mid = 2 * (k - 1) - 1
    index = 0
    for i in range(2 * k - 1):
        if mid > 0:
            print(left * ' ' + persons[index] + mid * ' ' + persons[index+1])
            index += 2
        else:
            print(left * ' ' + persons[index])
            index += 1
        if left < k - 1:
            left += 1
            mid = max(0, mid - 2)




import sys
if __name__ == "__main__":
    n= int(sys.stdin.readline().strip())
    persons = sys.stdin.readline().strip()
    fun9(n,persons)
