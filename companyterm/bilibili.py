def findRes(inputStr):
    if len(inputStr) >= 2:
        if inputStr[:2] == "00" or inputStr[2:] == "00":
            return 0
    prepre = 1
    pre = 1
    lastNum = 0
    if inputStr[0] == '0':
        pre = 0

    for i in range(1,len(inputStr)):
        preNum = int(inputStr[i-1:i+1])
        lastNum = preNum
        if 1 <= preNum <= 26:
            if inputStr[i] == '0' or lastNum == 0:
                t = pre
                pre = prepre
                prepre = t
            else:
                t = prepre + pre
                prepre = pre
                pre = t
        else:
            prepre = pre
    return pre

def reverseStr(inputStr):
    strls = inputStr.split(' ')
    for i, s in enumerate(strls):
        if len(s) % 2 == 1:
            strls[i] = s[::-1]
    return ' '.join(strls)

def Conv(picture, kernel):
    m = len(kernel)
    h = len(picture)
    w = len(picture[0])
    output = []
    for i in range(h-m+1):
        line = []
        for j in range(w-m+1):
            s = 0
            for p in range(m):
                for q in range(m):
                    s += picture[i + p][j + q] * kernel[p][q]
            line.append(min(s,255))
        output.append(line)
    return output

# import sys
# if __name__ == "__main__":
#     temp1 = list(map(int,sys.stdin.readline().strip().split()))
#     picture = []
#     for i in range(temp1[0]):
#         picture.append(list(map(int,sys.stdin.readline().strip().split())))
#     m = int(sys.stdin.readline().strip())
#     kernel = []
#     for i in range(m):
#         kernel.append(list(map(float,sys.stdin.readline().strip().split())))
#
#     res = Conv(picture,kernel)
#     for line in res:
#         print(' '.join(list(map(str,list(map(int,line))))))


import sys
if __name__ == "__main__":
    inputStr = sys.stdin.readline().strip()
    print(findRes(inputStr))

