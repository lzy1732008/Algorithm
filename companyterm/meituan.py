#黑白矩阵 最小修改数
def minNums(m,n,nums):
    same00 = []
    same01 = []
    same10 = []
    same11 = []
    for i in range(m):
        for j in range(n):
            if i % 2 == 0 and j % 2 == 0:
                same00.append(nums[i][j])
            elif i % 2 == 0 and j % 2 == 1:
                same01.append(nums[i][j])
            elif i % 2 == 1 and j % 2 == 0:
                same10.append(nums[i][j])
            elif i % 2 == 1 and j % 2 == 1:
                same11.append(nums[i][j])

    same00 = sorted(same00)
    same11 = sorted(same11)
    same10 = sorted(same10)
    same01 = sorted(same01)
    dict00 = findMost(same00)
    dict11 = findMost(same11)
    dict10 = findMost(same10)
    dict01 = findMost(same01)
    mins = 100000000
    i = len(dict00) - 1
    j = len(dict01) - 1
    while i >= 0 and j >= 0:
        if dict00[i][0] == dict01[j][0]:
            if dict00[i][1] < dict01[j][1]:
                i -= 1
            else:
                j -= 1
        else: break




def findMost(nums):
    lendict = {}
    start = 0
    for i in range(len(nums)):
        if nums[i] != nums[start]:
            lendict[str(nums[start])] = i - start
            start = i
    p = lendict.items()
    lendict = sorted(p,key=p[1])
    return lendict

def longestSubSeq(strA,strB):
    startA,end = 0,1
    seq = ''
    while startA < len(strA):
        if strA[startA] == strA[0]:
            while end < len(strB) and startA + end < len(strA):
                if strA[startA+end] == strB[end]:
                    end += 1
                else:
                    break
            if startA + end < len(strA) and strA[startA+end] == strB[end]:
                end += 1
            if end > len(seq):
                seq = strB[:end]
        startA += 1
    return seq

def fun100(input,limit):
    lines = []
    words = input.split(' ')
    localStr = ''
    for word in words:
        if len(localStr+' '+word) > limit+1:
            lines.append(localStr)
            localStr = word
        elif len(localStr+' '+word) == limit+1:
            lines.append(localStr+' '+word)
            localStr = ''
        else:
            localStr += ' '+ word
    lines.append(localStr)
    print(lines)

def log(text):
    def wapper(func):
        def decorator(*args, **kw):
            print('function name:', func.__name__)
            print("this is my input text:",text)
            return func(*args, **kw)
        return decorator
    return wapper



@log('happy')
def happyFunction(input):
    print("HappyFunctionInput",input)







#
# nums1 = [3,1,4,5,2]
# nums2 = [4,7,2,3,4]
# nums3 = [8,5,3,6,7]
# res = topM(nums1,nums2,nums3,5,3)
# print(res)


import sys
if __name__ == "__main__":
    n1 = int(sys.stdin.readline().strip())
    strA = list(map(int,sys.stdin.readline().strip().split(' ')))
    n2 = int(sys.stdin.readline().strip())
    strB = list(map(int,sys.stdin.readline().strip().split(' ')))
    res = longestSubSeq(strA,strB)
    print(res)

