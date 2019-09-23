def coinCount(n):
    rest = 1024 - n
    nums = 0
    coinls = [64,16,4,1]
    for n in coinls:
        temp = int(rest/n)
        nums += temp
        rest = rest - n * temp
        if rest == 0:
            break
    return nums

def checkstr(strs):
    newstr = []
    for s in strs:
        i = 0
        while i < len(s):
            flag = 0
            if i+3 <= len(s):
                subs_3 = s[i:i + 3]
                # 首先判断第一个
                if subs_3[0] == subs_3[1] and subs_3[1] == subs_3[2]:
                    s = s[:i] + subs_3[:-1] + s[i+3:]
                    flag = 1
                    # 将最后一个删除

                if i + 4 <= len(s):
                    subs_4 = s[i:i + 4]
                    if subs_4[0] == subs_4[1] and subs_4[2] == subs_4[3]:
                        # 将最后一个删除
                        s = s[:i] + subs_4[:-1] + s[i + 4:]
                        flag = 1
            if flag == 0:
                i += 1
        #将新的s添加到newstr中
        newstr.append(s)
    return newstr


def rewards(scores):
    r = [1] * len(scores)
    while True:
        nums = 0
        for i in range(len(scores)):
            left = (i - 1 + len(scores)) % len(scores)
            right = (i + 1) % len(scores)
            temp = -1
            if scores[i] > scores[left]:
                if r[i] <= r[left]:
                    temp = r[left] + 1

            if scores[i] > scores[right]:
                if r[i] <= r[right]:
                    temp = max(r[right] + 1, temp)
            if temp > -1:
                r[i] = temp
                nums += 1
        if nums == 0:
            break
    return sum(r)

import sys
if __name__ == "__main__":
    num = int(sys.stdin.readline().strip())
    result = []
    scoresls = []
    for i in range(num):
        s_nums = int(sys.stdin.readline().strip())
        scores = list(map(int,sys.stdin.readline().strip().split()))
        scoresls.append(scores)
    for scores in scoresls:
        result.append(rewards(scores))
    for t in result:
        print(t)

