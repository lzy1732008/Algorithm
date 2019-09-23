#快排
from jzoffer.ListNode import TreeLinkNode
def Qsort(tinput, low, high):
    i = low
    key = tinput[low]
    while low < high:
        while low < high: #这是一个易错点：左边和右边移动顺序固定，必须先移动左边，这样左边的才会指向第一个比key大的
            if tinput[low] > key: break
            low += 1
        while low < high:
            if tinput[high] <= key: break
            high -= 1
        tinput[low], tinput[high] = tinput[high], tinput[low]
    tinput[low-1], tinput[i] = tinput[i], tinput[low-1] #这是一个易错点
    return low


def QuickSort(tinput, low, high):
    if low < high:#易错点：你写成了while
        mid = Qsort(tinput, low, high)
        QuickSort(tinput, low, mid-1) #这是一个易错点，mid-1存放的是key,所以左边的递归范围只需要是low~mid-2
        QuickSort(tinput, mid, high)
    return tinput



#堆排序
#堆的建立
def DownFilter(input,start,end):
    i = start
    child = i * 2 + 1
    while child < end :
        if child + 1 < end and input[child] < input[child+1]: child += 1
        if input[child] > input[i]:
            input[child], input[i] = input[i], input[child]
            i = child
            child = i * 2 + 1
        else:
            break

def BuildHeadp(input):
    i = int(len(input)/2)
    while i >= 0:
        DownFilter(input, i, len(input))
        i -= 1
    return input



def HeapSort(input):
    BuildHeadp(input)
    for i in range(len(input)):
        input[0], input[len(input)-i-1] = input[len(input)-i-1], input[0]
        DownFilter(input,0,len(input)-i-1)
    return input


def viterbi(A,B,pi,O):
    stateNum = len(pi)
    theta = []
    phi = []
    temp = []
    for i in range(stateNum):
        temp.append(float(pi[i] * B[i][O[0]]))
    theta.append(temp)
    phi.append([0 for _ in range(stateNum)])


    for i, v in enumerate(O):
        if i == 0:
            continue
        line_t = []
        line_ph = []
        for p in range(stateNum):
            max_i = -1
            max_v = -1
            for q in range(stateNum):
                t = theta[i - 1][q] * A[q][p] * B[p][v]
                if t > max_v:
                    max_v = t
                    max_i = q
            line_t.append(max_v)
            line_ph.append(max_i)
        theta.append(line_t)
        phi.append(line_ph)

    max_i = -1
    max_v = -1
    for i, v in enumerate(theta[-1]):
        if v > max_v:
            max_v = v
            max_i = i

    last_i = max_i
    t = len(O) - 1
    while t > 0:
        print(last_i)
        last_i = phi[t][last_i]
        t -= 1


A = [
    [0.5, 0.2, 0.3],
    [0.3, 0.5, 0.2],
    [0.2, 0.3, 0.5]
]

B = [
    [0.5, 0.5],
    [0.4, 0.6],
    [0.7, 0.3]
]

pi = [0.2,0.4,0.4]

viterbi(A,B,pi,[0,1,0])


























