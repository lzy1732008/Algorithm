def fun1(a,b):
    b = sorted(b)
    if len(a) == 0:
        return a
    for i in range(len(a)-1):
        if a[i] >= a[i + 1]:
            for j in reversed(range(len(b))):
                if i + 2 < len(a):
                    if b[j] > a[i] and b[j] < a[i + 2]:
                        a[i+1] = b[j]
                        return a
                else:
                    if b[j] > a[i]:
                        a[i+1] = b[j]
                        return a
    return "NO"


def fun2(strls):
    str_dict={"A":[],"B":[],"C":[],"D":[],"E":[],"F":[],"G":[],"H":[],"I":[],"J":[],"K":[],"L":[],"M":[],"N":[],
              "O":[],"P":[],"Q":[],"R":[],"S":[],"T":[],"U":[],"V":[],"W":[],"X":[],"Y":[],"Z":[]}
    count = len(strls) - 1
    def BFS(count,s_dict,index):
        if count == 0:
            return True
        if len(s_dict[strls[index][-1]]) <= 1:
            return False
        s_dict[strls[index][-1]].remove(str(index)+"end")
        leafs = s_dict[strls[index][-1]]
        for leaf in leafs:
            if str(leaf).find("start") > 0:
                s_dict[strls[index][-1]].remove(leaf)
                new_index = int(leaf[:-5])
                temp = count - 1
                if not BFS(temp,s_dict,new_index):
                    s_dict[strls[index][-1]].append(leaf)
                else:
                    return True
        s_dict[strls[index][-1]].append(str(index)+"end")
        return False

    for i in range(len(strls)):
        s = strls[i]
        if i > 0:
           str_dict[s[0]].append(str(i)+"start")
        str_dict[s[-1]].append(str(i)+"end")
    return BFS(count,str_dict,index=0)


def fun3(n,m,costs,relations):
    pathls = []
    def BFS(past_ls,current_relations,current_path,current_cost):
        if len(current_path) == n:
            pathls.append([current_path,current_cost])
            return True

        caldidate = [i for i in range(1,n+1)]
        for relation in current_relations:
            end = int(relation.split(' ')[-1])
            if end in caldidate or end in past_ls:
                caldidate.remove(end)
        for e in past_ls:
            caldidate.remove(e)

        if len(caldidate) == 0:
            return False#有待

        for c in caldidate:
            temp_path = [e for e in current_path]
            temp_cost = [e for e in current_cost]
            temp_cost[c-1] = max(current_cost) + costs[c - 1]
            temp_path.append(c)
            temp_relations = []
            temp_past = [e for e in past_ls]
            temp_past.append(c)
            for r in current_relations:
                t = r.split(' ')
                if int(t[0]) != c:
                    temp_relations.append(r)

            BFS(temp_past,temp_relations,temp_path,temp_cost)

    current_path = []
    current_cost = [0 for i in range(n)]
    current_relations = [r for r in relations]
    past_ls = []
    BFS(past_ls, current_relations,current_path,current_cost)
    min_cost = 100000000000000000000000
    for path in pathls:
        if sum(path[1]) < min_cost:
            min_cost = sum(path[1])
    res = []
    for path in pathls:
        if sum(path[1]) == min_cost:
            res.append(' '.join(list(map(str,path[0]))))
    res = sorted(res)
    return res[0]



#
import sys
if __name__ == "__main__":
      # nums = list(map(int, sys.stdin.readline().strip().split()))
      # n = nums[0]
      # m = nums[1]
      # costs = list(map(int, sys.stdin.readline().strip().split()))
      # relations = []
      # for i in range(n):
      #     relations.append(sys.stdin.readline())
      # res = fun3(n,m,costs,relations)
      # print(res)


    a = list(map(int,sys.stdin.readline().strip().split()))
    b = list(map(int, sys.stdin.readline().strip().split()))
    maxv = fun1(a,b)
    s = ''
    for n in maxv:
        s += str(n) + ' '
    print(s[:-1])
    #
    # strls = list(sys.stdin.readline().strip().split())
    # res = fun2(strls)
    # if res:
    #     print("true")
    # else:
    #     print("false")






