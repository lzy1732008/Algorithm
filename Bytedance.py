class Point:
    x = 0
    y = 0
    def __init__(self,x,y):
        self.x = x
        self.y = y




def maxPoint(points):
    sort_x = sorted(points, key=lambda t:t.x)
    sort_y = sorted(points, key=lambda t:t.y)
    max_x = []
    max_y = []
    max_p = []
    while sort_x or sort_y:
        if sort_x:
            point_x = sort_x[-1]
            if point_x not in max_y:
                flag = 0
                for point in max_x:
                    if point.y > point_x.y:
                        flag = 1
                        break
                if flag == 0:
                   max_x.append(point_x)
                   max_p.append(point_x)
            sort_x.pop()

        if sort_y:
            point_y = sort_y[-1]
            if point_y not in max_p:
                flag = 0
                for point in max_x:
                    if point.x > point_y.x:
                        flag = 1
                        break
                if flag == 0:
                   max_y.append(point_y)
                   max_p.append(point_y)
            sort_y.pop()
    max_p = sorted(max_p, key=lambda t:t.x)
    return max_p



import sys
if __name__ == "__main__":
    num = int(sys.stdin.readline().strip())
    points = []
    for i in range(num):
        lines = list(map(int, sys.stdin.readline().strip().split()))
        p = Point(lines[0],lines[1])
        points.append(p)
    maxv = maxPoint(points)
    for i in range(len(maxv)):
        print(str(maxv[i].x)+' '+str(maxv[i].y))