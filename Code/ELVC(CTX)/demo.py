a, b = map(int, input().split())

line = input().split()
list=[]

for num in line:
    if int(num)%b==0:
        list.append(int(num))
list.sort()
for i in list:
    print(i, end=" ")






