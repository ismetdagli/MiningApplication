scores = open("edge_server_count.log", "r")

y=[]
x=[]
line_count=0
iteration=0
qos_failed=0
for line in scores:
  if "iteration" in line:
    double_dot_index=line.index(":")
    iteration=int(line[double_dot_index+1:len(line)-1])
    # print(iteration,end=' ')
  if "total task waited in queue:" in line:
    double_dot_index=line.index(":")
    qos_failed=int(line[double_dot_index+1:len(line)-1])
    # print(qos_failed,end=' ')
    print(qos_failed/iteration,end=' ')
    line_count+=1
    if (line_count%9)==0:
        print("")

# scores = open("edge_server_count.log", "r")  
# line_count=0
# iteration=0
# qos_failed=0
# for line in scores:
#   if "iteration" in line:
#     double_dot_index=line.index(":")
#     iteration=int(line[double_dot_index+1:len(line)-1])
#     print(iteration,end=' ')
#     line_count+=1
#     if (line_count%9)==0:
#         print("")

# scores = open("edge_server_count.log", "r")  
# line_count=0
# iteration=0
# qos_failed=0
# for line in scores:
#   if "tal task waited in queue" in line:
#     double_dot_index=line.index(":")
#     iteration=int(line[double_dot_index+1:len(line)-1])
#     print(iteration,end=' ')
#     line_count+=1
#     if (line_count%9)==0:
#         print("")