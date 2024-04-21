import time
import sys
from datetime import datetime

now = datetime.now()

current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

# print ('argument list', sys.argv)
print(sys.argv[1], sys.argv[2])
argument_edges = sys.argv[1]
argument_servers =sys.argv[2]
# argument_FPS =sys.argv[3]


number_of_edges=int(argument_edges)
number_of_servers=int(argument_servers)
sim_time = 0
frame_number = 0
FPS =33
# FPS = int(argument_FPS) # FPS IS EQUAL TO how many frames are generated (and target to process) per second
iteration=0
frame_completed=0

deadline_per_task=[20,33,30,25]

#for each server
allow_server_multi_tenancy=[]

#for each edge device
memory_use = []

for i in range(number_of_servers):
    allow_server_multi_tenancy.append(True)

for i in range(number_of_edges):
    memory_use.append(0)

# Resource availability for two edges and two servers
# 0 and 1 edges, 2 and 3 servers


#Stores available PUs, 
gpu_available=[]
cpu_available=[]
vpi_available=[]
for i in range(number_of_edges+number_of_servers):
    gpu_available.append(True)
    cpu_available.append(True)
    vpi_available.append(True)

# gpu_available = [[True], [True],[True], [True], [True]] 
# cpu_available = [[True], [True],[True], [True], [True]]
# vpi_available = [[True], [True],[True], [True], [True]]

# Task queues and working tasks for each edge
queue = []
for i in range(number_of_edges):
    queue.append(["motion",i])
    

#stores the TASK. Each task has 
# 0)Task name, 
# 1)Expected finish "time" of the task, 
# 2)PU, 
# 3)running device , 0,1 for nanos; 2,3 for servers
# 4)Updated or not
# 5)which device it belongs to (which edge generated the task)
# 6)Start time
working_task = []

#Profiling variables
time_frame_created=[]
time_frame_completed=[]
bandwidth_when_ask_edgeandserver=0
bandwidth_when_ask_server=0
tasks_asked_server=[0,0,0,0]
tasks_asked_waited_in_queue=[0,0,0,0]
task_processing_time=[]



server_with_rendering=[]
for i in range(number_of_servers):
    server_with_rendering.append([0]*number_of_edges)

qos_failed_tasks=[]

# Function to attempt running a task on an edge device
def attempt_run_on_edge(task_function, min_time, edge):
    print("attempt locally, task: ",task_function)
    
    if task_function == "motion": # motion prediction 
        if gpu_available[edge]:
            if min_time > 10: #10ms on GPU
                max_time=10 * (memory_use[edge]/100+1)
                device="gpu"
                return device,max_time 
        elif cpu_available[edge]:
            if min_time > 31: #31ms on CPU
                max_time=31 * (memory_use[edge]/100+1)
                device="cpu"
                return device,max_time
        return None,None
    elif task_function == "rendering": # rendering 
        if gpu_available[edge]:
            if min_time > 100: 
                max_time=100 * (memory_use[edge]/100+1)
                device="gpu"
                return device,max_time 
        elif cpu_available[edge]:
            if min_time > 200:
                max_time=200 * (memory_use[edge]/100+1)
                device="cpu"
                return device,max_time
        return None,None
    elif task_function == "decode": # decode 
        if gpu_available[edge]:
            if min_time > 11: 
                max_time=11 * (memory_use[edge]/100+1)
                device="gpu"
                return device,max_time 
        elif cpu_available[edge]:
            if min_time > 16:
                max_time=16 * (memory_use[edge]/100+1)
                device="cpu"
                return device,max_time
        return None,None
    elif task_function == "reproject": # reproject 
        if gpu_available[edge]:
            if min_time > 11: 
                max_time=11 * (memory_use[edge]/100+1)
                device="gpu"
                return device,max_time 
        if vpi_available[edge]:
            if min_time > 25:
                max_time=25 * (memory_use[edge]/100+1)
                device="vpi"
                return device,max_time
        elif cpu_available[edge]:
            if min_time > 28:
                max_time=28 * (memory_use[edge]/100+1)
                device="cpu"
                return device,max_time
        return None,None
    else:
        print(str("Edge: wrong task is given: "+task_function))
        return None,None

# Function to attempt running a task on a server
def attempt_run_on_server(task_function, min_time, server):
    max_time = 999
    device="none"

    if task_function == "motion": # motion prediction 
        if gpu_available[server]:
            if min_time > 5: #10ms on GPU
                max_time=5
                device="gpu"
                return device,max_time
        if allow_server_multi_tenancy[server-number_of_edges]:
            if min_time > 10: 
                max_time=10
                device="gpu"
                print("multi-tenancy can be on gpu")
                # allow_server_multi_tenancy[server-number_of_edges]=False
                return device,max_time
        if cpu_available[server]:
            if min_time > 15: #31ms on CPU
                max_time=15
                device="cpu"
                return device,max_time
        return None,None
    if task_function == "rendering": # rendering
        print("gpu availability: ",gpu_available )
        print("allow_server_multi_tenancy: ",allow_server_multi_tenancy)
        if gpu_available[server]:
            if min_time > 20: 
                max_time=20
                device="gpu"
                print("server rendering gpu")
                return device,max_time
        if allow_server_multi_tenancy[server-number_of_edges]:
            if min_time > 30: 
                max_time=30
                device="gpu"
                print("multi-tenancy can be on gpu")
                # allow_server_multi_tenancy[server-number_of_edges]=False
                return device,max_time
        if cpu_available[server]:
            if min_time > 40:
                max_time=40
                device="cpu"
                return device,max_time
        return None,None
    elif task_function == "decode": # rendering 
        if gpu_available[server]:
            if min_time > 3: 
                max_time=3
                device="gpu"
                return device,max_time
        if cpu_available[server]:
            if min_time > 8:
                max_time=8
                device="cpu"
                return device,max_time
        return None,None
    elif task_function == "reproject": # rendering 
        if gpu_available[server]:
            if min_time > 9: 
                max_time=9
                device="gpu"
                return device,max_time
        if cpu_available[server]:
            if min_time > 20:
                max_time=20
                device="cpu"
                return device,max_time 
        return None,None
    else:
        print(str("Server: wrong task is given : "+task_function))
        return None,None


def find_next_task(task_function):
    if task_function == "motion":
        return "rendering"
    if task_function == "rendering":
        return "decode"
    if task_function == "decode":
        return "reproject"
    if task_function == "reproject":
        return "done"

def find_task_completed_sim_time(working_task,next_frame_time):
    shortest_time= sys.maxsize
    index=999
    print("Working task: ",working_task)
    for i in range(len(working_task)):
        if shortest_time > working_task[i][1]:
            shortest_time = working_task[i][1]
            index=i
    
    if next_frame_time > shortest_time: # next task will start (the task after working_task[index][0])
        print(str("find next task: "+working_task[index][0]))
        return working_task[index],shortest_time,index
    else: # next frame will start
        return "next frame",next_frame_time,index

#Device is unnecessary/redundany right now, use as 0 for now.
def assign_task_to_PU(edge_or_server,device,PU):
    if PU == "gpu":
        gpu_available[edge_or_server]=False
    elif PU == "cpu":
        cpu_available[edge_or_server]=False
    elif PU == "vpi":
        vpi_available[edge_or_server]=False
    else: 
        print("Wrong device assigned")
        exit()

#Device is unnecessary/redundany right now, use as 0 for now.
def release_task_from_PU(edge_or_server,device,PU):
    if PU == "gpu":
        gpu_available[edge_or_server]=True
    elif PU == "cpu":
        cpu_available[edge_or_server]=True
    elif PU == "vpi":
        vpi_available[edge_or_server]=True
    else: 
        print("Wrong device released")
        exit()

def check_device_availability():
    temp=2
    print("CPU available: ",cpu_available)
    print("GPU available: ",gpu_available)
    print("VPI available: ",vpi_available)

def bandwidth(task_function, PU):
    # return 0
    if task_function == "motion":
        if PU == "gpu":
            return 50
        if PU == "cpu":
            return 25
    if task_function == "rendering":
        if PU == "gpu":
            return 50
        if PU == "cpu":
            return 25
    if task_function == "decode":
        if PU == "gpu":
            return 50
        if PU == "cpu":
            return 25
    if task_function == "reproject":
        if PU == "gpu":
            return 50
        if PU == "cpu":
            return 25
        if PU == "vpi":
            return 25
    else:
        print("wrong bandwidth call")
        return 0

def update_expected_finish_time_edge(edge_id):
    print("Updating edge times")
    print("Before edge update: ",working_task)
    updated=False
    for i in range(len(working_task)):
        if  ((working_task[i][3] == edge_id) and (working_task[i][4]==False)):
            print("working task i: ", working_task[i], i)
            working_task[i][1] = ((working_task[i][1]-sim_time) * (memory_use[edge_id]/100+1)) + sim_time
            working_task[i][4] = True
            updated=True
    print("After edge update: ", working_task)
    return updated

def update_expected_finish_time_server(server_id):
    print("Updating server times")
    print("Before server update: ",working_task)
    updated=False
    # tasks_in_servers_gpu=[]
    # for i in range(number_of_servers):
    #     tasks_in_servers_gpu.append(0)
    for i in range(len(working_task)):
        #Slows down only GPU on server
        # if working_task[i][3] >= (number_of_edges-1):
        #     tasks_in_servers_gpu[working_task[i][3]-number_of_edges]+=1
        if  ((working_task[i][3] == server_id) and (working_task[i][2]=="gpu") and (working_task[i][4]==False)):
            print("target working task: ",working_task[i])
            working_task[i][1]= ((working_task[i][1]-sim_time) * (50/100+1)) + sim_time
            working_task[i][4] = True
            updated=True
    # if tasks_in_servers_gpu[server_id-number_of_edges]>1:
    #     allow_server_multi_tenancy[server_id-number_of_edges]=False
    #     updated=True
    # else:
    #     allow_server_multi_tenancy[server_id-number_of_edges]=True

    print("After server update: ", working_task)

    return updated


def update_multi_tenancy_after_assignment():
    tasks_in_servers_gpu=[]
    for i in range(number_of_servers):
        tasks_in_servers_gpu.append(0)
    for i in range(len(working_task)):
        if working_task[i][3] >= (number_of_edges):
            tasks_in_servers_gpu[working_task[i][3]-number_of_edges]+=1
    print("updating multi tenancy: ", working_task)
    print("before updating multi tenancy: ", allow_server_multi_tenancy)
    for server_id in range(number_of_servers): 
        if tasks_in_servers_gpu[server_id]>1:
            allow_server_multi_tenancy[server_id]=False
        else:
            allow_server_multi_tenancy[server_id]=True
    
    print("after updating multi tenancy: ", allow_server_multi_tenancy)


def task_index(task_function):
    if task_function == "motion":
        return 0
    if task_function == "rendering":
        return 1
    if task_function == "decode":
        return 2
    if task_function == "reproject":
        return 3

# Function to process tasks for a specific edge
# def process_tasks_for_edge(queue, working_task, edge_index):
#     if len(queue) > 0:
#         current_task = queue.pop(0)
#         PU, shortest_time = attempt_run_on_edge(current_task, 33, edge_index)
#         if PU is not None:
#             # Handle task assignment on edge
#             working_task.append([current_task, sim_time + shortest_time, PU, edge_index])
#         else:
#             # If not possible on edge, try on servers
#             for server_index in range(2, 4):
#                 PU, shortest_time = attempt_run_on_server(current_task, 33, server_index)
#                 if PU:
#                     working_task.append([current_task, shortest_time + sim_time, PU, server_index])
#                     break
#     return queue, working_task

# Main simulation loop
# while True:
#     if sim_time < 100000:
#         # Process tasks for both edges
#         queue_edge_1, working_task_edge_1 = process_tasks_for_edge(queue_edge_1, working_task_edge_1, 0)
#         queue_edge_2, working_task_edge_2 = process_tasks_for_edge(queue_edge_2, working_task_edge_2, 1)

#         # Handle task completion and updates for both edges
#         # ...

def available_PU_found(PUs):
    for i in range(len(PUs)):
        if PUs[i] != None:
            return True
    return False



def available_devices_to_run():
    for task_waiting in range(len(queue)):
        PU, shortest_time=attempt_run_on_edge(current_task[0],FPS,current_task[1])
        if PU != None:
            return True
        else:
            for server_id in range(number_of_edges,number_of_edges+number_of_servers):  
                PU, shortest_time=attempt_run_on_server(current_task[0],FPS,server_id)
                if PU != None:
                    return True
    return False

sim_time_reminder_server=sim_time
time_interval_switched=0
sim_time_reminder=sim_time

tasks_asked_waited_in_queue_df=[0,0,0,0]

assign_tasks=[]
tasks_assigned_same_edge=[]

tasks_failed_to_assign_edge=[]
for i in range(number_of_edges):
    tasks_failed_to_assign_edge.append(0)
while True:
    if (sim_time < 10000):
        print("Sim time: ", sim_time)
        print("Start, working tasks: ",working_task)
        print("Queue: ", queue)
        print(check_device_availability())
        if sim_time_reminder != sim_time:
            sim_time_reminder = sim_time
            time_interval_switched+=1
        if len(queue)>0:
            current_task=queue.pop(0)
            print("Current Task:", current_task)
            PU, shortest_time=attempt_run_on_edge(current_task[0],deadline_per_task[task_index(current_task[0])],current_task[1])
            
            if PU != None: #we assigned task to one PU
                assign_task_to_PU(current_task[1],0,PU)   #edge/device index, device number, PU
                print("Assigned the task locally device")
                memory_use[current_task[1]]+=bandwidth(current_task[0],PU)
                updated=update_expected_finish_time_edge(current_task[1])
                working_task.append([current_task[0],sim_time+shortest_time,PU,current_task[1],updated,current_task[1],sim_time]) # 0 is edge
                print("task:", current_task)
                print("PU:", PU)

                        #Checking other edges in the network
            if PU == None:
                tasks_failed_to_assign_edge[0]+=1 #first edge device could not assign the task
                #keeps tuple of PU and shortest_time for each edge device (except id 0)
                shortest_times_edges=[]
                PUs_edges=[]
                #Ask each edge device
                for i in range (1,number_of_edges):
                    #print("Entered none, find other edges available i:",i)
                    PU, shortest_time=attempt_run_on_edge(current_task[0],deadline_per_task[task_index(current_task[0])],current_task[1])
                    shortest_times_edges.append([PU,shortest_time])
                shortest_time=999
                index=-1
                for edge_id in range (0,number_of_edges-1):
                    if shortest_times_edges[edge_id][1]==None:
                        continue
                    if shortest_time>shortest_times_edges[edge_id][1]:
                        shortest_time=shortest_times_edges[edge_id][1]
                        index=edge_id
                if index == -1:
                    for i in range(1,number_of_edges):
                        tasks_failed_to_assign_edge[i]+=1
                #print("edge index: ", index+1)
                if index != -1:
                    assign_task_to_PU(index+1,0,shortest_times_edges[index][0])   #edge/device index, device number, PU
                    #print("current_task[0] and PU: ", current_task[0],shortest_times_edges[index][0])
                    memory_use[index+1]+=bandwidth(current_task[0],shortest_times_edges[index][0])
                    updated=update_expected_finish_time_edge(index+1)
                    working_task.append([current_task[0],sim_time+shortest_time,shortest_times_edges[index][0],index+1,updated,frame_number,sim_time]) # 0 is edge
                    #print("Edge id: ", index+1," assigned task:", current_task)
                    updated=update_expected_finish_time_edge(index+1)
                else: 
                    PU=None
            
            #checking servers
            if PU == None:  
                print("Entered none, find another device ")
                if current_task[0] == "motion":
                    tasks_asked_server[0]+=1
                if current_task[0] == "rendering":
                    tasks_asked_server[1]+=1
                if current_task[0] == "decode":
                    tasks_asked_server[2]+=1
                if current_task[0] == "reproject":
                    tasks_asked_server[3]+=1
                PUs=[]
                shortest_times=[]
                for server_id in range(number_of_edges, number_of_edges+number_of_servers):
                    print("asking to Server :", current_task, " served id: ",server_id)
                    PU, shortest_time=attempt_run_on_server(current_task[0],deadline_per_task[task_index(current_task[0])],server_id)
                    PUs.append(PU)
                    shortest_times.append(shortest_time)
                
                if sim_time_reminder_server!=sim_time:
                    bandwidth_when_ask_edgeandserver+=1.3
                    bandwidth_when_ask_server+=0.3
                    sim_time_reminder_server=sim_time
                    #Availab PU not found enters if
                if not(available_PU_found(PUs)):
                    print("No available PU found: ", current_task)
                    qos_failed_tasks.append(current_task)
                    if current_task[0] == "motion":
                        tasks_asked_waited_in_queue[0]+=1
                    if current_task[0] == "rendering":
                        tasks_asked_waited_in_queue[1]+=1
                    if current_task[0] == "decode":
                        tasks_asked_waited_in_queue[2]+=1
                    if current_task[0] == "reproject":
                        tasks_asked_waited_in_queue[3]+=1
                    if sim_time_reminder!=sim_time:
                        if current_task[0] == "motion":
                            tasks_asked_waited_in_queue_df[0]+=1
                        if current_task[0] == "rendering":
                            tasks_asked_waited_in_queue_df[1]+=1
                        if current_task[0] == "decode":
                            tasks_asked_waited_in_queue_df[2]+=1
                        if current_task[0] == "reproject":
                            tasks_asked_waited_in_queue_df[3]+=1
                else:
                    index=0
                    print("Shortest time: ", shortest_times)
                    shortest_time=999
                    for server_id in range (len(PUs)):
                        if shortest_times[server_id]==None:
                            continue
                        if shortest_time>shortest_times[server_id]:
                            index=server_id
                            shortest_time=shortest_times[index]
                    print("Index: ",index)
                    assign_task_to_PU(index+number_of_edges,0,PUs[index])   #edge/device index, device number, PU
                    print("Served assigned:", current_task)
                    print ("index: ",index, "  allow_server_multi_tenancy: ", allow_server_multi_tenancy, " gpu_available: ",gpu_available)
                    # if allow_server_multi_tenancy[index] and (gpu_available[index+number_of_edges]==False):
                    #     allow_server_multi_tenancy[index]=False
                    #     print("allow_server_multi_tenancy updated: ",allow_server_multi_tenancy)
                    updated=update_expected_finish_time_server(index+number_of_edges)
                    
                    working_task.append([current_task[0],shortest_times[index]+sim_time,PUs[index],index+number_of_edges,updated,current_task[1],sim_time]) 
                    update_multi_tenancy_after_assignment()
                    if current_task[0]== "rendering":
                        server_with_rendering[index][current_task[1]]+=1

        print("memory use at assignment:", memory_use)
        task,time,index=find_task_completed_sim_time(working_task,FPS*(frame_number+1))
        print("Before Decision of next task: ", task)
        print("Queue: ",queue)
        print("Working task: ", working_task)
        if task == "next frame":
            print("Next frame will be called")
            frame_number+=1
            sim_time=frame_number*FPS
            for i in range(number_of_edges):
                queue.append(["motion",i])
            time_frame_created.append(sim_time)
        elif (len(queue) > 0) and available_devices_to_run():
            temp = 2 #redundant
            print("Continue is applied")
        else:
            task_processing_time.append([task[1],task[6]])
            print("Task will be completed: ",task)
            release_task_from_PU(task[3],0,task[2])
            # if (task[0]== "rendering") and (task[3] > (number_of_edges-1)):
            #     allow_server_multi_tenancy[task[3]-number_of_edges]=True 
            if task[3] < number_of_edges: #Edge devices (both) 
                memory_use[task[3]]-=bandwidth(task[0],task[2])
            next_task_name=find_next_task(task[0])
            print("Next task found: ", next_task_name)
            print("index of the task: ", index)
            if next_task_name != "done": # don't attach done, frame processing is completed.
                next_task=[next_task_name,task[5]]
                queue.append(next_task)
                print("queue added: ",queue)
                sim_time=time
            else: 
                task,time,index=find_task_completed_sim_time(working_task,FPS*(frame_number+1))
                frame_completed+=1
                sim_time=time
                print("qos_failed_tasks: ",qos_failed_tasks)
                for tasks in range(len(qos_failed_tasks)):
                    queue.append(qos_failed_tasks[tasks])
                qos_failed_tasks=[]
                print("FRAME IS COMPLETED, time: ",frame_completed, sim_time)
                time_frame_completed.append(sim_time)
            print("memory use after assignment:",memory_use)
            working_task.pop(index)

        print("-------------------------------------------------------")
        iteration+=1
        if ((frame_completed > 100*number_of_edges) or (5000 < sim_time)):
            print("bandwidth_when_ask_edgeandserver: ",bandwidth_when_ask_edgeandserver)
            print("Overhead with edgeandserver: %",100*bandwidth_when_ask_edgeandserver/sim_time/number_of_edges)
            print("bandwidth_when_ask_server: ",bandwidth_when_ask_server)
            
            print("Overhead with server: %",100*bandwidth_when_ask_server/sim_time/number_of_edges)
            # print("server_with_rendering: ",server_with_rendering)
            print("tasks_asked_server: ",tasks_asked_server)
            print("tasks_asked_waited_in_queue: ",tasks_asked_waited_in_queue)
            print("tasks_asked_waited_in_queue_df: ",tasks_asked_waited_in_queue_df)
            # print("task_processing_time: ",task_processing_time)
            print("frame: ",frame_number)
            print("iteration: ",iteration)
            print("time_interval_switched: ",time_interval_switched)
            print("total task waited in queue: ", tasks_asked_waited_in_queue[0]+tasks_asked_waited_in_queue[1]+tasks_asked_waited_in_queue[2]+tasks_asked_waited_in_queue[3])
            print("Sim time: ", sim_time)
            print("")
            break


    else:
        break