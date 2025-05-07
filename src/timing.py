from time import perf_counter as pc
import json
import math
import subprocess
import slurm_helper as h


def time_this(func,args,store_in=None,ret_output=False,label="result"):
    """takes a function and its arguements and times how long they take to execute.
    --------
    ARGS
    ---------
    func (function): some function that has been written
    args (list): list of various arguements in order to be inputted
    store_in (string): None if you don't want to store the time, or the filepath
        where you want to store it if you do.
    ret_output (bool): whether or not to return the output of the inputted function.
    ---------
    RETURNS
    ---------
    t (float): time required to run
    out (varies): returns this if specified. It is the output of the function
        that was timed
    """
    start = pc()
    out = func(*args)
    end = pc()
    t = end-start
    if store_in is not None:
        with open(store_in,'a') as f:
            f.write(label + ":" + str(t) + '\n')
    if ret_output:
        return out, t
    else: 
        return t

def serialize(data):
    return json.dumps(data)

def deserialize(data):
    return json.loads(data)

def prep_scatter(data_list,tot_threads,verbose=False):
    """prepares a list to be scattered among a certain number of threads 
    for parallelization. Total threads can also be the total nodes."""
    # make one less so the divisor matrix empty list addition doesn't go over the allocated
    # number of nodes.
    tot_threads -= 1
    # create lists with both methods
    # NOTE: the empty list is for the calcualtion of the divisor matrix.
    num_in_each = int(len(data_list)/tot_threads)
    scatter_list1 = [data_list[j*num_in_each:(j+1)*num_in_each] if j != (tot_threads-1) else data_list[j*num_in_each:] for j in range(tot_threads)]
    scatter_list1.append([])
    num_in_each = math.ceil(len(data_list)/tot_threads)
    scatter_list2 = [data_list[j*num_in_each:(j+1)*num_in_each] if j != (tot_threads-1) else data_list[j*num_in_each:] for j in range(tot_threads)]
    scatter_list2.append([])
    # check which is better
    diff1,diff2 = abs(len(scatter_list1[-2])-len(scatter_list1[-3])), abs(len(scatter_list2[-2]) - len(scatter_list2[-3]))
    if verbose: print(f"diff1: {diff1}\ndiff2: {diff2}")

    if diff1 <= diff2:
        if verbose: print("using full remainder method")
        return scatter_list1, True
    else:
        if verbose: print("using remainder distribution method")
        return scatter_list2, False

def SearchGraphName(string_to_mathc = None):
    """takes a string to match and prints out files that match then prompts the user
    to pick one to run the time test on"""
    if string_to_match is None:
        string_to_match = input("Enter string to use to find desired graph: ") 
    out = subprocess.check_output(f"find /home/jrhmc1/Desktop/EquitablePartitions/ -type f -name '{string_to_match}'",shell=True,text=True)
    print("Files found were: \n\n")
    # format the output so each choice is an element in a list
    out = out.split('\n')[:-1]
    for i,opt in enumerate(out):
        print(f"{i}: {opt}")

    choice = input("Which file do you want to use?\n(based on numberical label): ")
    again = True
    while again:
        try:
            path = out[choice]
            return path
        except Exception as e:
            print(f"Failed with exception:\n{e}\nTry again.")
            for i,opt in enumerate(out):
                print(f"{i}: {opt}")

            choice = input("Which file do you want to use?\n(based on numberical label): ")

def CheckSlurmParameters(skip_check=False):
    """Gets the slurm parameters saved in slurm_params.txt
    ARGS:
    ----------------
    skip_check (bool): if True shows params, asks if any need to be changed and changes them according
        to input. If False then just returns the parameters as they are
    RETURNS:
    -----------------
    param_dict (dict): dictionary with indices as keys mapping to a list with param names and values
    """
    with open('./../../Slurm/slurm_params.txt','r') as file:
        params = file.read()
    param_dict = {}
    # print the params for the user to see.
    for i,par in enumerate(params.strip().split('\n')):
        label,val = par.split('=')
        param_dict[i] = [label,val]
    
    # just return the dictionary if you don't want to change anything
    if skip_check:
        return param_dict
    
    # change slurm parameters if needed
    while True:
        print("The current slurm parameters are:")
        print(param_dict)
        choice = h.parse_input("\nDo any of these need to be changed? (Y/N): ")   
        if choice == 'Y':
            while True:
                choice = input("Type index of change: ")
                try: 
                    choice = int(choice)
                    new_val = input(f"Change {param_dict[choice][0]} from {param_dict[choice][1]} to what: ")
                    param_dict[choice][1] = new_val
                    break
                except Exception as e:
                    print(f"encountered {e}. Try again!")
                    continue
        else:
            # save the changes
            with open("/home/jrhmc1/Desktop/EquitablePartitions/Slurm/slurm_params.txt","w") as f:
                for label,val in param_dict.values():
                    f.write(f"{label}={val}\n")
            # and return the parameter dictionary
            return param_dict