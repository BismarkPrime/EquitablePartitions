import sys
sys.path.append("/home/jrhmc1/Desktop/EquitablePartitions/src/")
import pandas as pd
import numpy as np
import slurm_helper as h
import subprocess, os
from matplotlib import pyplot as plt
import argparse

RESULTS_DIR = "/home/jrhmc1/Desktop/EquitablePartitions/Results/"

    #NOTE: took out plot=True here because I don't think I was using it. but if errors hopefully I'll see this.
def MakeSpeedGraph(plot_name="Sl_v_Se_times.png",match_points=False,fit_lines=False,split_parallel=False):
    """Makes the speed graph from the data in the Master Keeper csv file
    PARAMETERS:
        plot_name (str): name of file to save graph to
        match_points (bool): If true will create a graph only from the points that have both a serial
            and a parallel time recorded
        fit_lines (bool): If True will plot fit lines to the data with the slope of the liens recorded
            for analyzing complexity
        split_parallel (bool): If True will split the parallel line into two lines. One for taking into 
            account the reloading of the graph and one to not take that into account.
    """
    df = pd.read_csv("Master_keeper.csv",index_col="SlurmID")
    # make the graph size column
    # alternatively this could be done with:
    # df['Graph_size'] = pd.to_numeric(df['Graph_name'].str.extract(r'(\d+)')[0])
    df['Graph_size'] = pd.to_numeric(df['Graph_name'].str.split('bertha',expand=True)[1],errors='coerce') # you need expand=True here so it returns the split result as another dataframe
    df = df.sort_values(by='Graph_size')
    if match_points:
        # only take points that have a serial and parallel value.
        df = df[df.Graph_size.isin(df[df.Run_type=='parallel'].Graph_size)]
    # plot serial and parallel on same graph
    for name, group in df.groupby('Run_type'):
        if name == 'serial':
            x,y = group['Graph_size'],group['Total_wGraph']
            plt.loglog(x,y,'-o',markersize=5,label="QR method",base=2,color='steelblue')
            if fit_lines: # get fitting lines to tell the order of the method.
                poly,order = FindFit(x.iloc[-7:],y.iloc[-7:])
                plt.loglog(x.iloc[-7:],np.exp(poly(np.log(x.iloc[-7:]))),'--',
                label=r"O($n^{{" + f"{np.round(order,3)}" + r"}}$)",base=2,color='red',alpha=.6)
        elif name == 'parallel': # will run to graph second type if parallel
            x,y = group['Graph_size'],group['Total_wGraph']
            plt.loglog(x,y,'-o',markersize=5,label="LEParD Parallel",base=2,color="goldenrod")
            if fit_lines:
                poly,order = FindFit(x.iloc[-9:],y.iloc[-9:])
                plt.loglog(x.iloc[-9:],np.exp(poly(np.log(x.iloc[-9:]))),'--',
                label=r"O($n^{{" + f"{np.round(order,3)}" + r"}}$)",base=2,color='mediumseagreen',alpha=.6)
            if split_parallel:
                plt.loglog(group['Graph_size'],group['Total_noGraph'],'--o',markersize=5,label="LEParD Parallel-no graph",base=2)

    plt.xlabel("Layered Graph Size")
    plt.ylabel("Run Time (seconds)")
    plt.title("Parallelized LEParD vs. QR Method")
    plt.legend()
    plt.savefig(plot_name)

def FindFit(x,y):
    """find the best exponential fit of the data given as x and y
    """
    # transform y = a*x**b into log(y) = log(a) + b*log(x)
    logx = np.log(x); logy = np.log(y)
    # fit this with a 1 degree polynomial
    coeff = np.polyfit(logx,logy,deg=1)
    order = coeff[0]
    # returning the best fit polynomial and the order for the legend.
    return np.poly1d(coeff),order

def FindRecord(size):
    """Help function for CheckRecord
    """
    # read in dataframe
    df = pd.read_csv("Master_keeper.csv",index_col="SlurmID")
    # get the sizes from the graph names
    sizes = df.Graph_name.str.extract(r'(\d+)')
    # make a mask for the parts of the dataframe matching what you are wanting to check
    mask = sizes[0].str.contains(fr'(^{size}$)')
    # see if any matches existed.
    if df[mask].any().any():
        print(f"Found these matches:\n{df[mask][['Graph_name','Graph_format','Run_type','Total_wGraph']]}")
    else:
        print("Found no matches. run probably not recorded yet.")

def CheckRecord(to_check):
    """Checks if the graph size in to_check has already been recorded and prints the results
    if it has or tells you it doesn't have them if it doesn't.
    PARAMETERS:
        to_check (string or list): the sizes to check as strings
    RETURNS:
        just what it prints
    """
    # check each individually if gave a list
    if type(to_check) is list or type(to_check) is np.array:
        for size in to_check:
            FindRecord(size)
    # if only one was given check that one.
    else:
        if type(to_check) is not str:
            print("You must give sizes to check as strings")
            return
        
        else:
            FindRecord(to_check)

def DisplayRecord():
    # read in dataframe
    df = pd.read_csv("Master_keeper.csv",index_col="SlurmID")
    print(df)
    return df

if __name__ == "__main__":
    # get needed command line arguements
    parser = argparse.ArgumentParser(description="This script has the capability to manipulate various aspect "
                                                "of data that has yet to be stored or has already been stored "
                                                "including graphing data from Master_keeper.csv, or reseting "
                                                "the labels of specific runs")
    parser.add_argument("--match_points",action='store_true',help="Will determine whether or not all recorded runs"
                                                                "are plotted or only those sizes that have both slurm"
                                                                "and serial runs.")
    parser.add_argument("--clean",action='store_true',help="Eliminate nan and inf times in master keeper")
    parser.add_argument("--reinit",action='store_true',help="erase all data from master keeper")
    parser.add_argument("--reset_recorded_runs",action='store_true',help="make all 'recorded' runs appear as if "
                                                                        "they haven't been checked yet")
    parser.add_argument("--reset_problem_runs",action='store_true',help="make all 'problem' runs appear as if "
                                                                        "they haven't been checked yet")
    parser.add_argument("--reset_all_runs",action='store_true',help="make all runs appear as if "
                                                                        "they haven't been checked yet")
    parser.add_argument("--serialVslurm",action='store_true',help="make a graph of the recorded runs. Inlcluding the "
                                                                "--match_points argument will only graph point that have "
                                                                "both a slurm and serial run recorded.")
    parser.add_argument("--plot_name",dest="plot_name",default="Sl_v_Se_times.png",help="assings the name of the resulting plot for serialVslurm.")
    parser.add_argument("--check_record",dest="to_check",default=None,help="Checks if the inputted run sizes have been recorded yet and prints them if they have.")
    parser.add_argument("--fit_lines",action='store_true',help="will include fit lines for complexity reference in plot")
    parser.add_argument("--split_parallel",action='store_true',help="will plot both parallel lines, one including second graph loadin and one without it.")
    parser.add_argument("--display_runs",action='store_true',help="displays recorded runs as a dataframe.")

    args = parser.parse_args()

    if args.clean: # anything with infinity meant the run didn't work so drop it.
        df = pd.read_csv("Master_keeper.csv",index_col="SlurmID")
        df.drop(df[df['Total_wGraph']==np.inf].index,inplace=True)
        df.to_csv("Master_keeper.csv",index_label="SlurmID")
    elif args.reinit: # restart the run timing as a whole.
        choice = h.parse_input("This will erase all saved data, are you sure you want to proceed? (Y/N)")
        if choice == 'Y':
            df = pd.DataFrame(columns=['Graph_name','Graph_size','Graph_format','Run_type',"Nontriv_ep_perc","Nontriv_lep_perc",'Partitioned_times','Total_noGraph','Total_wGraph'])
            df.to_csv(RESULTS_DIR + "Master_keeper.csv",index_label="SlurmID")
        else:
            print("ABORTING...")
    elif args.reset_recorded_runs: # if there was a problem 
        for direct_name in os.listdir(RESULTS_DIR):
            if 'recorded' in direct_name:
                full_path = RESULTS_DIR + direct_name
                subprocess.run(f"mv {full_path} {full_path.split('_recorded')[0]}",shell=True)
    elif args.reset_problem_runs: # make the problem runs be checked again
        for direct_name in os.listdir(RESULTS_DIR):
            if 'prob' in direct_name:
                full_path = RESULTS_DIR + direct_name
                subprocess.run(f"mv {full_path} {full_path.split('_prob')[0]}",shell=True)
    elif args.reset_all_runs: # reset all the runs to be rechecked
        for direct_nme in os.listdir(RESULTS_DIR):
            full_path = RESULTS_DIR + direct_name
            if 'prob' in direct_name:
                subprocess.run(f"mv {full_path} {full_path.split('_prob')[0]}",shell=True)
            elif 'recorded' in direct_name:
                subprocess.run(f"mv {full_path} {full_path.split('_recorded')[0]}",shell=True)
    elif args.serialVslurm: # make the graph based on the data in the master keeper csv
        MakeSpeedGraph(match_points=args.match_points,plot_name=args.plot_name,
                        fit_lines=args.fit_lines,split_parallel=args.split_parallel)
    elif args.to_check:
        CheckRecord(args.to_check)
    elif args.display_runs:
        DisplayRecord()
        
    else:
        print("The keyword provided was not one of the options, please provide one of the options below:\n\n"
            "\tclean:\t\t\teliminate any np.infs in the timing csv\n"
            "\treinit:\t\t\twipe all recorded data in the master keeper csv\n"
            "\treset_recorded_runs:\tany runs labeled as recorded will be reset so the 'record' keyword in Aristotle will check them again\n"
            "\treset_problem_runs:\tsame as reset_recorded_runs but for runs labled with prob instead\n"
            "\treset_all_runs:\t\tDoes what the previous two do.\n"
            "\tserialVslurm:\t\tRead the Master_keeper.csv and create graphs based on the data there.\n")


            