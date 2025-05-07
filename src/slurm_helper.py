import os

def PrepSlurmScript(slurm_paste,start_script):
    """puts the slurm details into the beginning of the slurm script that's about to run
    hopefully this will decrease the command line size
    """
    with open(start_script,"r") as f:
        data = f.read()
    stripped = data.split('import')
    data = ""
    # eliminate the previous slurm commands
    for piece in stripped[1:]:
        data += "import"+piece
    # write in the new slurm params and the rest of the data
    with open(start_script,"w") as f:
        f.write(slurm_paste+'\n\n\n'+data)


def parse_input(prompt):
    """makes sure input is of the form specified in the prompt. Which is signified by being inside
    parentheses and separated by backslashes '/'
    """
    keywords = prompt.split('(')[-1][:-3].split('/')
    inp = input(prompt)
    while True:
        if inp in keywords:
            return inp
        else:
            inp = input(f"Not valid input. You must choose one of -> {keywords}: ")

def start_section(content: str) -> None:
    border = '#' * os.get_terminal_size().columns
    pad_left = (os.get_terminal_size().columns - len(content)) // 2
    print(f"{border}\n\n{' ' * pad_left}{content.upper()}\n\n{border}")