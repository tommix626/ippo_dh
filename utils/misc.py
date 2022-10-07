import os


def hard_update(target, source):
    """
    Copy network parameters from source to target
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def check_seed(r:list, r1:list, r2:list):
    # return 1
    # get the last index
    if len(r) <= 0 or len(r1) <= 0 or len(r2) <= 0:
        return 1

    r_avg = sum(r[-10:]) / len(r[-10:])
    r1_avg = sum(r1[-10:]) / len(r1[-10:])
    r2_avg = sum(r2[-10:]) / len(r2[-10:])
    if r_avg > 0.8:
        return 1
    print(f"r={r_avg},r1={r1_avg},r2={r2_avg}, rejected.")
    return 0

def comb_path(rootpath,folder):
    """
    Utils Function,combine and create path
    """
    newpath = os.path.join(rootpath, folder)
    if not os.path.exists(newpath):
        os.makedirs(newpath)
        print("Creating new path:",newpath)
    return newpath