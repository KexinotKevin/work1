import argparse as ap

modname = 'FC'
atlas = 'bna246'
dtname = 'HCD'
lbtype = 'cogcomp'
datasetDir = '/media/shulab/WD_10T/datasets/'
dataDir = datasetDir + dtname
projDir = os.getcwd()

# HCD : 
# 'src_subject_id', 'sex',
# HCD_net_cogcomp01.csv

# S1200 : 
# 'Subject', 'Gender',
# S1200_889_CogScores.csv

method = "robreg"
# "kridge", "ridge", "robreg"

if method=="kridge":
    model=KernelRidge(kernel="rbf")
elif method=="ridge":
    model=Ridge()
elif method=="robreg":
    model=HuberRegressor()


# if HCD
labellist = ["nih_fluidcogcomp_unadjusted", "nih_fluidcogcomp_ageadjusted", 
			"nih_crycogcomp_unadjusted", "nih_crycogcomp_ageadjusted", 
			"nih_eccogcomp_unadjusted", "nih_eccogcomp_ageadjusted", 
			"nih_totalcogcomp_unadjusted", "nih_totalcogcomp_ageadjusted"]

# if S1200
# labellist = ["CogFluidComp_Unadj", "CogFluidComp_AgeAdj",
#             "CogCrystalComp_Unadj", "CogCrystalComp_AgeAdj",
#             "CogEarlyComp_Unadj", "CogEarlyComp_AgeAdj",
#             "CogTotalComp_Unadj", "CogTotalComp_AgeAdj"]

############## 1. Configurations ##############
def argparser():
    args = ap.ArgumentParser()
    args.add_argument("--")