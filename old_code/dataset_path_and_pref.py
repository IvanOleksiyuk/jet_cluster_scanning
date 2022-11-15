import numpy as np
import pickle
import pandas as pd
from scipy.ndimage import gaussian_filter
from io import StringIO
from dark_machine_data import prepare_data_dark_machines
def dataset_path_and_pref(DATASET, REVERSE):
    
    f = open("image_data_sets_path.txt", "r")
    image_data_sets_path=f.read()
    image_data_sets_path=image_data_sets_path[:-1]
    f.close()
    DM_data_sets_path="C:/bachelor work/Spyder/data_set/training_files/"
    
    tra_data_field=[None, None]
    con_data_field=[None, None]
    bg_val_data_field=[None, None]
    sg_val_data_field=[None, None]
    
    if DATASET==1:
        pref="QCD"
        pref2="top"
        tra_data_path=image_data_sets_path+"Xtra-100KQCD-pre3-2.pickle"
        con_data_path=image_data_sets_path+"Xtra-100Ktop-pre3-2.pickle"
        bg_val_data_path=image_data_sets_path+"Xval-40KQCD-pre3-2.pickle"
        sg_val_data_path=image_data_sets_path+"Xval-40Ktop-pre3-2.pickle"
    if DATASET==2:
        pref2="DMl"
        pref="QCDl"
        tra_data_path=image_data_sets_path+"Xtra-100KQCDl.pickle"
        con_data_path=image_data_sets_path+"Xtra-100KDMl.pickle"
        bg_val_data_path=image_data_sets_path+"Xval-40KQCDl.pickle"
        sg_val_data_path=image_data_sets_path+"Xval-40KDMl.pickle"
    if DATASET==3:
        pref="bg"
        pref2="Q200M100"
        tra_data_path=image_data_sets_path+"Xtra_HB_bg0k-100k-.pickle"
        con_data_path=image_data_sets_path+"Xtra_HB_sig-Q200M1000k-100k-.pickle"
        bg_val_data_path=image_data_sets_path+"Xval_HB_bg100k-120k-.pickle"
        sg_val_data_path=image_data_sets_path+"Xval_HB_sig-Q200M100100k-120k-.pickle"
    if DATASET==4:
        pref="bg"
        pref2="Q50M10"
        tra_data_path=image_data_sets_path+"Xtra_HB_bg0k-100k-.pickle"
        con_data_path=image_data_sets_path+"Xtra_HB_sig-Q50M100k-100k-.pickle"
        bg_val_data_path=image_data_sets_path+"Xval_HB_bg100k-120k-.pickle"
        sg_val_data_path=image_data_sets_path+"Xval_HB_sig-Q50M10100k-120k-.pickle"
    if DATASET==14:
        pref="QCDl"
        pref2="lptQ50M10"
        tra_data_path=image_data_sets_path+"Xtra-100KQCDl.pickle"
        con_data_path=image_data_sets_path+"Xtra_jets_q50m10lpt0k-0k-.pickle"
        bg_val_data_path=image_data_sets_path+"Xval-40KQCDl.pickle"
        sg_val_data_path=image_data_sets_path+"Xtes_jets_q50m10lpt0k-20k-.pickle"
        
    if DATASET=="background_test":
        pref="QCDl"
        pref2="bkglptcms"
        tra_data_path=image_data_sets_path+"Xtra-100KQCDl.pickle"
        con_data_path=image_data_sets_path+"Xtra_bkglptcms0k-0k-.pickle"
        bg_val_data_path=image_data_sets_path+"Xval-40KQCDl.pickle"
        sg_val_data_path=image_data_sets_path+"Xval_bkglptcms0k-3k-.pickle"
        
    if DATASET==5.0:
        pref="2d10s"
        pref2="2d1s6m"
        tra_data_path=image_data_sets_path+"Xtra_2d10s0k-100k-.pickle"
        con_data_path=image_data_sets_path+"Xtra_2d1s6m0k-100k-.pickle"
        bg_val_data_path=image_data_sets_path+"Xval_2d10s100k-120k-.pickle"
        sg_val_data_path=image_data_sets_path+"Xval_2d1s6m100k-120k-.pickle"
    if DATASET==5.1:
        pref="1d1Us"
        pref2="1d2Us"
        tra_data_path=image_data_sets_path+"Xtra_1d1Us0k-100k-.pickle"
        con_data_path=image_data_sets_path+"Xtra_1d2Us0k-100k-.pickle"
        bg_val_data_path=image_data_sets_path+"Xval_1d1Us100k-120k-.pickle"
        sg_val_data_path=image_data_sets_path+"Xval_1d2Us100k-120k-.pickle"
    if DATASET==5.2:
        pref="2d1Us"
        pref2="2d2Us"
        tra_data_path=image_data_sets_path+"Xtra_2d1Us0k-100k-.pickle"
        con_data_path=image_data_sets_path+"Xtra_2d2Us0k-100k-.pickle"
        bg_val_data_path=image_data_sets_path+"Xval_2d1Us100k-120k-.pickle"
        sg_val_data_path=image_data_sets_path+"Xval_2d2Us100k-120k-.pickle"
    if DATASET==5.3:
        pref="5d1Us"
        pref2="5d2Us"
        tra_data_path=image_data_sets_path+"Xtra_5d1Us0k-100k-.pickle"
        con_data_path=image_data_sets_path+"Xtra_5d2Us0k-100k-.pickle"
        bg_val_data_path=image_data_sets_path+"Xval_5d1Us100k-120k-.pickle"
        sg_val_data_path=image_data_sets_path+"Xval_5d2Us100k-120k-.pickle"
    if DATASET==5.4:
        pref="tor10u2s"
        pref2="2d7s"
        tra_data_path=image_data_sets_path+"Xtra_tor10u2s0k-100k-.pickle"
        con_data_path=image_data_sets_path+"Xtra_2d7s0k-100k-.pickle"
        bg_val_data_path=image_data_sets_path+"Xval_tor10u2s100k-120k-.pickle"
        sg_val_data_path=image_data_sets_path+"Xval_2d7s100k-120k-.pickle"
    if DATASET==5.5:
        pref="2d10s"
        pref2="2d1s4m"
        tra_data_path=image_data_sets_path+"Xtra_2d10s0k-100k-.pickle"
        con_data_path=image_data_sets_path+"Xtra_2d1s4m0k-100k-.pickle"
        bg_val_data_path=image_data_sets_path+"Xval_2d10s100k-120k-.pickle"
        sg_val_data_path=image_data_sets_path+"Xval_2d1s4m100k-120k-.pickle"
    if DATASET==6:
        pref="2dslope"
        pref2="2dus"
        tra_data_path=image_data_sets_path+"Xtra_2dslope0k-100k-.pickle"
        con_data_path=image_data_sets_path+"Xtra_2dus0k-100k-.pickle"
        bg_val_data_path=image_data_sets_path+"Xval_2dslope100k-120k-.pickle"
        sg_val_data_path=image_data_sets_path+"Xval_2dus100k-120k-.pickle"
    if DATASET==6.1:
        pref="5dslope"
        pref2="5dus"
        tra_data_path=image_data_sets_path+"Xtra_5dslope0k-100k-.pickle"
        con_data_path=image_data_sets_path+"Xtra_5dus0k-100k-.pickle"
        bg_val_data_path=image_data_sets_path+"Xval_5dslope100k-120k-.pickle"
        sg_val_data_path=image_data_sets_path+"Xval_5dus100k-120k-.pickle"
    if DATASET==7:
        pref="2dexp1"
        pref2="2du2-1.5"
        tra_data_path=image_data_sets_path+"Xtra_2dexp10k-100k-.pickle"
        con_data_path=image_data_sets_path+"Xtra_2du2-1.50k-100k-.pickle"
        bg_val_data_path=image_data_sets_path+"Xval_2dexp1100k-120k-.pickle"
        sg_val_data_path=image_data_sets_path+"Xval_2du2-1.5100k-120k-.pickle"
    if DATASET==8:
        pref="2d1s"
        pref2="tor4u1s" 
        tra_data_path=image_data_sets_path+"Xtra_2d1s0k-100k-.pickle"
        con_data_path=image_data_sets_path+"Xtra_tor4u1s0k-100k-.pickle"
        bg_val_data_path=image_data_sets_path+"Xval_2d1s100k-120k-.pickle"
        sg_val_data_path=image_data_sets_path+"Xval_tor4u1s100k-120k-.pickle"
    if DATASET==8.1:
        pref="5d1s"
        pref2="tor5d4u1s" 
        tra_data_path=image_data_sets_path+"Xtra_5d1s0k-100k-.pickle"
        con_data_path=image_data_sets_path+"Xtra_tor5d4u1s0k-100k-.pickle"
        bg_val_data_path=image_data_sets_path+"Xval_5d1s100k-120k-.pickle"
        sg_val_data_path=image_data_sets_path+"Xval_tor5d4u1s100k-120k-.pickle"
    
    if DATASET=="1h5":
        pref="QCDh5"
        pref2="toph5"
        tra_data_path=image_data_sets_path+"h5/top_samples/top-img-bkg.h5"
        con_data_path=image_data_sets_path+"h5/top_samples/top-img-sig.h5"
        bg_val_data_path=image_data_sets_path+"h5/top_samples/top-img-bkg.h5"
        sg_val_data_path=image_data_sets_path+"h5/top_samples/top-img-sig.h5"
        tra_data_field=[None, 100000]
        con_data_field=[None, 100000]
        bg_val_data_field=[100000, 120000]
        sg_val_data_field=[100000, 120000]
        
    if DATASET=="2h5":
        pref="QCDlh5"
        pref2="DMlh5"
        tra_data_path=image_data_sets_path+"h5/aachen/aachen-img-bkg.h5"
        con_data_path=image_data_sets_path+"h5/aachen/aachen-img-sig.h5"
        bg_val_data_path=image_data_sets_path+"h5/aachen/aachen-img-bkg-test.h5"
        sg_val_data_path=image_data_sets_path+"h5/aachen/aachen-img-sig.h5"
        tra_data_field=[None, 100000]
        con_data_field=[0, 0]
        bg_val_data_field=[None, 20000]
        sg_val_data_field=[None, 20000]
    if DATASET=="3h5":
        pref="3h5"
        pref2="Q200M100h5"
        tra_data_path=image_data_sets_path+"h5/aachen/aachen-img-bkg.h5"
        con_data_path=image_data_sets_path+""
        bg_val_data_path=image_data_sets_path+"h5/aachen/Xval_HB_bg100k-120k-.pickle"
        sg_val_data_path=image_data_sets_path+""
        tra_data_field=[None, 100000]
        con_data_field=[0, 0]
        bg_val_data_field=[None, 20000]
        sg_val_data_field=[None, 20000]
    if DATASET=="4h5":
        pref="QCDlh5"
        pref2="Q50M10cmslpth5"
        tra_data_path=image_data_sets_path+"h5/aachen/aachen-img-bkg.h5"
        con_data_path=image_data_sets_path+"h5/new/jets-q50m10-lpt-cms-img.h5"
        bg_val_data_path=image_data_sets_path+"h5/aachen/aachen-img-bkg-test.h5"
        sg_val_data_path=image_data_sets_path+"h5/new/jets-q50m10-lpt-cms-img.h5"
        tra_data_field=[None, 100000]
        con_data_field=[0, 0]
        bg_val_data_field=[None, 20000]
        sg_val_data_field=[None, 20000]
        
    if DATASET=="2f":
        pref="QCDf"
        pref2="DMlf"
        tra_data_path=image_data_sets_path+"final/qcd_img.h5"
        con_data_path=image_data_sets_path+"final/aachen_img.h5"
        bg_val_data_path=image_data_sets_path+"final/qcd_img.h5"
        sg_val_data_path=image_data_sets_path+"final/aachen_img.h5"
        tra_data_field=[None, 100000]
        con_data_field=[0, 0]
        bg_val_data_field=[100000, 120000]
        sg_val_data_field=[None, 20000]
    if DATASET=="4f":
        pref="QCDf"
        pref2="Q50M10lf"
        tra_data_path=image_data_sets_path+"final/qcd_img.h5"
        con_data_path=image_data_sets_path+"final/heidelberg_img.h5"
        bg_val_data_path=image_data_sets_path+"final/qcd_img.h5"
        sg_val_data_path=image_data_sets_path+"final/heidelberg_img.h5"
        tra_data_field=[None, 100000]
        con_data_field=[0, 0]
        bg_val_data_field=[100000, 120000]
        sg_val_data_field=[None, 20000]
    
    if type(DATASET)==str:
        if DATASET[0:2]=="DM":
            sig_num=(int)(DATASET[-1])
            if DATASET[2:5]=="_1_":
                anomaly_list=["glgl1400_neutralino1100_chan1.csv",
                              "glgl1600_neutralino800_chan1.csv",
                              "monojet_Zp2000.0_DM_50.0_chan1.csv",
                              "monotop_200_A_chan1.csv",
                              "sqsq_sq1800_neut800_chan1.csv",
                              "sqsq1_sq1400_neut800_chan1.csv",
                              "stlp_st1000_chan1.csv",
                              "stop2b1000_neutralino300_chan1.csv"]
                pref="BGchan1"
                pref2="SG"+DATASET[-1]+"chan1"
                tra_data_path=DM_data_sets_path+"chan1/background_chan1_7.79.csv"
                bg_val_data_path=DM_data_sets_path+"chan1/background_chan1_7.79.csv"
                con_data_path=DM_data_sets_path+"chan1/"+anomaly_list[sig_num]
                sg_val_data_path=DM_data_sets_path+"chan1/"+anomaly_list[sig_num]
                tra_data_field=[None, 100000]
                con_data_field=[0, 0]
                bg_val_data_field=[100000, 120000]
                sg_val_data_field=[None, 20000]
                
                
            elif DATASET[2:5]=="_2a":
                anomaly_list=["chaneut_cha200_neut50_chan2a.csv",
                              "chaneut_cha250_neut150_chan2a.csv",
                              "chaneut_cha300_neut100_chan2a.csv",
                              "chaneut_cha400_neut200_chan2a.csv",
                              "gluino_1000.0_neutralino_1.0_chan2a.csv",
                              "pp23mt_50_chan2a.csv",
                              "pp24mt_50_chan2a.csv"]
                pref="BGchan2a"
                pref2="SG"+DATASET[-1]+"chan2a"
                tra_data_path=DM_data_sets_path+"chan2a/background_chan2a_309.6.csv"
                bg_val_data_path=DM_data_sets_path+"chan2a/background_chan2a_309.6.csv"
                con_data_path=DM_data_sets_path+"chan2a/"+anomaly_list[sig_num]
                sg_val_data_path=DM_data_sets_path+"chan2a/"+anomaly_list[sig_num]
                tra_data_field=[None, 18000]
                con_data_field=[0, 0]
                bg_val_data_field=[18000, None]
                sg_val_data_field=[None, 20000]
                
            elif DATASET[2:5]=="_2b":
                anomaly_list=["chacha_cha300_neut140_chan2b.csv",
                              "chacha_cha400_neut60_chan2b.csv",
                              "chacha_cha600_neut200_chan2b.csv",
                              "chaneut_cha200_neut50_chan2b.csv",
                              "chaneut_cha250_neut150_chan2b.csv",
                              "gluino_1000.0_neutralino_1.0_chan2b.csv",
                              "pp23mt_50_chan2b.csv",
                              "pp24mt_50_chan2b.csv",
                              "stlp_st1000_chan2b.csv"]
                pref="BGchan2b"
                pref2="SG"+DATASET[-1]+"chan2b"
                tra_data_path=DM_data_sets_path+"chan2b/background_chan2b_7.8.csv"
                bg_val_data_path=DM_data_sets_path+"chan2b/background_chan2b_7.8.csv"
                con_data_path=DM_data_sets_path+"chan2b/"+anomaly_list[sig_num]
                sg_val_data_path=DM_data_sets_path+"chan2b/"+anomaly_list[sig_num]
                tra_data_field=[None, 100000]
                con_data_field=[0, 0]
                bg_val_data_field=[100000, 120000]
                sg_val_data_field=[None, 20000]
                
            elif DATASET[2:5]=="_3_":
                anomaly_list=["glgl1400_neutralino1100_chan3.csv",
                              "glgl1600_neutralino800_chan3.csv",
                              "gluino_1000.0_neutralino_1.0_chan3.csv",
                              "monojet_Zp2000.0_DM_50.0_chan3.csv",
                              "monotop_200_A_chan3.csv",
                              "monoV_Zp2000.0_DM_1.0_chan3.csv",
                              "sqsq_sq1800_neut800_chan3.csv",
                              "sqsq1_sq1400_neut800_chan3.csv",
                              "stlp_st1000_chan3.csv",
                              "stop2b1000_neutralino300_chan3.csv"]
                pref="BGchan3"
                pref2="SG"+DATASET[-1]+"chan3"
                tra_data_path=DM_data_sets_path+"chan3/background_chan3_8.02.csv"
                bg_val_data_path=DM_data_sets_path+"chan3/background_chan3_8.02.csv"
                con_data_path=DM_data_sets_path+"chan3/"+anomaly_list[sig_num]
                sg_val_data_path=DM_data_sets_path+"chan3/"+anomaly_list[sig_num]
                tra_data_field=[None, 100000]
                con_data_field=[0, 0]
                bg_val_data_field=[100000, 120000]
                sg_val_data_field=[None, 20000]
    
    if DATASET=="LHCORnD100K_2K":
        pref="ORnDbg"
        pref2="ORnDsg"
        tra_data_path="C://datasets/100K_BG.pickle"
        con_data_path="C://datasets/2K_SG.pickle"
        bg_val_data_path="C://datasets/100K_BG.pickle"
        sg_val_data_path="C://datasets/2K_SG.pickle"

    if DATASET=="LHCORnDp100K_2K":
        pref="ORnDbgp"
        pref2="ORnDsgp"
        tra_data_path="C://datasets/100K_BGp.pickle"
        con_data_path="C://datasets/2K_SGp.pickle"
        bg_val_data_path="C://datasets/100K_BGp.pickle"
        sg_val_data_path="C://datasets/2K_SGp.pickle"
        
    if DATASET=="LHCORnD100K_2K_1l":
        pref="ORnDbg1l"
        pref2="ORnDsg1l"
        tra_data_path="C://datasets/100K_BG1l.pickle"
        con_data_path="C://datasets/2K_SG1l.pickle"
        bg_val_data_path="C://datasets/100K_BG1l.pickle"
        sg_val_data_path="C://datasets/2K_SG1l.pickle"
        
    if DATASET=="LHCORnD100K_2K_2l":
        pref="ORnDbg2l"
        pref2="ORnDsg2l"
        tra_data_path="C://datasets/100K_BG2l.pickle"
        con_data_path="C://datasets/2K_SG2l.pickle"
        bg_val_data_path="C://datasets/100K_BG2l.pickle"
        sg_val_data_path="C://datasets/2K_SG2l.pickle"
    
    if DATASET=="moon_demo":
        pref="moon"
        pref2="moonl"
        tra_data_path=image_data_sets_path+"moon_demo.pickle"
        con_data_path=image_data_sets_path+"moon_demo.pickle"
        bg_val_data_path=image_data_sets_path+"moon_demo.pickle"
        sg_val_data_path=image_data_sets_path+"moon_demo.pickle"
    
    if REVERSE:
        tra_data_path, con_data_path = con_data_path, tra_data_path
        bg_val_data_path, sg_val_data_path = sg_val_data_path, bg_val_data_path
        pref, pref2 = pref2, pref
        
    DI={} #dataset info dictionary
    DI["pref"]=pref
    DI["pref2"]=pref2 
    DI["tra_data_path"]=tra_data_path
    DI["con_data_path"]=con_data_path
    DI["bg_val_data_path"]=bg_val_data_path
    DI["sg_val_data_path"]=sg_val_data_path
    DI["tra_data_field"]=tra_data_field 
    DI["con_data_field"]=con_data_field
    DI["bg_val_data_field"]= bg_val_data_field
    DI["sg_val_data_field"]=sg_val_data_field
    return DI

def prepare_data(path, crop=None, field=None, preproc=None, SIGMA=0, simple_load=False, reshape=True, standard=None):     
    #read from pickle
    if path[-6:]=="pickle":
        X = pickle.load(open(path, "rb"))
        X = X[field[0]:field[1]]
    #read from "h5"
    if path[-2:]=="h5":
        store = pd.HDFStore(path, 'r')
        X=store.select("table")
        X=X.values
        X=X[field[0]:field[1], :1600]
        X=X.reshape(-1, 40, 40, 1)
        store.close()
    if path[-3:]=="csv":
        X, standard = prepare_data_dark_machines(path, field=field, standard=standard)
    X = X[:crop]
    if crop!=None:
        if len(X)<crop:
            print("dataset is too short!")
    if preproc is not None:
        print("preprocessing active")
        X = preproc(X)
    if SIGMA>0:
        if len(X.shape)==4:
            X = gaussian_filter(X, sigma=[0, SIGMA, SIGMA, 0]) 
        else:
            X = gaussian_filter(X, sigma=[0, SIGMA, SIGMA]) 
    if len(X.shape)>2:
        X = X.reshape((X.shape[0], 1600))
    print("Succesfully prepared", path, "with", len(X), "events")
    return X, standard




