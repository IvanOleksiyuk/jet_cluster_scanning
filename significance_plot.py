# this file creates the central plot of the paper
# We see how much does significance change ehwn we chang the signl fraction
# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Imports block
import numpy as np
import scipy.stats as sts
from matplotlib import pyplot as plt
from scipy.special import gammainc
import pickle
from utils.utils import p2Z
from utils.os_utils import list_files
import os
import utils.set_matplotlib_default
import copy

def find_corresponding_x_values(specific_y, y_values, x_values):
    corresponding_x_values = []

    for i in range(len(y_values) - 1):
        if (y_values[i] <= specific_y <= y_values[i + 1]) or (y_values[i] >= specific_y >= y_values[i + 1]):
            x_interpolation = np.interp(specific_y, [y_values[i], y_values[i + 1]], [x_values[i], x_values[i + 1]])
            print([x_values[i], x_values[i + 1]])
            print("interpolating", specific_y, "between", y_values[i], "and", y_values[i + 1], "to", x_interpolation)
            corresponding_x_values.append(x_interpolation)

    return corresponding_x_values

def get_median_and_quar_err(Zs):
    median = np.median(Zs, axis=1)
    q1 = np.quantile(Zs, 0.25, axis=1)
    q3 = np.quantile(Zs, 0.75, axis=1)
    err_low = median - q1
    err_high = q3 - median
    return median, err_low, err_high

def make_first_invalid_valid(valid):
    shape = valid.shape
    valid = np.where(valid)[0]
    valid = np.array([valid[0]-1, ]+list(valid))
    validd = np.full(shape, False)
    validd[valid] = True
    return validd

def make_last_valid_invalid(invalid):
    shape = invalid.shape
    invalid = np.where(invalid)[0]
    invalid = np.array(list(invalid)+[invalid[-1]+1, ])
    invalidd = np.full(shape, False)
    invalidd[invalid] = True
    return invalidd

def find_stat_validity(y, intersect=True):
    valid = np.logical_not(y==y[0])
    invalid = np.logical_not(valid)
    if intersect:
        valid = make_first_invalid_valid(valid)
    return valid, invalid



def draw_line_yerr(x, y, yerr=None, color="black", label=None, linestyle="solid", capsize=5, style="band+lowstat", validd="auto"):
    if style == "errorbar":
        plt.errorbar(
            x,
            y,
            yerr=yerr,
            label=label,
            color=color,
            capsize=capsize,
            linestyle=linestyle,
        )
    elif style == "band":
        if isinstance(yerr, list):
            plt.plot(x, y, label=label, color=color, linestyle=linestyle)
            plt.fill_between(x, y-yerr[0], y+yerr[1], alpha=0.2, color=color)
        else:
            plt.plot(x, y, label=label, color=color, linestyle=linestyle)
            plt.fill_between(x, y-yerr, y+yerr, alpha=0.2, color=color)
    
    elif style == "band+lowstat":
        
        if validd=="all":
            valid = np.full(y.shape, True)
        elif validd=="auto":
            valid, invalid = find_stat_validity(y)
        else:
            valid = validd
            invalid = np.logical_not(validd)
            invalid = make_last_valid_invalid(invalid)
        plt.plot(x[valid], y[valid], label=label, color=color, linestyle=linestyle)
        if not np.all(valid):
            plt.plot(x[invalid], y[invalid], color=color, linestyle="dotted")
        if isinstance(yerr, list):
            plt.fill_between(x[valid], y[valid]-yerr[0][valid], y[valid]+yerr[1][valid], alpha=0.2, color=color)
            


def significance_plot(plot_idealised = True, 
                      plot_realistic = True, 
                      plot_BH = False,
                      use_points_insuff_stats = False,
                      style="band+lowstat",
                      use_cs_name=False):
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load the data block + define some parameters

    # load the mass spectra
    data_path = "../../scratch/DATA/LHCO/"
    mjj_bg = np.load(data_path + "mjj_bkg_sort.npy")
    mjj_sg = np.load(data_path + "mjj_sig_sort.npy")
    print(np.std(mjj_sg))
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Py Bump Hunter block
    # perform bump hunting using the pyBumpHunter package
    if plot_BH:
        import pyBumpHunter as BH

        def get_hunter_scheme(hunter) -> np.ndarray:
            if hunter.str_scale == "lin":
                sig_str = np.arange(
                    hunter.str_min,
                    hunter.str_min + hunter.str_step * len(hunter.sigma_ar),
                    step=hunter.str_step,
                )
            else:
                sig_str = np.array(
                    [
                        i % # The above code is a Python comment. Comments in Python start with a hash
                        # symbol (#) and are used to provide explanations or notes within the
                        # code. In this case, the comment simply contains the number 1.
                        10 * 10 ** (hunter.str_min + i // 10)
                        for i in range(len(hunter.sigma_ar) + len(hunter.sigma_ar) // 10 + 1)
                        if i % 10 != 0
                    ]
                )
            return sig_str


        BH_list = []
        BH_set_list = ["CURTAINS2"]  # "BHD"
        BH_color_list = ["black", "indigo"]
        for BH_set_name in BH_set_list:
            BHsettings = {
                "rang": [3000, 4600],
                "width_min": 2,
                "width_max": 10,
                "width_step": 2,
                "scan_step": 1,
                "npe": 100000,
                "nworker": 6,
                "seed": 42,
                "use_sideband": True,
                "str_min": -4,
                "sigma_limit": 10,
                "str_scale": "log",
                "bins": np.linspace(3000, 4600, 16),
            }

            if BH_set_name == "CURTAINS":
                n_bins = 16
                bins = np.linspace(3000, 4600, n_bins + 1)
                BHsettings["bins"] = bins
            if BH_set_name == "CURTAINS2":
                n_bins = 16
                bins = np.linspace(3000, 4600, n_bins + 1)
                BHsettings["bins"] = bins
                BHsettings["width_min"] = 1
            elif BH_set_name == "BHD":
                n_bins = 60
                bins = np.linspace(3000, 4600, n_bins + 1)
                BHsettings["bins"] = bins

            files_list = list_files("BHcache")
            if BH_set_name + ".pickle" in files_list:
                raw_str, raw_sens = pickle.load(open(f"BHcache/{BH_set_name}.pickle", "rb"))
            else:
                hunter = BH.BumpHunter1D(**BHsettings)
                # Run the bump hunter without any cuts
                hunter.signal_inject(mjj_sg, mjj_bg)
                raw_str = get_hunter_scheme(hunter)
                raw_sens = hunter.sigma_ar.copy()
                min_l = min(
                    len(raw_sens), len(raw_str)
                )  # Sometimes the hunter lengths dont match
                raw_sens = raw_sens[:min_l]
                raw_str = raw_str[:min_l]
                pickle.dump((raw_str, raw_sens), open(f"BHcache/{BH_set_name}.pickle", "wb"))
            BH_list.append((raw_str, raw_sens))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Cluster scanning block
    # Load significances from the results of t_statistic_distribution.py
    CS_list = []

    if plot_realistic:
        CS_list += [r"plots/ts_distribs/V4prep05_1_maxdev3_msdersCURTAINS_15mean_results.pickle"]
    if plot_idealised:
        CS_list+= [r"plots/ts_distribs/V4prep05_1_maxdev3_msdersCURTAINS_15mean_ideal_results.pickle"]


    name_list = [os.path.basename(path) for path in CS_list]
    results_list = []
    for i, path in enumerate(CS_list):
        results = pickle.load(open(path, "rb"))
        results_list.append(results)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Ideal fit block

    # plt.plot(sig_frac, sign_ideal, color="red")
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot
    # Create the figure
    fig = plt.figure(figsize=(3.5, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1])
    axs=[]
    axs.append(fig.add_subplot(gs[0]))
    axs.append(fig.add_subplot(gs[1]))
    
    plt.sca(axs[0])

    if plot_BH:
        for i, r in enumerate(BH_list):
            raw_str, raw_sens = r
            plt.errorbar(
                raw_str * len(mjj_sg),
                raw_sens[:, 0],
                xerr=0,
                yerr=[raw_sens[:, 1], raw_sens[:, 2]],
                marker="x",
                color=BH_color_list[i],
                uplims=raw_sens[:, 2] == 0,
                label="pyBumpHunter" + BH_set_list[i],
                #alpha=0.1,          
            )

    for results, name in zip(results_list, name_list):
        # Use median and quartiles for the error bars
        Zs = results["Zs"]
        min_length = min([len(Z) for Z in results["Zs"]])
        Zs = [Z[:min_length] for Z in Zs]
        Zs = np.stack(Zs, axis=1).T
        Zs[np.isinf(Zs)] = p2Z(results["p_upper_bound"][0])
        print(Zs.shape)
        median_CS, err_low_CS, err_high_CS = get_median_and_quar_err(Zs)
        x_CS = np.round(np.array(results["contaminations"]) * len(mjj_bg))
        if use_cs_name == True:
            label = name.replace("_", " ")
        elif use_cs_name:
            label = use_cs_name
        else:
            label = "Cluster Scanning"
        draw_line_yerr(x_CS, median_CS, [err_low_CS, err_high_CS], label=label, style=style, color="blue")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("CS")
        print(find_corresponding_x_values(3, np.flip(median_CS), np.flip(x_CS)))
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    # print(results["contaminations"])
    # print(results["Z_mean_ps"])
    # print(np.mean(results["Zs"], axis=1))
    # plt.plot([0.005, 0.0025], [3.21, 0.92], color="red")

    if plot_realistic:
        res_list = ["gf_results/MLSnormal_positiv4_parambootstrap_true10000x100_binning16_Zs.pickle",
                    "gf_results/MLSnormal_positiv3_parambootstrap_true10000x100_binning16_Zs.pickle",]
        labal_list = ["4 par fit", "3 par fit"]
        color_list = ["maroon", "peru"]
        for resn, lab, color in zip(res_list, labal_list, color_list):
            res = pickle.load(open(resn, "rb"))
            median, err_low, err_high = get_median_and_quar_err(res["Zs"])
            if color=="maroon":
                median_fit = median
            x_fit = np.round(res["sig_fractions"] * len(mjj_sg))
            draw_line_yerr(x_fit, median, [err_low, err_high], label=lab, style=style, color=color, linestyle="dashed")
            
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print(lab)
            print(find_corresponding_x_values(3, np.flip(median), np.flip(x_fit)))
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

    if plot_idealised:
        res = pickle.load(open("gf_results/MLSnormal_positiv3_parambootstrap_true10000x100_binning16_Zs_ideal.pickle", "rb"))
        x_fit = np.round(res["sig_fractions"] * len(mjj_sg))
        median_fit, err_low, err_high = get_median_and_quar_err(res["Zs"])
        draw_line_yerr(x_fit, median_fit, [err_low, err_high], label="Idealised fit", style=style, color="maroon", linestyle="dashed")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print(find_corresponding_x_values(3, np.flip(median_fit), np.flip(x_fit)))
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

    if use_cs_name == True:
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    else:
        plt.legend(loc="lower right")
    plt.gca().xaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1)) 
    plt.grid(which='both')
    plt.xscale("log")
    plt.xlim(5e2, 5e3)
    plt.xlabel("$\epsilon$")
    plt.ylabel("Z $[\sigma]$")

    def plot_significances(Zs, contaminations, label=None, color="blue"):
        median, err_low, err_high = get_median_and_quar_err(Zs)
        draw_line_yerr(contaminations, median, [err_low, err_high], label=label, style=style, color=color)

    def filter_common_points(x1, y1_, x2, y2_):
        common_points = np.intersect1d(x1, x2)
        y1 = copy.deepcopy(y1_)
        y2 = copy.deepcopy(y2_)
        if isinstance(y1, list):
            for i in range(len(y1)):
                y1[i] = y1[i][np.isin(x1, common_points)]
        else:
            y1 = y1[np.isin(x1, common_points)]
        if isinstance(y2, list):
            for i in range(len(y2)):
                y2[i] = y2[i][np.isin(x2, common_points)]
        else:
            y2 = y2[np.isin(x2, common_points)]
        return common_points, y1, y2

    def NS2SoB(x):
        return x*(100000/95710)/378303

    plt.sca(axs[1])
    plt.xlabel("$S/B$")
    plt.ylabel("SI")
    plt.grid()
    print(x_CS)
    print(x_fit)
    
    valid_CS, invalid_CS = find_stat_validity(median_CS)
    valid_fit, invalid_fit = find_stat_validity(median_fit)
    
    x, CS, fit = filter_common_points(x_CS, [median_CS, err_low_CS, err_high_CS, valid_CS], x_fit, [median_fit, valid_fit])
    valid_CS = CS[3]
    valid_fit = fit[1]
    cut=np.max(np.where(np.logical_and(np.logical_not(valid_CS), np.logical_not(valid_fit))))+1
    print(cut)
    valid_CS = CS[3][cut:]
    valid_fit = fit[1][cut:]
    median_fit = fit[0][cut:]
    median_CS = CS[0][cut:]
    err_low_CS = CS[1][cut:]
    err_high_CS = CS[2][cut:]
    x=x[::-1][cut:]
    print(x)
    #plt.axhline(1, color="maroon", linestyle="dashed")
    #plt.plot(NS2SoB(x[::-1]), median_CS/median_fit, color="blue")

    draw_line_yerr(NS2SoB(x), median_CS/median_fit, [err_low_CS/median_fit, err_high_CS/median_fit], style=style, color="blue", validd=valid_CS)
    draw_line_yerr(NS2SoB(x), median_CS/median_CS, style=style, color="maroon", validd="all", linestyle="dashed")

    plt.xscale("log")
    plt.ylim(0, None)
    plt.xlim(NS2SoB(10**2), NS2SoB(10**4))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save the plot
    if plot_idealised and not plot_realistic:
        plt.savefig("plots/main/significances_idealised.png", bbox_inches="tight", dpi=300)
    elif plot_realistic and not plot_idealised:
        plt.savefig("plots/main/significances_realistic.png", bbox_inches="tight", dpi=300)
    else:
        plt.savefig("plots/main/significances.png", bbox_inches="tight", dpi=300)



if __name__ == "__main__":
    significance_plot(plot_idealised = True, 
                      plot_realistic = False,
                      use_cs_name = "Idealised CS")
    significance_plot(plot_idealised = False,
                        plot_realistic = True)