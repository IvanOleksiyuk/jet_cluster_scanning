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

def get_median_and_quar_err(Zs):
    median = np.median(Zs, axis=1)
    q1 = np.quantile(Zs, 0.25, axis=1)
    q3 = np.quantile(Zs, 0.75, axis=1)
    err_low = median - q1
    err_high = q3 - median
    return median, err_low, err_high

def draw_line_yerr(x, y, yerr, color="black", label=None, linestyle="solid", capsize=5, style="errorbar"):
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


def significance_plot(plot_idealised = True, 
                      plot_realistic = True, 
                      plot_BH = False,
                      use_points_insuff_stats = False,
                      style="band",
                      use_cs_name=False):
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load the data block + define some parameters

    # load the mass spectra
    data_path = "../../DATA/LHCO/"
    mjj_bg = np.load(data_path + "mjj_bkg_sort.npy")
    mjj_sg = np.load(data_path + "mjj_sig_sort.npy")

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
                        i % 10 * 10 ** (hunter.str_min + i // 10)
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

    fig = plt.figure(figsize=(5, 5))

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
        median, err_low, err_high = get_median_and_quar_err(Zs)
        x = np.array(results["contaminations"]) * len(mjj_bg)
        if use_cs_name:
            label = name.replace("_", " ")
        else:
            label = "Cluster scanning"
        draw_line_yerr(x, median, [err_low, err_high], label=label, style=style, color="blue")
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
            x = res["sig_fractions"] * len(mjj_sg)
            print(x)
            draw_line_yerr(x, median, [err_low, err_high], label=lab, style=style, color=color, linestyle="dashed")

    if plot_idealised:
        res = pickle.load(open("gf_results/MLSnormal_positiv3_parambootstrap_true10000x100_binning16_Zs_ideal.pickle", "rb"))
        x = res["sig_fractions"] * len(mjj_sg)
        median, err_low, err_high = get_median_and_quar_err(res["Zs"])
        draw_line_yerr(x, median, [err_low, err_high], label="Idealised fit", style=style, color="maroon", linestyle="dashed")

    if use_cs_name:
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    else:
        plt.legend(loc="lower right")
    plt.grid()
    plt.xscale("log")
    plt.xlim(10**2, 10**4)
    plt.xlabel("$\epsilon$")
    plt.ylabel("Z $[\sigma]$")

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
                      plot_realistic = False)
    significance_plot(plot_idealised = False,
                        plot_realistic = True)