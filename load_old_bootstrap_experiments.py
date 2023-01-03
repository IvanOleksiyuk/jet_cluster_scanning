import pickle


def load_old_bootstrap_experiments05_1():
    save_path = (
        "char/old_char/BS26w60wk50MBret0con0.0W100ste200rewsqrtsme1ID1/"
    )
    counts_windows_boot = []
    res = pickle.load(open(save_path + "res.pickle", "rb"))
    counts_windows_boot += res["counts_windows_boot"]
    # print("found boostraps: ", len(counts_windows_boot))
    for i in range(46):
        res = pickle.load(
            open(
                "char/old_char/BS26w60wk50MBret0con0.0W100ste200rewsqrtsme1ID0/"
                + "res{0:04d}.pickle".format(i),
                "rb",
            )
        )
        counts_windows_boot += res["counts_windows_boot"]
    for i in range(40):
        res = pickle.load(
            open(
                "char/old_char/BS26w60wk50MBret0con0.0W100ste200rewsqrtsme1ID1/"
                + "res{0:04d}.pickle".format(i),
                "rb",
            )
        )
        counts_windows_boot += res["counts_windows_boot"]
    for i in range(37):
        res = pickle.load(
            open(
                "char/old_char/BS26w60wk50MBret0con0.0W100ste200rewsqrtsme1ID2/"
                + "res{0:04d}.pickle".format(i),
                "rb",
            )
        )
        counts_windows_boot += res["counts_windows_boot"]
    for i in range(44):
        res = pickle.load(
            open(
                "char/old_char/BS26w60wk50MBret0con0.0W100ste200rewsqrtsme1ID3/"
                + "res{0:04d}.pickle".format(i),
                "rb",
            )
        )
        counts_windows_boot += res["counts_windows_boot"]
    for i in range(18):
        res = pickle.load(
            open(
                "char/old_char/BS26w60wk50MBret0con0.0W100ste200rewsqrtsme1ID4/"
                + "res{0:04d}.pickle".format(i),
                "rb",
            )
        )
        counts_windows_boot += res["counts_windows_boot"]
    for i in range(4):
        res = pickle.load(
            open(
                "char/old_char/BS26w60wk50MBret0con0.0W100ste200rewsqrtsme1ID5/"
                + "res{0:04d}.pickle".format(i),
                "rb",
            )
        )
        counts_windows_boot += res["counts_windows_boot"]

    bootstraps = len(counts_windows_boot)
    print("found boostraps: ", bootstraps)
    return counts_windows_boot


def load_old_bootstrap_experiments00():
    counts_windows_boot = []
    for i in range(47):  # 26
        res = pickle.load(
            open(
                "char/old_char/BS26w60wk50MBret0con0.0W100ste200rewnonesme0ID0/"
                + "res{0:04d}.pickle".format(i),
                "rb",
            )
        )
        counts_windows_boot += res["counts_windows_boot"]
    for i in range(49):  # 26
        res = pickle.load(
            open(
                "char/old_char/BS26w60wk50MBret0con0.0W100ste200rewnonesme0ID1/"
                + "res{0:04d}.pickle".format(i),
                "rb",
            )
        )
        counts_windows_boot += res["counts_windows_boot"]
    for i in range(45):  # 26
        res = pickle.load(
            open(
                "char/old_char/BS26w60wk50MBret0con0.0W100ste200rewnonesme0ID1/"
                + "res{0:04d}.pickle".format(i),
                "rb",
            )
        )
        counts_windows_boot += res["counts_windows_boot"]
    for i in range(44):  # 26
        res = pickle.load(
            open(
                "char/old_char/BS26w60wk50MBret0con0.0W100ste200rewnonesme0ID1/"
                + "res{0:04d}.pickle".format(i),
                "rb",
            )
        )
        counts_windows_boot += res["counts_windows_boot"]
    for i in range(15):
        res = pickle.load(
            open(
                "char/old_char/BS26w60wk50MBret0con0.0W100ste200rewsqrtsme1ID4/"
                + "res{0:04d}.pickle".format(i),
                "rb",
            )
        )
        counts_windows_boot += res["counts_windows_boot"]

    bootstraps = len(counts_windows_boot)
    print("found boostraps: ", bootstraps)
    return counts_windows_boot
