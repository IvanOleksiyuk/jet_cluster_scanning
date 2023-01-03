import matplotlib.pyplot as plt
import scipy
import numpy as np


def curvefit_eval(anomaly_poor_sp, anomaly_rich_sp, binning, tf, plot=True):

    # curvefit thingy
    bin_widths = binning.T[1] - binning.T[0]
    window_centers = anomaly_poor_sp.x

    bg = scipy.interpolate.interp1d(window_centers, anomaly_poor_sp.y[0])
    p0_mu = window_centers[
        np.argmax(anomaly_rich_sp.y[0] - anomaly_poor_sp.y[0])
    ]

    def f(x, w, n, mu, sig):
        return w * bg(x) + n * (bin_widths) * tf / np.sqrt(
            2 * np.pi
        ) / sig * np.exp(-((x - mu) ** 2) / 2 / sig**2)

    p0 = (1, 0, p0_mu, 20)
    print("p0_mu", p0_mu)
    rrr = scipy.optimize.curve_fit(
        f,
        window_centers,
        anomaly_rich_sp.y[0],
        sigma=np.sqrt(
            anomaly_rich_sp.err[0] ** 2 + anomaly_poor_sp.err[0] ** 2
        ),
        p0=p0,
        bounds=(
            [0, 0, binning.min(), 10],
            [2, 10000, binning.max(), (binning.max() - binning.min()) / 2],
        ),
    )
    print(rrr[0])
    # likelyhood spectrum
    # plt.plot(window_centers, f(window_centers, *p0), color="green")
    chisq_fit = np.mean(
        (anomaly_rich_sp.y[0] - f(window_centers, *rrr[0])) ** 2
        / (anomaly_rich_sp.err[0] ** 2 + anomaly_poor_sp.err[0] ** 2)
    )
    if plot:
        plt.plot(
            window_centers,
            f(window_centers, *rrr[0]),
            color="green",
            label="Curvefit $w=${:.03f}, $n=${:.01f}, \n $\mu=${:.01f}, $\sigma=${:.01f}, \n".format(
                *rrr[0]
            )
            + r"$\tilde{\chi}^2/n_{dof}=$"
            + "{:.3f}".format(chisq_fit),
        )
        plt.legend()
