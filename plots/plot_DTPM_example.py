import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams.update({'font.size': 14})

def plot_perf_ener_edp(run_script = False):
    fig, (ax_perf, ax_ener, ax_edp) = plt.subplots(3, 1, figsize=(10,8))

    x_labels = ["Performance", "Ondemand", "Powersave", "IL-DTPM"]

    combined_csv = []
    file_list = ["results-Perf.csv", "results-OD.csv", "results-Powersave.csv", "results-IL-DTPM.csv"]
    for file_name in file_list:
        if run_script:
            csv_file = pd.read_csv("./reports/" + file_name)
        else:
            csv_file = pd.read_csv("../reports/" + file_name)
        combined_csv.append(csv_file.tail(1))

    combined_csv = pd.concat(combined_csv, sort=False)

    perf_values = combined_csv["Execution time(us)"].to_list()
    y_perf = [i / perf_values[0] for i in perf_values]
    ener_values = pd.to_numeric(combined_csv["Total energy consumption(J)"]).to_list()
    y_ener = [i / ener_values[0] for i in ener_values]
    edp_values = pd.to_numeric(combined_csv["EDP"]).to_list()
    y_edp = [i / edp_values[0] for i in edp_values]

    x = np.arange(len(x_labels))
    ax_perf.bar(x, y_perf, color='#800020', edgecolor='black', width=0.5)
    ax_ener.bar(x, y_ener, color='#2F3E5E', edgecolor='black', width=0.5)
    ax_edp.bar(x, y_edp, color='#008000', edgecolor='black', width=0.5)

    ax_perf.set_axisbelow(True)
    ax_perf.yaxis.grid(color='gray', linestyle='dashed', linewidth=0.5, zorder=0)
    ax_ener.set_axisbelow(True)
    ax_ener.yaxis.grid(color='gray', linestyle='dashed', linewidth=0.5, zorder=0)
    ax_edp.set_axisbelow(True)
    ax_edp.yaxis.grid(color='gray', linestyle='dashed', linewidth=0.5, zorder=0)

    ax_perf.set_ylabel("Exec. Time")
    ax_ener.set_ylabel("Energy Cons.")
    ax_edp.set_ylabel("EDP")

    ax_perf.set_xticks(x, minor=False)
    ax_perf.set_xticklabels(x_labels, fontdict=None, minor=False)
    ax_ener.set_xticks(x, minor=False)
    ax_ener.set_xticklabels(x_labels, fontdict=None, minor=False)
    ax_edp.set_xticks(x, minor=False)
    ax_edp.set_xticklabels(x_labels, fontdict=None, minor=False)

    y_lim_perf = 2.51
    y_lim_ener = 1.10
    y_lim_edp = 1.10
    ax_perf.set_ylim([0, y_lim_perf])
    ax_perf.set_yticks(np.arange(0, y_lim_perf, 0.25))
    ax_ener.set_ylim([0, y_lim_ener])
    ax_ener.set_yticks(np.arange(0, y_lim_ener, 0.1))
    ax_edp.set_ylim([0, y_lim_edp])
    ax_edp.set_yticks(np.arange(0, y_lim_edp, 0.1))

    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_perf_ener_edp()