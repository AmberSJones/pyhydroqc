import numpy as np
import matplotlib.pyplot as plt

# colors = ['#63ACBE', '#EE442F', '#601A4A']  # retro palette
# colors = ['#0F2080', '#F5793A', '#A95AA1']  # zesty palette
colors = ['#0C7BDC', '#F3870D', '#24026A']  # custom
models = ['ARIMA', 'LSTM\nunivar', 'LSTM\nunivar\nbidir', 'LSTM\nmulti', 'LSTM\nmulti\nbidir']
cols = ['Temperature', 'Specific Conductance', 'pH', 'Dissolved Oxygen']
rows = ['Franklin Basin', 'Tony Grove', 'Water Lab', 'Main Street', 'Mendon', 'Blacksmith Fork']
ind = np.arange(5)

# To extract data from site_detect object from code output #

TP_all = []
FN_all = []
FP_all = []
TP_rules = []
FP_rules = []
TP_aggregate = []
FN_aggregate = []
FP_aggregate = []

for j in range(0, len(site_detect)):
    for i in range(0, len(site_detect[j].LSTM_univar)):
        TP_series = (site_detect[j].ARIMA[i].metrics.true_positives,
                 site_detect[j].LSTM_univar[i].metrics.true_positives,
                 site_detect[j].LSTM_univar_bidir[i].metrics.true_positives,
                 site_detect[j].LSTM_multivar.metrics_array[i].true_positives,
                 site_detect[j].LSTM_multivar_bidir.metrics_array[i].true_positives)
        TP_all.append((TP_series))

        FN_series = (site_detect[j].ARIMA[i].metrics.false_negatives,
                 site_detect[j].LSTM_univar[i].metrics.false_negatives,
                 site_detect[j].LSTM_univar_bidir[i].metrics.false_negatives,
                 site_detect[j].LSTM_multivar.metrics_array[i].false_negatives,
                 site_detect[j].LSTM_multivar_bidir.metrics_array[i].false_negatives)
        FN_all.append(FN_series)

        FP_series = (site_detect[j].ARIMA[i].metrics.false_positives,
                 site_detect[j].LSTM_univar[i].metrics.false_positives,
                 site_detect[j].LSTM_univar_bidir[i].metrics.false_positives,
                 site_detect[j].LSTM_multivar.metrics_array[i].false_positives,
                 site_detect[j].LSTM_multivar_bidir.metrics_array[i].false_positives)
        FP_all.append(FP_series)

        TP_rules_series = site_detect[j].rules_metrics[i].true_positives
        TP_rules.append(TP_rules_series)

        FP_rules_series = site_detect[j].rules_metrics[i].false_positives
        FP_rules.append(FP_rules_series)

        TP_agg_series = site_detect[j].aggregate_metrics[i].true_positives
        TP_aggregate.append(TP_agg_series)

        FN_agg_series = site_detect[j].aggregate_metrics[i].false_negatives
        FN_aggregate.append(FN_agg_series)

        FP_agg_series = site_detect[j].aggregate_metrics[i].false_positives
        FP_aggregate.append(FP_agg_series)

 # To create and modify plots without running modeling code #

TP_all = [(849, 792, 792, 792, 792),
             (6871, 2437, 2487, 6813, 6815),
             (15601, 13330, 13285, 15211, 12864),
             (645, 590, 570, 620, 570),
             (34678, 34675, 34675, 34675, 34675),
             (7468, 4703, 1300, 4589, 3622),
             (2774, 1766, 1686, 1830, 1705),
             (2320, 768, 750, 768, 756),
             (1114, 1024, 1023, 1020, 1011),
             (9599, 7873, 9159, 8005, 7890),
             (10451, 10270, 9964, 10260, 9920),
             (12142, 11873, 11905, 13352, 11909),
             (279, 287, 328, 294, 293),
             (3449, 3071, 3177, 3257, 3264),
             (12873, 12798, 12823, 12830, 12792),
             (7818, 7545, 7544, 7576, 7544),
             (5536, 5516, 5517, 5516, 5516),
             (13376, 11495, 11494, 13412, 11458),
             (16683, 16418, 13702, 13811, 13663),
             (6032, 4783, 5524, 5627, 4810),
             (975, 943, 943, 947, 947),
             (2478, 2304, 2117, 2152, 1984),
             (5724, 5679, 5653, 5657, 5653),
             (2768, 2958, 2785, 2961, 2786)]

FN_all = [(43, 88, 88, 88, 88),
             (77, 4497, 4446, 132, 128),
             (260, 2511, 2556, 636, 2975),
             (58, 101, 119, 77, 119),
             (1, 1, 1, 1, 1),
             (188, 2927, 6328, 3036, 4002),
             (147, 1146, 1224, 1084, 1206),
             (869, 2398, 2414, 2400, 2410),
             (25, 98, 99, 104, 111),
             (588, 2294, 1001, 2170, 2278),
             (440, 608, 912, 620, 955),
             (1808, 2056, 2024, 585, 2019),
             (180, 172, 132, 165, 166),
             (203, 571, 464, 393, 376),
             (255, 320, 296, 294, 326),
             (313, 571, 572, 552, 572),
             (1, 4, 3, 4, 4),
             (954, 2819, 2820, 928, 2855),
             (98, 344, 3056, 2950, 3095),
             (88, 1298, 551, 516, 1276),
             (760, 771, 770, 767, 766),
             (24, 159, 348, 325, 476),
             (60, 97, 120, 118, 120),
             (289, 99, 271, 99, 270)]

FP_all = [(373, 253, 252, 261, 252),
             (209, 151, 147, 189, 211),
             (1606, 1565, 1563, 1568, 1564),
             (3050, 2931, 2913, 2950, 2919),
             (6047, 6015, 6014, 6014, 6014),
             (72, 17, 6, 83, 29),
             (209, 149, 143, 152, 154),
             (1386, 1266, 1261, 1312, 1294),
             (71, 61, 73, 66, 59),
             (57, 91, 53, 159, 64),
             (70, 23, 21, 51, 49),
             (112, 7, 9, 17, 14),
             (196, 210, 214, 216, 212),
             (384, 244, 263, 299, 290),
             (108, 23, 31, 55, 27),
             (71, 12, 11, 20, 23),
             (216, 207, 207, 207, 211),
             (82, 27, 57, 130, 49),
             (31, 8, 5, 20, 12),
             (4030, 3716, 3720, 3752, 3735),
             (13, 0, 0, 1, 0),
             (2178, 1721, 1645, 1785, 1704),
             (66, 47, 45, 52, 57),
             (182, 169, 168, 178, 177)]

TP_rules = [783,
             1887,
             12079,
             564,
             34677,
             1178,
             595,
             499,
             985,
             7647,
             9743,
             11812,
             279,
             502,
             12743,
             7538,
             5499,
             11435,
             6468,
             3206,
             936,
             415,
             1014,
             2619]

FP_rules = [251,
             145,
             1560,
             2895,
             6004,
             0,
             140,
             1508,
             58,
             2,
             0,
             0,
             195,
             1,
             0,
             0,
             206,
             0,
             0,
             3703,
             0,
             1337,
             438,
             201]

TP_aggregate = [(849,
                 6914,
                 15620,
                 645),
                (34678,
                 7484,
                 2779,
                 2338),
                (1126,
                 9627,
                 10499,
                 12169),
                (329,
                 3450,
                 12892,
                 7818),
                (5536,
                 13376,
                 16686,
                 6061),
                (975,
                 2501,
                 5731,
                 2982)]

FN_aggregate = [(43,
                 71,
                 241,
                 58),
                (1,
                 188,
                 147,
                 865),
                (13,
                 577,
                 399,
                 1781),
                (131,
                 203,
                 236,
                 313),
                (1,
                 954,
                 95,
                 72),
                (760,
                 24,
                 54,
                 96)]

FP_aggregate = [(377,
                 267,
                 1611,
                 3092),
                (6048,
                 102,
                 225,
                 1428),
                (89,
                 177,
                 122,
                 130),
                (222,
                 628,
                 136,
                 90),
                (220,
                 159,
                 43,
                 4061),
                (13,
                 2858,
                 75,
                 195)]

# PLOT FOR ALL MODELS #
#######################
fig, axes = plt.subplots(nrows=6, ncols=4)
for i, ax in enumerate(axes.flat):
    p1 = ax.bar(ind, TP_all[i], color=colors[0])
    p2 = ax.bar(ind, FN_all[i], color=colors[1], bottom=np.array(TP_all[i]))
    p3 = ax.bar(ind, FP_all[i], color=colors[2], bottom=np.array(FN_all[i])+np.array(TP_all[i]))
    p4 = ax.axhline(y=TP_rules[i], linewidth=1.5, linestyle='--', color='k')
    p5 = ax.axhline(y=np.array(FN_all[i][0]) + np.array(TP_all[i][0]) + FP_rules[i], linewidth=1.5, linestyle='--', color='gray')
    # ax.legend((p1[0],p2[0],p3[0]),('True Positives', 'False Negatives', 'False Positives'))
    ax.set_xticks(ind)
    if(i >= 20):
        ax.set_xticklabels(models)
    else:
        ax.set_xticklabels(('', '', '', '', ''))
for ax, col in zip(axes[0], cols):
    ax.set_title(col)
for ax, row in zip(axes[:, 0], rows):
    ax.set_ylabel(row, size='large')
plt.figlegend((p1[0], p2[0], p3[0], p4, p5),
              ('True Positives', 'False Negatives', 'False Positives', 'Rules Based\nTrue Positives', 'Rules Based\nFalse Positives'),
              loc=(0.905, 0.47))
plt.show()


# PLOT FOR AGGREGATED MODELS #
##############################
titles = ['Franklin Basin', 'Tony Grove', 'Water Lab', 'Main Street', 'Mendon', 'Blacksmith Fork']
labels = ['Temperature', 'Specific Conductance', 'pH', 'Dissolved Oxygen']
colors = ['#0C7BDC', '#F3870D', '#24026A']  # custom

fig, axes = plt.subplots(nrows=1, ncols=6)
for i, ax in enumerate(axes.flat):
    p1 = ax.bar(labels, TP_aggregate[i], color=colors[0])
    p2 = ax.bar(labels, FN_aggregate[i], color=colors[1], bottom=np.array(TP_aggregate[i]))
    p3 = ax.bar(labels, FP_aggregate[i], color=colors[2], bottom=np.array(TP_aggregate[i])+np.array(FN_aggregate[i]))
plt.show()

fig, axes = plt.subplots(nrows=6, ncols=4)
for i, ax in enumerate(axes.flat):
    p1 = ax.bar(1, TP_aggregate[i], color=colors[0])
    p2 = ax.bar(1, FN_aggregate[i], color=colors[1], bottom=np.array(TP_aggregate[i]))
    p3 = ax.bar(1, FP_aggregate[i], color=colors[2], bottom=np.array(FN_aggregate[i])+np.array(TP_aggregate[i]))
    p4 = ax.axhline(y=TP_rules[i], linewidth=1.5, linestyle='--', color='k')
    p5 = ax.axhline(y=np.array(FN_aggregate[i]) + np.array(TP_aggregate[i]) + FP_rules[i], linewidth=1.5, linestyle='--', color='gray')
    # ax.legend((p1[0],p2[0],p3[0]),('True Positives', 'False Negatives', 'False Positives'))
for ax, col in zip(axes[0], cols):
    ax.set_title(col)
for ax, row in zip(axes[:, 0], rows):
    ax.set_ylabel(row, size='large')
plt.figlegend((p1[0], p2[0], p3[0], p4, p5),
              ('True Positives', 'False Negatives', 'False Positives'
               , 'Rules Based\nTrue Positives', 'Rules Based\nFalse Positives'),
              loc=(0.905, 0.47))
plt.show()