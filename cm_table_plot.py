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

plt.figlegend((p1[0], p2[0], p3[0]
               , p4, p5
               ),
              ('True Positives', 'False Negatives', 'False Positives'
               , 'Rules Based\nTrue Positives', 'Rules Based\nFalse Positives'
               ),
              loc=(0.905, 0.47))
plt.show()


# PLOT FOR AGGREGATED MODELS #
# not sure how useful this is #
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

plt.figlegend((p1[0], p2[0], p3[0]
               , p4, p5
               ),
              ('True Positives', 'False Negatives', 'False Positives'
               , 'Rules Based\nTrue Positives', 'Rules Based\nFalse Positives'
               ),
              loc=(0.905, 0.47))
plt.show()





# old numbers. rules based are made up.


TP = [(849, 836, 835, 835, 835),            # FB temp
      (6871, 4032, 4020, 4646, 4623),       # FB cond
      (15692, 15148, 15126, 14898, 14751),  # FB ph
      (645, 593, 570, 590, 574),            # FB do
      (34755, 34750, 34750, 34750, 34750),  # TG temp
      (7468, 3047, 1729, 4940, 2411),       # TG cond
      (2783, 2643, 2515, 2592, 2539),       # TG ph
      (2320, 764, 756, 792, 2188),          # TG do
      (1114, 1047, 1031, 1011, 1024),       # WL temp
      (9668, 8255, 8249, 8309, 8119),       # WL cond
      (10499, 10341, 10042, 10401, 10016),  # WL ph
      (13573, 13376, 13429, 13409, 13415),  # WL do
      (311, 346, 320, 322, 346),            # MS temp
      (3449, 3049, 3082, 3088, 3108),       # MS cond
      (12873, 12808, 12805, 12832, 12782),  # MS ph
      (7818, 7544, 7544, 7544, 7544),       # MS do
      (5536, 5510, 5517, 5509, 5515),       # Me temp
      (13375, 11514, 11513, 11498, 11863),  # Me cond
      (16681, 16420, 16343, 16373, 16301),  # Me ph
      (6031, 5569, 5520, 5576, 4831),       # Me do
      (975, 942, 947, 947, 948),            # BF temp
      (2478, 1917, 1171, 2150, 1614),       # BF cond
      (5724, 5656, 5653, 5491, 5469),       # BF ph
      (2768, 2967, 2785, 2982, 2786),       # BF do
     ]


TP_rules = [783,  # FB temp
            1887,  # FB cond
            12079,  # FB ph
            564,  # FB do
            30000,  # TG temp
            1000,  # TG cond
            2000,  # TG ph
            700,  # TG do
            900,  # WL temp
            8000,  # WL cond
            10000,  # WL ph
            13000,  # WL do
            300,  # MS temp
            3000,  # MS cond
            12000,  # MS ph
            7000,  # MS do
            5000,  # Me temp
            10000,  # Me cond
            15000,  # Me ph
            4000,  # Me do
            800,  # BF temp
            1500,  # BF cond
            5000,  # BF ph
            2000,  # BF do
            ]


FN = [(43, 45, 45, 45, 45),                 # FB temp
      (77, 2904, 2913, 2288, 2310),         # FB cond
      (260, 781, 802, 1042, 1174),          # FB ph
      (58, 99, 119, 102, 115),              # FB do
      (0, 0, 0, 0, 0),                      # TG temp
      (188, 4583, 5895, 2705, 5213),        # TG cond
      (147, 278, 404, 331, 379),            # TG ph
      (869, 2401, 2408, 2390, 976),         # TG do
      (25, 75, 91, 111, 98),                # WL temp
      (531, 1914, 1918, 1862, 2049),        # WL cond
      (414, 562, 855, 503, 880),            # WL ph
      (373, 557, 499, 521, 513),            # WL do
      (148, 113, 139, 137, 113),            # MS temp
      (203, 588, 552, 553, 526),            # MS cond
      (255, 310, 313, 288, 336),            # MS ph
      (313, 572, 572, 572, 572),            # MS do
      (1, 10, 3, 11, 5),                    # Me temp
      (954, 2801, 2800, 2816, 2450),        # Me cond
      (98, 336, 410, 384, 452),             # Me ph
      (88, 527, 555, 529, 1242),            # Me do
      (760, 772, 766, 767, 765),            # BF temp
      (24, 541, 1285, 318, 842),            # BF cond
      (21, 37, 38, 33, 39),                 # BF ph
      (289, 91, 271, 86, 270),              # BF do
     ]

FP = [(863, 760, 738, 751, 750),            # FB temp
      (534, 484, 471, 487, 476),            # FB cond
      (24127, 24076, 24075, 24083, 24085),  # FB ph
      (3345, 3237, 3208, 3238, 3231),       # FB do
      (10680, 10647, 10645, 10644, 10644),  # TG temp
      (134, 81, 66, 88, 119),               # TG cond
      (17697, 17638, 17631, 17641, 17636),  # TG ph
      (2218, 2097, 2093, 2101, 2102),       # TG do
      (155, 156, 152, 155, 144),            # WL temp
      (89, 95, 84, 90, 154),                # WL cond
      (1543, 1501, 1505, 1537, 1514),       # WL ph
      (333, 234, 227, 230, 233),            # WL do
      (500, 516, 519, 515, 520),            # MS temp
      (447, 308, 294, 334, 316),            # MS cond
      (3200, 3124, 3117, 3135, 3138),       # MS ph
      (894, 834, 837, 843, 842),            # MS do
      (607, 598, 598, 598, 598),            # Me temp
      (82, 24, 31, 29, 72),                 # Me cond
      (4494, 4471, 4468, 4481, 4476),       # Me ph
      (4249, 3958, 3940, 3941, 4013),       # Me do
      (24, 12, 11, 11, 11),                 # BF temp
      (2178, 1668, 1671, 1667, 1668),       # BF cond
      (6298, 6278, 6277, 6282, 6282),       # BF ph
      (347, 335, 333, 336, 335),            # BF do
     ]

FP_rules = [251,  # FB temp
            145,  # FB cond
            1560,  # FB ph
            2895,  # FB do
            5000,  # TG temp
            30,  # TG cond
            12000,  # TG ph
            1500,  # TG do
            50,  # WL temp
            20,  # WL cond
            500,  # WL ph
            100,  # WL do
            200,  # MS temp
            100,  # MS cond
            1000,  # MS ph
            500,  # MS do
            200,  # Me temp
            5,  # Me cond
            1000,  # Me ph
            1000,  # Me do
            5,  # BF temp
            700,  # BF cond
            2000,  # BF ph
            100,  # BF do
            ]


fig, axes = plt.subplots(nrows=6, ncols=4)

for i, ax in enumerate(axes.flat):

    p1 = ax.bar(ind, TP[i], color=colors[0])
    p2 = ax.bar(ind, FN[i], color=colors[1], bottom=np.array(TP[i]))
    p3 = ax.bar(ind, FP[i], color=colors[2], bottom=np.array(FN[i])+np.array(TP[i]))
    p4 = ax.axhline(y=TP_rules[i], linewidth=1.5, linestyle='--', color='k')
    p5 = ax.axhline(y=np.array(FN[i][0]) + np.array(TP[i][0]) + FP_rules[i], linewidth=1.5, linestyle='--', color='gray')
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
              ('True Positives', 'False Negatives', 'False Positives', 'Rules Based True Positives', 'Rules Based False Positives'),
              loc=(0.905, 0.47))
plt.show()
