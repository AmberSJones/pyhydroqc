import numpy as np
import matplotlib.pyplot as plt

# colors = ['#63ACBE', '#EE442F', '#601A4A']  # retro palette
# colors = ['#0F2080', '#F5793A', '#A95AA1']  # zesty palette
colors = ['#0C7BDC', '#F3870D', '#24026A']  # custom
models = ['ARIMA', 'LSTM\nunivar', 'LSTM\nunivar\nbidir', 'LSTM\nmulti', 'LSTM\nmulti\nbidir']
cols = ['Temperature', 'Specific Conductance', 'pH', 'Dissolved Oxygen']
rows = ['Franklin Basin', 'Tony Grove', 'Water Lab', 'Main Street', 'Mendon', 'Blacksmith Fork']
ind = np.arange(5)

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
