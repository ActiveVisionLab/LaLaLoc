def scenes_split(split):
    splits = {
        "train": (
            list(range(0, 3000)),
            [
                335,
                683,
                1192,
                1753,
                1852,
                2205,
                2209,
                2223,
                2339,
                2357,
                2401,
                2956,
                2309,
                278,
                379,
                1212,
                1840,
                1855,
                2025,
                2110,
                2593,
            ],
        ),
        "val": (list(range(3000, 3250)), [2110, 3086, 3117, 3121, 3239]),
        "test": (list(range(3250, 3500)), [3307]),
    }
    ids, to_remove = splits[split]
    ids = [i for i in ids if i not in to_remove]
    return ids
