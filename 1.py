import numpy as np
def mutate_part2(part2, part1_len):
    rng = np.random.default_rng()
    m = len(part2)
    if m == 0:
        return part2  # 空的直接返回

    keep_num = rng.integers(1, m + 1)  # +1 保证至少留一个
    keep_indices = rng.choice(range(m), size=keep_num, replace=False)
    kept = np.array([part2[i] for i in keep_indices])

    new_needed = m - keep_num
    all_choices = np.setdiff1d(np.arange(2, part1_len-1), kept)

    if new_needed > 0 and all_choices.size >= new_needed:
        new_vals = rng.choice(all_choices, size=new_needed, replace=False)
        return np.sort(np.concatenate([kept, new_vals]))
    else:
        return np.sort(kept)

if __name__ == '__main__':
    part2 = [21,50,75,99]
    res = mutate_part2(part2, 120)
    print(res)