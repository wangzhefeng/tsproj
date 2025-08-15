def min_geese(s: str) -> int:
    """
    返回使得字符串 s 合法的最少大雁数量（并发数），否则返回 -1。
    约束：1 <= len(s) <= 1000，字符集仅 {'q','u','a','c','k'}。
    """
    # 基本检查
    if not (1 <= len(s) <= 1000): 
        return -1
    if set(s) - set("quack"):
        return -1

    # 阶段计数：当前有多少只雁卡在 q/u/a/c 上（k 代表已完成）
    q = u = a = c = 0
    max_concurrent = 0

    for ch in s:
        if ch == 'q':
            q += 1
        elif ch == 'u':
            if q == 0: return -1
            q -= 1; u += 1
        elif ch == 'a':
            if u == 0: return -1
            u -= 1; a += 1
        elif ch == 'c':
            if a == 0: return -1
            a -= 1; c += 1
        elif ch == 'k':
            if c == 0: return -1
            c -= 1  # 一只完成
        # 当前并发：正在叫的雁 = q+u+a+c
        max_concurrent = max(max_concurrent, q + u + a + c)

    # 结束时必须全部完成
    if q or u or a or c:
        return -1
    return max_concurrent


print(min_geese("quackquack"))                                # 1
print(min_geese("quqauackck"))                                # 2
print(min_geese("qaauuqcckk"))                                # -1
print(min_geese("quacquack"))                                 # -1
print(min_geese("qququaauqccauqkkcauqqkcauuqkcaaukccakkck"))  # 5