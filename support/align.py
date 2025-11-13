import re

def sort_instructions(lines):
    cnot_instr, hh_swaps, vv_swaps, waits = [], [], [], []

    # 정규식: type, 앞뒤 숫자, LQ, data 추출
    swap_pattern = re.compile(r'^(?P<type>(hh|vv)-SWAP)\s+(\d+),\s+(\d+)\s+\(([^,]+),\s+(LQ\d+)-data(\d+)\)')
    wait_pattern = re.compile(r'^(hh|vv)-SWAP-Wait\s+(\d+)\s+\((LQ\d+)-data(\d+)\)')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if "CNOT" in line:
            cnot_instr.append(line)
        elif "SWAP-Wait" in line:
            m = wait_pattern.match(line)
            if m:
                lq, data = m.group(3), int(m.group(4))
                waits.append((lq, data, line))
        elif "SWAP" in line:
            m = swap_pattern.match(line)
            if m:
                typ = m.group("type")
                lq, data = m.group(6), int(m.group(7))
                if typ == "hh-SWAP":
                    hh_swaps.append((lq, data, line))
                else:
                    vv_swaps.append((lq, data, line))

    # 정렬: LQ 번호, data 인덱스 기준
    key_func = lambda x: (int(x[0][2:]), x[1])

    cnot_instr.sort()  # 단순 정렬
    hh_swaps.sort(key=key_func)
    vv_swaps.sort(key=key_func)
    waits.sort(key=key_func)

    # 하나의 목록으로 합치기
    result = []
    result.extend(cnot_instr)
    result.extend([x[2] for x in hh_swaps])
    result.extend([x[2] for x in vv_swaps])
    result.extend([x[2] for x in waits])

    return result


if __name__ == "__main__":
    input_data = """
    vv-SWAP  37,  55 ( dummy83, LQ1-data0) : False, True
    vv-SWAP  21,  39 ( dummy68, LQ1-data3) : False, True
    vv-SWAP 117,  99 ( dummy60, LQ1-data4) : False, True
    vv-SWAP   4,  22 ( dummy58, LQ1-data6) : False, True
    hh-SWAP  26,  25 ( dummy49, LQ1-data8) : False, True
    vv-SWAP 114,  96 ( dummy40, LQ1-data10) : False, True
    vv-SWAP  56,  38 ( dummy94, LQ1-data13) : False, True
    vv-SWAP  76,  58 ( dummy54, LQ1-data14) : False, True    
    vv-SWAP  60,  78 ( dummy17, LQ1-data15) : False, True
    vv-SWAP  79,  61 ( dummy62, LQ1-data16) : False, True
    hh-SWAP   7,   6 ( dummy35, LQ1-data17) : False, True
    vv-SWAP 111,  93 ( dummy27, LQ1-data18) : False, True
    vv-SWAP  92, 110 ( dummy14, LQ1-data20) : False, True
    hh-SWAP  43,  42 ( dummy33, LQ1-data21) : False, True
    hh-SWAP  82,  83 ( dummy43, LQ2-data19) : False, True
    hh-SWAP  30,  31 ( dummy13, LQ2-data3) : False, True
    hh-SWAP 103, 104 ( dummy82, LQ2-data10) : False, True
    vv-SWAP  46,  64 ( dummy77, LQ2-data0) : False, True
    vv-SWAP  27,  45 ( dummy28, LQ2-data20) : False, True
    vv-SWAP 120, 102 ( dummy80, LQ2-data4) : False, True
    vv-SWAP  65,  47 ( dummy92, LQ2-data12) : False, True
    hh-SWAP  80,  81 ( dummy25, LQ2-data2) : False, True
    hh-SWAP   8,   9 ( dummy29, LQ2-data7) : False, True
    vv-SWAP 124, 106 ( dummy30, LQ2-data18) : False, True
    hh-SWAP 137, 138 ( dummy85, LQ2-data16) : False, True
    hh-SWAP  33,  34 ( dummy59, LQ2-data8) : False, True
    vv-SWAP  68,  50 ( dummy67, LQ2-data22) : False, True
    hh-SWAP  13,  14 ( dummy53, LQ2-data6) : False, True
    hh-SWAP  11,  12 ( dummy18, LQ2-data17) : False, True
 hh-SWAP-Wait 100 (LQ2-data13)       : True
 hh-SWAP-Wait  67 (LQ2-data9)        : True
 hh-SWAP-Wait  84 (LQ2-data1)        : True
 hh-SWAP-Wait 115 (LQ1-data5)        : True
 hh-SWAP-Wait  63 (LQ2-data21)       : True
 hh-SWAP-Wait 121 (LQ2-data5)        : True
 hh-SWAP-Wait   3 (LQ1-data7)        : True
 hh-SWAP-Wait  77 (LQ1-data2)        : True
 hh-SWAP-Wait 101 (LQ2-data11)       : True
 hh-SWAP-Wait 132 (LQ1-data19)       : True
 hh-SWAP-Wait  75 (LQ1-data1)        : True
 hh-SWAP-Wait  48 (LQ2-data15)       : True
 hh-SWAP-Wait 116 (LQ1-data12)       : True
 hh-SWAP-Wait  44 (LQ1-data22)       : True
 hh-SWAP-Wait  73 (LQ1-data11)       : True
 hh-SWAP-Wait  62 (LQ1-data9)        : True
 hh-SWAP-Wait 105 (LQ2-data14)       : True
    """

    lines = input_data.strip().split("\n")
    sorted_result = sort_instructions(lines)

    print("\n".join(sorted_result))