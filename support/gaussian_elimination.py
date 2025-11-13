import os
import simplejson as json
from icecream import ic
import numpy as np

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=100)

def rref_gf2(A, b=None, full_reduce=True):
    """
    GF(2)에서 행렬 A (0/1)와 선택적 우변 b (0/1)에 대해
    RREF(또는 Echelon)으로 변환.

    Parameters
    ----------
    A : (m, n) array_like of {0,1}, dtype bool/uint8 추천
    b : (m,) or (m,1) array_like of {0,1}, optional
    full_reduce : bool
        True  -> RREF (위/아래 모두 소거)
        False -> Echelon (아래만 소거)

    Returns
    -------
    A_red : (m, n) np.ndarray (uint8, 0/1)
    b_red : (m,) np.ndarray (uint8, 0/1) or None
    pivots : list[int]  피벗 열 인덱스
    """
    A = (np.array(A, copy=True) & 1).astype(np.uint8)
    m, n = A.shape

    b_arr = None
    if b is not None:
        b_arr = (np.array(b, copy=True).reshape(-1) & 1).astype(np.uint8)
        assert b_arr.shape[0] == m, "b 길이가 A 행 수와 같아야 합니다."

    row = 0
    pivots = []

    for col in range(n):
        # 1) 피벗행 찾기: row..m-1 중에서 A[:,col]==1인 첫 행
        # (argmax로 빠르게, 단 '1이 하나도 없을 수도' 있으니 체크)
        sub = A[row:, col]
        if not sub.any():
            continue
        pivot = row + int(sub.argmax())  # 첫 1의 위치

        # 2) 피벗행을 위로 올리기 (row와 swap)
        if pivot != row:
            A[[row, pivot]] = A[[pivot, row]]
            if b_arr is not None:
                b_arr[row], b_arr[pivot] = b_arr[pivot], b_arr[row]

        # 3) 같은 열의 다른 1들을 XOR로 소거
        if full_reduce:
            rows_to_elim = np.where((A[:, col] == 1) & (np.arange(m) != row))[0]
        else:
            rows_to_elim = np.where((A[row+1:, col] == 1))[0] + (row + 1)

        for r in rows_to_elim:
            A[r, :] ^= A[row, :]
            if b_arr is not None:
                b_arr[r] ^= b_arr[row]

        pivots.append(col)
        row += 1
        if row == m:
            break

    return A, b_arr, pivots


def solve_gf2(A, b):
    """
    GF(2) 선형시스템 Ax=b (0/1) 해 구하기.
    - 일관성 체크
    - 특정해 x0
    - 영공간(nullspace) 기저들 N = {n1, n2, ...}
    - 자유변수/피벗열 정보

    Returns
    -------
    (x0, null_basis, pivots, free_cols) or None  (해 없으면 None)
    """
    A_r, b_r, piv = rref_gf2(A, b, full_reduce=True)
    m, n = A_r.shape
    piv = list(piv)
    free = [j for j in range(n) if j not in piv]

    # 1) 일관성 체크: [0...0 | 1] 이면 해 없음
    if b_r is not None:
        zero_rows = (A_r.sum(axis=1) == 0)
        if np.any(zero_rows & (b_r == 1)):
            return None

    # 2) 특정해 x0: 피벗 열에 대해서만 값이 정해짐
    x0 = np.zeros(n, dtype=np.uint8)
    # RREF이므로 각 피벗행은 해당 피벗열 1, 나머지는 0
    # 자유열의 계수도 이미 위소거되어 있으므로 그냥 b_r의 값을 대입
    for i, c in enumerate(piv):
        x0[c] = b_r[i] if b_r is not None else 0

    # 3) 영공간 기저: 자유열마다 하나씩 구성
    null_basis = []
    for f in free:
        v = np.zeros(n, dtype=np.uint8)
        v[f] = 1
        # RREF에서 각 피벗행 i에 대해:
        # x[piv[i]] = XOR_j (A_r[i, free_j] * t_j)
        # Ax=0이 되려면 v[piv[i]] = A_r[i, f]
        for i, c in enumerate(piv):
            if A_r[i, f] == 1:
                v[c] ^= 1
        null_basis.append(v)

    return x0, null_basis, piv, free


def rank_gf2(A):
    """GF(2) 랭크(피벗 개수)"""
    _, _, piv = rref_gf2(A, None, full_reduce=False)
    return len(piv)

if __name__ == "__main__":
    # golay code
    # matrix = [
    # [0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    # [1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    # [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    # [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    # [1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    # [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    # [0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    # [0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    # [0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    # [1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # [1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    # [[15,7,3]] hamming code
    # parity_check_matrix = np.array([
    #     [0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],
    #     [0,0,0,1,1,1,1,0,0,0,0,1,1,1,1],
    #     [0,1,1,0,0,1,1,0,0,1,1,0,0,1,1],
    #     [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]])

    parity_check_matrix = np.array([
        [0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ])


    parity_check_matrix = np.array([
        [1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1]])


    A_rref, b_rref, pivots = rref_gf2(parity_check_matrix, None, full_reduce=True)
    print(A_rref)

    # zero_matrix = np.zeros((4, 15), dtype=np.bool)

    # stabilizer_matrix = np.block([
    #     [parity_check_matrix, zero_matrix],
    #     [zero_matrix, parity_check_matrix]])

    # ic(stabilizer_matrix)
    # A_rref, b_rref, pivots = rref_gf2(stabilizer_matrix, None, full_reduce=True)
    # print(A_rref)
