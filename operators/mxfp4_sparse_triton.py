import torch, math
import triton
import triton.language as tl
from einops import rearrange


@triton.jit
def clip(x, min_val, max_val):
    return tl.minimum(tl.maximum(x, min_val), max_val)

@triton.jit
def to_mxfp4(tensor, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    FP32_EXPONENT_BIAS = 127.0
    FP32_MIN_NORMAL = tl.exp2(-FP32_EXPONENT_BIAS + 1)
    ebits, mbits = 2.0, 3.0
    emax = tl.exp2(ebits - 1)
    max_norm = tl.exp2(emax) * (tl.exp2(mbits-1) - 1) / tl.exp2(mbits-2)

    tensor = tl.reshape(tensor,(BLOCK_M, 2, 32))

    shared_exp = tl.max(tl.abs(tensor), axis=-1, keep_dims=True)

    mask = (shared_exp == 0).to(shared_exp.dtype)
    shared_exp = tl.floor(
        tl.log2(shared_exp + FP32_MIN_NORMAL * mask)
    )
    mask = (tensor > -FP32_EXPONENT_BIAS).to(tensor.dtype)
    tensor = tensor * mask
    shared_exp = shared_exp - emax
    scale_emax = tl.exp2(8.0-1.0) - 1
    shared_exp = tl.where(shared_exp > scale_emax, float('nan'), shared_exp)
    shared_exp = tl.where(shared_exp < -scale_emax, -scale_emax, shared_exp)

    tensor = tensor / (tl.exp2(shared_exp))
    mask = (tensor == 0).to(tensor.dtype)
    private_exp = tl.floor(
        tl.log2(tl.abs(tensor) + mask)
    )

    min_exp = -(tl.exp2(ebits-1)) + 2
    private_exp = tl.maximum(private_exp, min_exp)

    tensor = tensor / (tl.exp2(private_exp)) * (tl.exp2(mbits - 2))
    tensor_sign = (tensor > 0).to(tensor.dtype) - (tensor < 0).to(tensor.dtype)  # tensor_sign = torch.sign(tensor)
    tensor = tensor_sign * tl.floor(tl.abs(tensor) + 0.5)
    tensor = tensor / (tl.exp2(mbits - 2)) * (tl.exp2(private_exp))

    tensor = clip(tensor, -max_norm, max_norm)
    tensor = tl.where(tensor == float('inf'), float('inf'), tensor)
    tensor = tl.where(tensor == -float('inf'), -float('inf'), tensor)
    tensor = tl.where(tensor == float('nan'), float('nan'), tensor)

    recovered_tensor = tensor * (tl.exp2(shared_exp))
    recovered_tensor = tl.reshape(recovered_tensor, (BLOCK_M, BLOCK_N))
    return recovered_tensor


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q, qm, kv_len,
                    K_ptrs, K_fp_ptrs, V_ptrs, Pool_Mask_base, Seq_Mask_ptrs,  # 传入两组 Mask 的指针
                    stride_kn, stride_vn, stride_kfpn, stride_pmaskn, stride_smaskn,   # 传入两组 Mask 的列步长
                    start_m, sm_scale: tl.constexpr,
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
                    POOL_SIZE: tl.constexpr,
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,
                    stride_pmaskm: tl.constexpr,    # pool_mask 的行步长
                    ):
    lo, hi = 0, kv_len
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # 1. 加载 K / K_fp
        k_mask = offs_n[None, :] < (kv_len - start_n)
        k = tl.load(K_ptrs, mask = k_mask)
        k_fp = tl.load(K_fp_ptrs, mask = k_mask)

        # 2. 计算 Attention Score
        s_delta = tl.dot(qm, k_fp).to(tl.float32)
        qk = tl.dot(q, k) + s_delta
        qk_scale = sm_scale * 1.4426950408889634
        qk *= qk_scale

        # 3. 处理双重 Mask
        # (a) 处理 Pool Mask: 动态计算当前 start_n 对应的绝对 pooled column index
        abs_pooled_n = (start_n + offs_n) // POOL_SIZE
        Pool_Mask_ptrs = Pool_Mask_base + abs_pooled_n[None, :] * stride_pmaskn
        p_mask = tl.load(Pool_Mask_ptrs, mask=k_mask)

        # (b) 处理 Seq Mask: 直接加载当前的指针，指针会在循环末尾步进
        s_mask = tl.load(Seq_Mask_ptrs, mask=k_mask)

        # 结合两个 Mask (假设 True 代表保留, False 代表遮罩)
        combined_mask = p_mask & s_mask
        qk = qk + tl.where(combined_mask, 0, -1.0e6)

        # 4. Softmax 统计量更新
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        m_ij = tl.math.ceil(m_ij) # mij approximate


        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)

        p = to_mxfp4(p,BLOCK_M,BLOCK_N).to(tl.float16)

        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij

        acc = acc * alpha[:, None]

        # 5. 加载 V 并累加
        v = tl.load(V_ptrs, mask=offs_n[:, None] < (kv_len - start_n))


        acc += tl.dot(p, v, out_dtype=tl.float32)

        # 6. 更新状态
        m_i = m_ij

        # 7. 步进指针
        K_ptrs += BLOCK_N * stride_kn
        K_fp_ptrs += BLOCK_N * stride_kfpn
        V_ptrs += BLOCK_N * stride_vn
        # Pool Mask 在头部根据 abs_pooled_n 动态计算，不需要累加
        # Seq Mask 是常规按列排布的，需要累加
        Seq_Mask_ptrs += BLOCK_N * stride_smaskn

    return acc, l_i, m_i

@triton.jit
def _attn_fwd(Q, K, V, Qm, K_fp, Pool_Mask, Seq_Mask, Out, Lse,
              stride_qz, stride_qh, stride_qn,
              stride_kz, stride_kh, stride_kn,
              stride_vz, stride_vh, stride_vn,
              stride_oz, stride_oh, stride_on,
              stride_qmz, stride_qmh, stride_qmn,
              stride_kfpz, stride_kfph, stride_kfpn,
              stride_pmaskz, stride_pmaskh, stride_pmaskm, stride_pmaskn, # Pool Mask 步长
              stride_smaskn,
              qo_len, kv_len, H: tl.constexpr, num_kv_groups: tl.constexpr,
              sm_scale: tl.constexpr,
              HEAD_DIM: tl.constexpr,
              BLOCK_M: tl.constexpr,
              BLOCK_N: tl.constexpr,
              POOL_SIZE: tl.constexpr,
              STAGE: tl.constexpr,
              RETURN_LSE: tl.constexpr,
              ):
    start_m = tl.program_id(0)
    off_z = tl.program_id(2).to(tl.int64)
    off_h = tl.program_id(1).to(tl.int64)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)

    Q_ptrs = Q + (off_z * stride_qz + off_h * stride_qh) + offs_m[:, None] * stride_qn + offs_k[None, :]
    Qm_ptrs = Qm + (off_z * stride_qmz + off_h * stride_qmh) + offs_m[:, None] * stride_qmn + offs_k[None, :]

    K_ptrs = K + (off_z * stride_kz + (off_h // num_kv_groups) * stride_kh) + offs_n[None, :] * stride_kn + offs_k[:, None]
    K_fp_ptrs = K_fp + (off_z * stride_kfpz + (off_h // num_kv_groups) * stride_kfph) + offs_n[None, :] * stride_kfpn + offs_k[:, None]
    V_ptrs = V + (off_z * stride_vz + (off_h // num_kv_groups) * stride_vh) + offs_n[:, None] * stride_vn + offs_k[None, :]
    O_block_ptr = Out + (off_z * stride_oz + off_h * stride_oh) + offs_m[:, None] * stride_on + offs_k[None, :]

    # ---------------- 计算 Mask 指针 ----------------
    # 1. Pool Mask Base
    pooled_m = offs_m // POOL_SIZE
    Pool_Mask_base = (
        Pool_Mask +
        off_z * stride_pmaskz +
        off_h * stride_pmaskh +
        pooled_m[:, None] * stride_pmaskm
    )

    # 2. Seq Mask Pointers
    # 注意: 加入了 off_z 和 off_h 偏移防止跨 batch/head 数据混乱
    Seq_Mask_ptrs = Seq_Mask + offs_n[None, :] * stride_smaskn
    # ------------------------------------------------

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    q = tl.load(Q_ptrs, mask = offs_m[:, None] < qo_len)
    qm = tl.load(Qm_ptrs, mask = offs_m[:, None] < qo_len)

    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, qm, kv_len,
                                    K_ptrs, K_fp_ptrs, V_ptrs, Pool_Mask_base, Seq_Mask_ptrs,
                                    stride_kn, stride_vn, stride_kfpn,
                                    stride_pmaskn, stride_smaskn,  # 传入对应步长
                                    start_m, sm_scale,
                                    BLOCK_M, HEAD_DIM, BLOCK_N, POOL_SIZE,
                                    4 - STAGE, offs_m, offs_n,
                                    stride_pmaskm
                                    )
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask = (offs_m[:, None] < qo_len))

    if RETURN_LSE:
        lse_ptrs = Lse + (off_z * qo_len * H + off_h * qo_len) + offs_m
        l_i = tl.log2(l_i) + m_i
        tl.store(lse_ptrs, l_i, mask = (offs_m < qo_len))


def forward(q, k, v, qm, k_fp, pool_mask=None, seq_mask=None, sm_scale=1.0, tensor_layout="HND", output_dtype=torch.float16, return_lse=False, pool_size=128):
    BLOCK_M = 128
    BLOCK_N = 64
    stage = 1

    o = torch.empty(q.shape, dtype=output_dtype, device=q.device)

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(1), q.stride(2)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(1), v.stride(2)
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(1), o.stride(2)
        stride_bz_qm, stride_h_qm, stride_seq_qm = qm.stride(0), qm.stride(1), qm.stride(2)
        stride_bz_k_fp, stride_h_k_fp, stride_seq_k_fp = k_fp.stride(0), k_fp.stride(1), k_fp.stride(2)

        # ---------------- 1. 处理 Pool Mask ----------------
        pooled_q = (qo_len + pool_size - 1) // pool_size
        pooled_k = (kv_len + pool_size - 1) // pool_size

        if pool_mask is not None:
            pool_mask = pool_mask.contiguous()
            assert pool_mask.shape[0] == b and pool_mask.shape[1] == h_qo, \
                f"pool_mask batch/head mismatch: {pool_mask.shape} vs {(b,h_qo)}"

            if not (pool_mask.shape[2] == pooled_q and pool_mask.shape[3] == pooled_k):
                pm = pooled_q - pool_mask.shape[2]
                pn = pooled_k - pool_mask.shape[3]
                if pm < 0 or pn < 0:
                    raise ValueError("pool_mask has larger pooled dims than sequence allows")
                if pm > 0 or pn > 0:
                    pad = torch.ones((b, h_qo, pm, pool_mask.shape[3]), dtype=pool_mask.dtype, device=pool_mask.device)
                    pool_mask = torch.cat([pool_mask, pad], dim=2)
                    if pn > 0:
                        pad2 = torch.ones((b, h_qo, pooled_q, pn), dtype=pool_mask.dtype, device=pool_mask.device)
                        pool_mask = torch.cat([pool_mask, pad2], dim=3)
            stride_pmaskz, stride_pmaskh, stride_pmaskm, stride_pmaskn = pool_mask.stride()
        else:
            # 如果不提供，默认全为 True (不遮挡)
            pool_mask = torch.ones((b, h_qo, pooled_q, pooled_k), dtype=torch.bool, device=q.device)
            stride_pmaskz, stride_pmaskh, stride_pmaskm, stride_pmaskn = pool_mask.stride()

        # ---------------- 2. 处理 Seq Mask ----------------
        if seq_mask is not None:
            seq_mask = seq_mask.expand(b, 1, 1, kv_len).contiguous()
            stride_smaskn = seq_mask.stride(-1)
        else:
            # 注意：Triton 中的 tl.where 判断中，Mask 应该为 boolean，True 表示保留
            seq_mask = torch.ones((b, 1, 1, kv_len), dtype=torch.bool, device=q.device)
            # seq_mask = torch.zeros((b, 1, 1, kv_len), dtype=torch.float16, device=q.device)
            stride_smaskn = seq_mask.stride(-1)

    else:
        raise ValueError(f"tensor_layout {tensor_layout} not supported")

    HEAD_DIM_K = head_dim
    num_kv_groups = h_qo // h_kv

    if return_lse:
        lse = torch.empty([b, h_qo, qo_len], dtype=torch.float32, device=q.device)
    else:
        lse = torch.empty([0], dtype=torch.float32, device='cpu')

    grid = (triton.cdiv(qo_len, BLOCK_M), h_qo, b)
    _attn_fwd[grid](
        q, k, v, qm, k_fp, pool_mask, seq_mask, o, lse,
        stride_bz_q, stride_h_q, stride_seq_q,
        stride_bz_k, stride_h_k, stride_seq_k,
        stride_bz_v, stride_h_v, stride_seq_v,
        stride_bz_o, stride_h_o, stride_seq_o,
        stride_bz_qm, stride_h_qm, stride_seq_qm,
        stride_bz_k_fp, stride_h_k_fp, stride_seq_k_fp,
        # 传入两组 Mask 的 Strides
        stride_pmaskz, stride_pmaskh, stride_pmaskm, stride_pmaskn,
        stride_smaskn,
        qo_len, kv_len,
        h_qo, num_kv_groups,
        sm_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM_K,
        POOL_SIZE=pool_size,
        STAGE=stage, RETURN_LSE=return_lse,
        num_warps=4 if head_dim == 64 else 8,
        num_stages=3 if head_dim == 64 else 2)

    return o, lse

