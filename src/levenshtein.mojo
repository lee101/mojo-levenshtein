"""Bit-parallel and weighted edit-distance kernels exposed through a C ABI."""

from std.sys.info import simd_width_of

comptime U32Ptr = UnsafePointer[UInt32, AnyOrigin[mut=True]]
comptime U64Ptr = UnsafePointer[UInt64, AnyOrigin[mut=True]]
comptime I64Ptr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime WAVEFRONT_LIMIT = 384


def trim_prefix_suffix(
    a: U32Ptr, n: Int, b: U32Ptr, m: Int
) -> Tuple[Int, Int, Int]:
    comptime W = simd_width_of[DType.float64]()
    var start = 0
    var limit = min(n, m)
    while start + W <= limit:
        var equal = a.load[width=W, alignment=1](start).eq(
            b.load[width=W, alignment=1](start)
        )
        if equal.cast[DType.uint32]().reduce_add() != UInt32(W):
            break
        start += W
    while start < limit and a[start] == b[start]:
        start += 1
    var an = n
    var bm = m
    while an - W >= start and bm - W >= start:
        var equal = a.load[width=W, alignment=1](an - W).eq(
            b.load[width=W, alignment=1](bm - W)
        )
        if equal.cast[DType.uint32]().reduce_add() != UInt32(W):
            break
        an -= W
        bm -= W
    while an > start and bm > start and a[an - 1] == b[bm - 1]:
        an -= 1
        bm -= 1
    return start, an, bm


def hash_slot(key: UInt32, mask: Int) -> Int:
    return Int((UInt64(key) * UInt64(0x9E3779B185EBCA87)) & UInt64(mask))


def build_masks(
    pattern: U32Ptr,
    start: Int,
    m: Int,
    masks: U64Ptr,
    keys: U32Ptr,
    values: I64Ptr,
    hash_size: Int,
    words: Int,
) -> Int:
    comptime W = simd_width_of[DType.float64]()
    var i = 0
    var empty = SIMD[DType.int64, W](-1)
    while i + W <= hash_size:
        values.store[alignment=8](i, empty)
        i += W
    while i < hash_size:
        values[i] = -1
        i += 1
    var distinct = 0
    for j in range(m):
        var key = pattern[start + j]
        var slot = hash_slot(key, hash_size - 1)
        while values[slot] != -1 and keys[slot] != key:
            slot = (slot + 1) & (hash_size - 1)
        var symbol = 0
        if values[slot] == -1:
            keys[slot] = key
            values[slot] = Int64(distinct)
            symbol = distinct
            distinct += 1
        else:
            symbol = Int(values[slot])
        var word = j >> 6
        var bit = j & 63
        masks[symbol * words + word] |= UInt64(1) << UInt64(bit)
    return distinct


def find_mask(
    key: UInt32, keys: U32Ptr, values: I64Ptr, hash_size: Int
) -> Int:
    var slot = hash_slot(key, hash_size - 1)
    while values[slot] != -1:
        if keys[slot] == key:
            return Int(values[slot])
        slot = (slot + 1) & (hash_size - 1)
    return -1


def wavefront_distance(
    text: U32Ptr,
    tn: Int,
    pattern: U32Ptr,
    pm: Int,
    wave: I64Ptr,
) -> Int:
    comptime W = simd_width_of[DType.float64]()
    comptime SPAN = 2 * WAVEFRONT_LIMIT + 3
    comptime STORAGE = 2 * SPAN
    var negative = SIMD[DType.int64, W](-1)
    var i = 0
    while i + W <= STORAGE:
        wave.store[alignment=8](i, negative)
        i += W
    while i < STORAGE:
        wave[i] = -1
        i += 1

    var previous = wave
    var current = wave + SPAN
    var center = WAVEFRONT_LIMIT + 1
    var x = 0
    var y = 0
    if x < tn and y < pm and text[x] == pattern[y]:
        x += 1
        y += 1
        while x + W <= tn and y + W <= pm:
            var equal = text.load[width=W, alignment=1](x).eq(
                pattern.load[width=W, alignment=1](y)
            )
            if equal.cast[DType.uint32]().reduce_add() != UInt32(W):
                break
            x += W
            y += W
        while x < tn and y < pm and text[x] == pattern[y]:
            x += 1
            y += 1
    if x == tn and y == pm:
        return 0
    previous[center] = Int64(x)

    var limit = min(WAVEFRONT_LIMIT, max(tn, pm))
    for score in range(1, limit + 1):
        for diagonal in range(-score, score + 1):
            var best = -1
            var prior = Int(previous[center + diagonal])
            var prior_y = prior - diagonal
            if prior >= 0 and prior < tn and prior_y < pm:
                best = prior + 1

            prior = Int(previous[center + diagonal - 1])
            if prior >= 0 and prior < tn:
                best = max(best, prior + 1)

            prior = Int(previous[center + diagonal + 1])
            prior_y = prior - diagonal - 1
            if prior >= 0 and prior_y < pm:
                best = max(best, prior)

            if best >= 0:
                x = best
                y = x - diagonal
                if x < tn and y < pm and text[x] == pattern[y]:
                    x += 1
                    y += 1
                    while x + W <= tn and y + W <= pm:
                        var equal = text.load[width=W, alignment=1](x).eq(
                            pattern.load[width=W, alignment=1](y)
                        )
                        if equal.cast[DType.uint32]().reduce_add() != UInt32(W):
                            break
                        x += W
                        y += W
                    while x < tn and y < pm and text[x] == pattern[y]:
                        x += 1
                        y += 1
                best = x
                if x == tn and y == pm:
                    return score
            current[center + diagonal] = Int64(best)
        var temporary = previous
        previous = current
        current = temporary
    return -1


def unit_distance(
    text: U32Ptr,
    n: Int,
    pattern: U32Ptr,
    m: Int,
    masks: U64Ptr,
    keys: U32Ptr,
    values: I64Ptr,
    hash_size: Int,
    work: U64Ptr,
    wavefront_enabled: Bool,
) -> Int:
    var start, text_end, pattern_end = trim_prefix_suffix(text, n, pattern, m)
    var tn = text_end - start
    var pm = pattern_end - start
    if pm == 0:
        return tn
    if tn == 0:
        return pm

    if wavefront_enabled and min(tn, pm) >= 512:
        var wave_result = wavefront_distance(
            text + start,
            tn,
            pattern + start,
            pm,
            work.bitcast[Int64](),
        )
        if wave_result >= 0:
            return wave_result

    comptime W = simd_width_of[DType.float64]()
    var words = (pm + 63) >> 6
    var distinct = build_masks(
        pattern, start, pm, masks, keys, values, hash_size, words
    )
    var pv = work
    var mv = work + words
    var w = 0
    var ones = SIMD[DType.uint64, W](~UInt64(0))
    var zeros = SIMD[DType.uint64, W](0)
    while w + W <= words:
        pv.store[alignment=8](w, ones)
        mv.store[alignment=8](w, zeros)
        w += W
    while w < words:
        pv[w] = ~UInt64(0)
        mv[w] = 0
        w += 1

    var score = pm
    var last_bit = UInt64(1) << UInt64((pm - 1) & 63)
    for i in range(tn):
        var symbol = find_mask(text[start + i], keys, values, hash_size)
        if symbol < 0:
            symbol = distinct
        var add_carry = UInt64(0)
        var ph_carry = UInt64(1)
        var mh_carry = UInt64(0)
        var final_ph = UInt64(0)
        var final_mh = UInt64(0)
        for w in range(words):
            var old_pv = pv[w]
            var old_mv = mv[w]
            var eq = masks[symbol * words + w]
            var xv = eq | old_mv
            var addend = eq & old_pv
            var wide_sum = (
                UInt128(addend) + UInt128(old_pv) + UInt128(add_carry)
            )
            var second = UInt64(wide_sum)
            add_carry = UInt64(wide_sum >> 64)
            var xh = (second ^ old_pv) | eq
            var ph = old_mv | ~(xh | old_pv)
            var mh = old_pv & xh
            final_ph = ph
            final_mh = mh
            var shifted_ph = (ph << 1) | ph_carry
            var shifted_mh = (mh << 1) | mh_carry
            ph_carry = ph >> 63
            mh_carry = mh >> 63
            pv[w] = shifted_mh | ~(xv | shifted_ph)
            mv[w] = shifted_ph & xv
        if (final_ph & last_bit) != 0:
            score += 1
        elif (final_mh & last_bit) != 0:
            score -= 1
    return score


def weighted_distance(
    a: U32Ptr,
    n: Int,
    b: U32Ptr,
    m: Int,
    insertion: Int,
    deletion: Int,
    substitution: Int,
    row: I64Ptr,
) -> Int:
    var start, a_end, b_end = trim_prefix_suffix(a, n, b, m)
    var an = a_end - start
    var bm = b_end - start
    for j in range(bm + 1):
        row[j] = Int64(j * insertion)
    for i in range(1, an + 1):
        var diagonal = row[0]
        row[0] = Int64(i * deletion)
        for j in range(1, bm + 1):
            var above = row[j]
            var replace = diagonal
            if a[start + i - 1] != b[start + j - 1]:
                replace += Int64(substitution)
            var remove = above + Int64(deletion)
            var insert = row[j - 1] + Int64(insertion)
            row[j] = min(replace, min(remove, insert))
            diagonal = above
    return Int(row[bm])


def indel_distance(
    text: U32Ptr,
    n: Int,
    pattern: U32Ptr,
    m: Int,
    masks: U64Ptr,
    keys: U32Ptr,
    values: I64Ptr,
    hash_size: Int,
    state: U64Ptr,
) -> Int:
    var start, text_end, pattern_end = trim_prefix_suffix(text, n, pattern, m)
    var tn = text_end - start
    var pm = pattern_end - start
    if pm == 0:
        return tn
    if tn == 0:
        return pm

    comptime W = simd_width_of[DType.float64]()
    var words = (pm + 63) >> 6
    var distinct = build_masks(
        pattern, start, pm, masks, keys, values, hash_size, words
    )
    var w = 0
    var zeros = SIMD[DType.uint64, W](0)
    while w + W <= words:
        state.store[alignment=8](w, zeros)
        w += W
    while w < words:
        state[w] = 0
        w += 1

    for i in range(tn):
        var symbol = find_mask(text[start + i], keys, values, hash_size)
        if symbol < 0:
            symbol = distinct
        var shift_carry = UInt64(1)
        var borrow = UInt64(0)
        for w in range(words):
            var old_state = state[w]
            var matches = masks[symbol * words + w]
            var x = matches | old_state
            var y = (old_state << 1) | shift_carry
            shift_carry = old_state >> 63
            var first = x - y
            var borrow1 = x < y
            var difference = first - borrow
            var borrow2 = first < borrow
            borrow = 0
            if borrow1 or borrow2:
                borrow = 1
            state[w] = x & ~difference

    var lcs = 0
    for w in range(words):
        var bits = state[w]
        while bits != 0:
            bits &= bits - 1
            lcs += 1
    return tn + pm - 2 * lcs


@export("ml_distance_unit_u32")
def ml_distance_unit_u32(
    a_addr: Int,
    n: Int,
    b_addr: Int,
    m: Int,
    masks_addr: Int,
    keys_addr: Int,
    values_addr: Int,
    hash_size: Int,
    work_addr: Int,
    wavefront_enabled: Int,
) abi("C") -> Int:
    if n == 0:
        return m
    if m == 0:
        return n
    return unit_distance(
        U32Ptr(unsafe_from_address=a_addr),
        n,
        U32Ptr(unsafe_from_address=b_addr),
        m,
        U64Ptr(unsafe_from_address=masks_addr),
        U32Ptr(unsafe_from_address=keys_addr),
        I64Ptr(unsafe_from_address=values_addr),
        hash_size,
        U64Ptr(unsafe_from_address=work_addr),
        wavefront_enabled != 0,
    )


@export("ml_distance_weighted_u32")
def ml_distance_weighted_u32(
    a_addr: Int,
    n: Int,
    b_addr: Int,
    m: Int,
    insertion: Int,
    deletion: Int,
    substitution: Int,
    row_addr: Int,
) abi("C") -> Int:
    if n == 0:
        return m * insertion
    if m == 0:
        return n * deletion
    return weighted_distance(
        U32Ptr(unsafe_from_address=a_addr),
        n,
        U32Ptr(unsafe_from_address=b_addr),
        m,
        insertion,
        deletion,
        substitution,
        I64Ptr(unsafe_from_address=row_addr),
    )


@export("ml_indel_distance_u32")
def ml_indel_distance_u32(
    a_addr: Int,
    n: Int,
    b_addr: Int,
    m: Int,
    masks_addr: Int,
    keys_addr: Int,
    values_addr: Int,
    hash_size: Int,
    state_addr: Int,
) abi("C") -> Int:
    if n == 0:
        return m
    if m == 0:
        return n
    return indel_distance(
        U32Ptr(unsafe_from_address=a_addr),
        n,
        U32Ptr(unsafe_from_address=b_addr),
        m,
        U64Ptr(unsafe_from_address=masks_addr),
        U32Ptr(unsafe_from_address=keys_addr),
        I64Ptr(unsafe_from_address=values_addr),
        hash_size,
        U64Ptr(unsafe_from_address=state_addr),
    )
