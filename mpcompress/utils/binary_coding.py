def Unary_encode(x: int):
    bits = "1" * x + "0"
    return bits


def Unary_decode(bits: str):
    x = bits.find("0")
    length = x + 1
    return x, length


def FixedLength_encode(x: int, length: int = 0):
    bits = f"{x:0{length}b}"
    return bits


def FixedLength_decode(bits: str, length: int = 0):
    x = int(bits[:length], 2)
    return x, length


def TruncatedBinary_encode(x: int, n: int):
    length = n.bit_length()
    u = 2**length - n
    if x < u:
        bits = f"{x:0{length-1}b}"
    else:
        bits = f"{x+u:0{length}b}"
    return bits


def TruncatedBinary_decode(bits: str, n: int):
    length = n.bit_length()
    u = 2**length - n
    x = int(bits[: length - 1], 2)
    if x < u:
        return x, length - 1
    else:
        x = int(bits[:length], 2) - u
        return x, length


def TruncatedRice_encode(x: int, k: int = 0):
    prefix = x >> k
    suffix = x - prefix
    bits = Unary_encode(prefix) + FixedLength_encode(suffix, length=k)
    return bits


def TruncatedRice_decode(bits: str, k: int = 0):
    prefix, l1 = Unary_decode(bits)
    suffix, l2 = FixedLength_decode(bits[l1:], k)
    x = prefix << k + suffix
    length = l1 + l2
    return x, length


def ExpGolomb_encode(x: int, k: int = 0):
    # https://zh.wikipedia.org/wiki/指数哥伦布码
    # xx = 2*x+1 if x<0 else 2*x+2
    # bits = Unary_encode(xx >> k) + FixedLength_encode(xx, length=k+1)
    AD = bin(x + 2**k)[2:]
    z = len(AD) - k - 1
    bits = "0" * z + AD
    return bits


def ExpGolomb_decode(Y: str, k: int = 0):
    z = Y.find("1")
    length = z + z + k + 1
    x = int(Y[z:length], 2) - 2**k
    return x, length


def signed_ExpGolomb_encode(x: int, k: int = 0):
    xx = -2 * x - 1 if x < 0 else 2 * x
    return ExpGolomb_encode(xx)


def test_ExpGolomb():
    for k in range(4):
        for x in range(10):
            bits = ExpGolomb_encode(x, k)
            x_, l = ExpGolomb_decode(bits, k)
            print(f"k={k} x={x} bits={bits} {x == x_}")
            assert x == x_

    symbols = [1, 2, 3, 4]
    bits = ""
    for sym in symbols:
        bits += ExpGolomb_encode(sym, 0)
    rec = []
    p = 0
    while p < len(bits):
        sym, l = ExpGolomb_decode(bits[p:], 0)
        rec.append(sym)
        p += l
    print(f"k=0 symbols={symbols} bits={bits} {symbols == rec}")
    assert symbols == rec
