def str_to_bits(s): return ''.join(f'{ord(c):08b}' for c in s)
def bits_to_str(bits): return ''.join([chr(int(bits[i:i+8], 2)) for i in range(0, len(bits), 8)])
