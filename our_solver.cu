// =============================================================================
// our_solver.cu — self-contained CUDA brute-force solver for Arweave puzzles.
//
// Algorithm (verified byte-for-byte against the puzzle page's CryptoJS code):
//   passphrase = hex( SHA512^11513(password) )
//   key, iv    = EvpKDF(passphrase, salt)   [MD5, 10000 iters, 128B key + 16B iv]
//   plaintext  = AES-1024-CBC decrypt       [Nk=32, 38 rounds, 128-byte key]
//   match      = plaintext contains the solution tag, e.g. "\"kty\":\"RSA\""
//
// All crypto is __host__ __device__: the SAME code runs on CPU (self-test gate)
// and GPU (brute-force kernel). The wallet JSON starts with the tag, so only the
// FIRST 2 CBC blocks are decrypted for the fast tag check.
//
// Usage:
//   ./our_solver --msg_file=msg.b64 --keys_file=keys.txt [--solution_tag='"kty":"RSA"']
//                [--block=256] [--grid=128]
//   ./our_solver --msg_file=msg.b64 --selftest-pw=RasputinWilhelmAlekhine
//                [--solution_tag='"kty":"RSA"']     <- CPU self-test (exit 0 if ok)
//
// keys.txt: one line per position, comma-separated words (concatenated in order).
// =============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define MAX_KEYS       16
#define MAX_WORDS      64
#define MAX_WORD_LEN   16
#define MAX_PHRASE     128
#define MAX_MSG        8192
#define SHA512_DIGEST_SIZE 64
#define AES_BLOCKLEN   16

// ---------------------------------------------------------------------------
// SHA-512 (verified against hashlib / CryptoJS)
// ---------------------------------------------------------------------------
typedef unsigned char BYTE;
typedef unsigned long long QWORD;

typedef struct {
    BYTE data[128];
    QWORD datalen;
    QWORD bitlen;
    QWORD state[8];
} SHA512_CTX;

// static const (NOT __constant__): readable from both host (self-test) and device.
// nvcc places file-scope const globals in device-accessible memory.
static const QWORD sha512_k[80] = {
    0x428a2f98d728ae22ULL,0x7137449123ef65cdULL,0xb5c0fbcfec4d3b2fULL,0xe9b5dba58189dbbcULL,0x3956c25bf348b538ULL,
    0x59f111f1b605d019ULL,0x923f82a4af194f9bULL,0xab1c5ed5da6d8118ULL,0xd807aa98a3030242ULL,0x12835b0145706fbeULL,
    0x243185be4ee4b28cULL,0x550c7dc3d5ffb4e2ULL,0x72be5d74f27b896fULL,0x80deb1fe3b1696b1ULL,0x9bdc06a725c71235ULL,
    0xc19bf174cf692694ULL,0xe49b69c19ef14ad2ULL,0xefbe4786384f25e3ULL,0x0fc19dc68b8cd5b5ULL,0x240ca1cc77ac9c65ULL,
    0x2de92c6f592b0275ULL,0x4a7484aa6ea6e483ULL,0x5cb0a9dcbd41fbd4ULL,0x76f988da831153b5ULL,0x983e5152ee66dfabULL,
    0xa831c66d2db43210ULL,0xb00327c898fb213fULL,0xbf597fc7beef0ee4ULL,0xc6e00bf33da88fc2ULL,0xd5a79147930aa725ULL,
    0x06ca6351e003826fULL,0x142929670a0e6e70ULL,0x27b70a8546d22ffcULL,0x2e1b21385c26c926ULL,0x4d2c6dfc5ac42aedULL,
    0x53380d139d95b3dfULL,0x650a73548baf63deULL,0x766a0abb3c77b2a8ULL,0x81c2c92e47edaee6ULL,0x92722c851482353bULL,
    0xa2bfe8a14cf10364ULL,0xa81a664bbc423001ULL,0xc24b8b70d0f89791ULL,0xc76c51a30654be30ULL,0xd192e819d6ef5218ULL,
    0xd69906245565a910ULL,0xf40e35855771202aULL,0x106aa07032bbd1b8ULL,0x19a4c116b8d2d0c8ULL,0x1e376c085141ab53ULL,
    0x2748774cdf8eeb99ULL,0x34b0bcb5e19b48a8ULL,0x391c0cb3c5c95a63ULL,0x4ed8aa4ae3418acbULL,0x5b9cca4f7763e373ULL,
    0x682e6ff3d6b2b8a3ULL,0x748f82ee5defb2fcULL,0x78a5636f43172f60ULL,0x84c87814a1f0ab72ULL,0x8cc702081a6439ecULL,
    0x90befffa23631e28ULL,0xa4506cebde82bde9ULL,0xbef9a3f7b2c67915ULL,0xc67178f2e372532bULL,0xca273eceea26619cULL,
    0xd186b8c721c0c207ULL,0xeada7dd6cde0eb1eULL,0xf57d4f7fee6ed178ULL,0x06f067aa72176fbaULL,0x0a637dc5a2c898a6ULL,
    0x113f9804bef90daeULL,0x1b710b35131c471bULL,0x28db77f523047d84ULL,0x32caab7b40c72493ULL,0x3c9ebe0a15c9bebcULL,
    0x431d67c49c100d4cULL,0x4cc5d4becb3e42b6ULL,0x597f299cfc657e2aULL,0x5fcb6fab3ad6faecULL,0x6c44198c4a475817ULL
};

__host__ __device__ inline void sha512_init(SHA512_CTX* ctx) {
    ctx->datalen = 0; ctx->bitlen = 0;
    ctx->state[0] = 0x6a09e667f3bcc908ULL; ctx->state[1] = 0xbb67ae8584caa73bULL;
    ctx->state[2] = 0x3c6ef372fe94f82bULL; ctx->state[3] = 0xa54ff53a5f1d36f1ULL;
    ctx->state[4] = 0x510e527fade682d1ULL; ctx->state[5] = 0x9b05688c2b3e6c1fULL;
    ctx->state[6] = 0x1f83d9abfb41bd6bULL; ctx->state[7] = 0x5be0cd19137e2179ULL;
}

__host__ __device__ inline void sha512_transform(SHA512_CTX* ctx, const BYTE data[]) {
    QWORD a, b, c, d, e, f, g, h, i, t1, t2, m[80];
    for (i = 0; i < 16; i++)
        m[i] = ((QWORD)data[i*8]<<56)|((QWORD)data[i*8+1]<<48)|((QWORD)data[i*8+2]<<40)|((QWORD)data[i*8+3]<<32)|
               ((QWORD)data[i*8+4]<<24)|((QWORD)data[i*8+5]<<16)|((QWORD)data[i*8+6]<<8)|((QWORD)data[i*8+7]);
    for (i = 16; i < 80; i++)
        m[i] = (((m[i-2] >> 19) | (m[i-2] << 45)) ^ ((m[i-2] >> 61) | (m[i-2] << 3)) ^ (m[i-2] >> 6)) +
               m[i-7] + (((m[i-15] >> 1) | (m[i-15] << 63)) ^ ((m[i-15] >> 8) | (m[i-15] << 56)) ^ (m[i-15] >> 7)) + m[i-16];
    a = ctx->state[0]; b = ctx->state[1]; c = ctx->state[2]; d = ctx->state[3];
    e = ctx->state[4]; f = ctx->state[5]; g = ctx->state[6]; h = ctx->state[7];
    for (i = 0; i < 80; i++) {
        t1 = h + (((e >> 14) | (e << 50)) ^ ((e >> 18) | (e << 46)) ^ ((e >> 41) | (e << 23))) +
             (((e) & (f)) ^ (~(e) & (g))) + sha512_k[i] + m[i];
        t2 = (((a >> 28) | (a << 36)) ^ ((a >> 34) | (a << 30)) ^ ((a >> 39) | (a << 25))) +
             (((a) & (b)) ^ ((a) & (c)) ^ ((b) & (c)));
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    ctx->state[0] += a; ctx->state[1] += b; ctx->state[2] += c; ctx->state[3] += d;
    ctx->state[4] += e; ctx->state[5] += f; ctx->state[6] += g; ctx->state[7] += h;
}

__host__ __device__ inline void sha512_update(SHA512_CTX* ctx, const BYTE data[], size_t len) {
    for (size_t i = 0; i < len; i++) {
        ctx->data[ctx->datalen++] = data[i];
        if (ctx->datalen == 128) {
            sha512_transform(ctx, ctx->data);
            ctx->bitlen += 1024;
            ctx->datalen = 0;
        }
    }
}

__host__ __device__ inline void sha512_final(SHA512_CTX* ctx, BYTE hash[]) {
    QWORD i = ctx->datalen;
    ctx->bitlen += ctx->datalen * 8;
    ctx->data[i++] = 0x80;
    if (ctx->datalen > 111) {
        while (i < 128) ctx->data[i++] = 0x00;
        sha512_transform(ctx, ctx->data);
        i = 0;
    }
    while (i < 112) ctx->data[i++] = 0x00;
    // zero bytes 112..119 (the 64-bit length field is 120..127)
    ctx->data[112] = 0; ctx->data[113] = 0; ctx->data[114] = 0; ctx->data[115] = 0;
    ctx->data[116] = 0; ctx->data[117] = 0; ctx->data[118] = 0; ctx->data[119] = 0;
    ctx->data[120] = (ctx->bitlen >> 56) & 0xFF;
    ctx->data[121] = (ctx->bitlen >> 48) & 0xFF;
    ctx->data[122] = (ctx->bitlen >> 40) & 0xFF;
    ctx->data[123] = (ctx->bitlen >> 32) & 0xFF;
    ctx->data[124] = (ctx->bitlen >> 24) & 0xFF;
    ctx->data[125] = (ctx->bitlen >> 16) & 0xFF;
    ctx->data[126] = (ctx->bitlen >> 8) & 0xFF;
    ctx->data[127] = (ctx->bitlen >> 0) & 0xFF;
    sha512_transform(ctx, ctx->data);
    for (i = 0; i < 8; i++) {
        hash[i*8+0] = (ctx->state[i] >> 56) & 0xff; hash[i*8+1] = (ctx->state[i] >> 48) & 0xff;
        hash[i*8+2] = (ctx->state[i] >> 40) & 0xff; hash[i*8+3] = (ctx->state[i] >> 32) & 0xff;
        hash[i*8+4] = (ctx->state[i] >> 24) & 0xff; hash[i*8+5] = (ctx->state[i] >> 16) & 0xff;
        hash[i*8+6] = (ctx->state[i] >> 8) & 0xff;  hash[i*8+7] = (ctx->state[i] >> 0) & 0xff;
    }
}

// SHA512^iterations over raw 64-byte digests, hex output at the end (page-exact)
__host__ __device__ inline void iterative_sha512(const uint8_t* input, int length, char* hex_output, int iterations) {
    SHA512_CTX ctx;
    BYTE buffer[64];
    sha512_init(&ctx);
    sha512_update(&ctx, input, length);
    sha512_final(&ctx, buffer);
    for (int i = 1; i < iterations; i++) {
        sha512_init(&ctx);
        sha512_update(&ctx, buffer, 64);
        sha512_final(&ctx, buffer);
    }
    for (int i = 0; i < 64; i++) {
        unsigned char c = buffer[i];
        unsigned char hi = (c >> 4) & 0xF, lo = c & 0xF;
        hex_output[2*i]   = (hi < 10) ? ('0' + hi) : ('a' + hi - 10);
        hex_output[2*i+1] = (lo < 10) ? ('0' + lo) : ('a' + lo - 10);
    }
}

// ---------------------------------------------------------------------------
// MD5 (verified against hashlib)
// ---------------------------------------------------------------------------
#define ROTL(x, n) (((x) << (n)) | ((x) >> (32-(n))))
#define FF(a,b,c,d,x,s,ac) {(a) += (((b)&(c))|((~(b))&(d))) + (x) + (uint32_t)(ac); (a) = ROTL((a),(s)); (a) += (b);}
#define GG(a,b,c,d,x,s,ac) {(a) += (((b)&(d))|((c)&(~(d)))) + (x) + (uint32_t)(ac); (a) = ROTL((a),(s)); (a) += (b);}
#define HH(a,b,c,d,x,s,ac) {(a) += ((b)^(c)^(d)) + (x) + (uint32_t)(ac); (a) = ROTL((a),(s)); (a) += (b);}
#define II(a,b,c,d,x,s,ac) {(a) += ((c)^((b)|(~(d)))) + (x) + (uint32_t)(ac); (a) = ROTL((a),(s)); (a) += (b);}

__host__ __device__ inline void md5_transform_fast(uint32_t* state, const uint32_t* M) {
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    FF(a,b,c,d, M[ 0], 7, 0xd76aa478); FF(d,a,b,c, M[ 1], 12, 0xe8c7b756); FF(c,d,a,b, M[ 2], 17, 0x242070db); FF(b,c,d,a, M[ 3], 22, 0xc1bdceee);
    FF(a,b,c,d, M[ 4], 7, 0xf57c0faf); FF(d,a,b,c, M[ 5], 12, 0x4787c62a); FF(c,d,a,b, M[ 6], 17, 0xa8304613); FF(b,c,d,a, M[ 7], 22, 0xfd469501);
    FF(a,b,c,d, M[ 8], 7, 0x698098d8); FF(d,a,b,c, M[ 9], 12, 0x8b44f7af); FF(c,d,a,b, M[10], 17, 0xffff5bb1); FF(b,c,d,a, M[11], 22, 0x895cd7be);
    FF(a,b,c,d, M[12], 7, 0x6b901122); FF(d,a,b,c, M[13], 12, 0xfd987193); FF(c,d,a,b, M[14], 17, 0xa679438e); FF(b,c,d,a, M[15], 22, 0x49b40821);
    GG(a,b,c,d, M[ 1], 5, 0xf61e2562); GG(d,a,b,c, M[ 6], 9, 0xc040b340); GG(c,d,a,b, M[11], 14, 0x265e5a51); GG(b,c,d,a, M[ 0], 20, 0xe9b6c7aa);
    GG(a,b,c,d, M[ 5], 5, 0xd62f105d); GG(d,a,b,c, M[10], 9, 0x02441453); GG(c,d,a,b, M[15], 14, 0xd8a1e681); GG(b,c,d,a, M[ 4], 20, 0xe7d3fbc8);
    GG(a,b,c,d, M[ 9], 5, 0x21e1cde6); GG(d,a,b,c, M[14], 9, 0xc33707d6); GG(c,d,a,b, M[ 3], 14, 0xf4d50d87); GG(b,c,d,a, M[ 8], 20, 0x455a14ed);
    GG(a,b,c,d, M[13], 5, 0xa9e3e905); GG(d,a,b,c, M[ 2], 9, 0xfcefa3f8); GG(c,d,a,b, M[ 7], 14, 0x676f02d9); GG(b,c,d,a, M[12], 20, 0x8d2a4c8a);
    HH(a,b,c,d, M[ 5], 4, 0xfffa3942); HH(d,a,b,c, M[ 8], 11, 0x8771f681); HH(c,d,a,b, M[11], 16, 0x6d9d6122); HH(b,c,d,a, M[14], 23, 0xfde5380c);
    HH(a,b,c,d, M[ 1], 4, 0xa4beea44); HH(d,a,b,c, M[ 4], 11, 0x4bdecfa9); HH(c,d,a,b, M[ 7], 16, 0xf6bb4b60); HH(b,c,d,a, M[10], 23, 0xbebfbc70);
    HH(a,b,c,d, M[13], 4, 0x289b7ec6); HH(d,a,b,c, M[ 0], 11, 0xeaa127fa); HH(c,d,a,b, M[ 3], 16, 0xd4ef3085); HH(b,c,d,a, M[ 6], 23, 0x04881d05);
    HH(a,b,c,d, M[ 9], 4, 0xd9d4d039); HH(d,a,b,c, M[12], 11, 0xe6db99e5); HH(c,d,a,b, M[15], 16, 0x1fa27cf8); HH(b,c,d,a, M[ 2], 23, 0xc4ac5665);
    II(a,b,c,d, M[ 0], 6, 0xf4292244); II(d,a,b,c, M[ 7], 10, 0x432aff97); II(c,d,a,b, M[14], 15, 0xab9423a7); II(b,c,d,a, M[ 5], 21, 0xfc93a039);
    II(a,b,c,d, M[12], 6, 0x655b59c3); II(d,a,b,c, M[ 3], 10, 0x8f0ccc92); II(c,d,a,b, M[10], 15, 0xffeff47d); II(b,c,d,a, M[ 1], 21, 0x85845dd1);
    II(a,b,c,d, M[ 8], 6, 0x6fa87e4f); II(d,a,b,c, M[15], 10, 0xfe2ce6e0); II(c,d,a,b, M[ 6], 15, 0xa3014314); II(b,c,d,a, M[13], 21, 0x4e0811a1);
    II(a,b,c,d, M[ 4], 6, 0xf7537e82); II(d,a,b,c, M[11], 10, 0xbd3af235); II(c,d,a,b, M[ 2], 15, 0x2ad7d2bb); II(b,c,d,a, M[ 9], 21, 0xeb86d391);
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
}

__host__ __device__ inline void md5_cuda(const uint8_t* input, size_t len, uint8_t* output) {
    uint32_t state[4] = {0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476};
    uint8_t buf[512];
    uint32_t M[16];
    size_t padded_len = len + 1;
    while (padded_len % 64 != 56) padded_len++;
    for (size_t i = 0; i < len; i++) buf[i] = input[i];
    buf[len] = 0x80;
    for (size_t i = len + 1; i < padded_len; i++) buf[i] = 0;
    uint64_t bits = len * 8;
    for (int i = 0; i < 8; i++) buf[padded_len + i] = (bits >> (i*8)) & 0xFF;
    size_t total_len = padded_len + 8;
    for (size_t off = 0; off < total_len; off += 64) {
        const uint8_t* bs = buf + off;
        for (int j = 0; j < 16; j++)
            M[j] = ((uint32_t)bs[j*4]) | ((uint32_t)bs[j*4+1]<<8) | ((uint32_t)bs[j*4+2]<<16) | ((uint32_t)bs[j*4+3]<<24);
        md5_transform_fast(state, M);
    }
    for (int i = 0; i < 4; i++) {
        output[i*4+0] = (state[i] >> 0) & 0xff;
        output[i*4+1] = (state[i] >> 8) & 0xff;
        output[i*4+2] = (state[i] >> 16) & 0xff;
        output[i*4+3] = (state[i] >> 24) & 0xff;
    }
}

// ---------------------------------------------------------------------------
// AES-1024 (Nk=32, 38 rounds) — verified against CryptoJS
// ---------------------------------------------------------------------------
static const uint8_t aes_sbox[256] = {
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16};
static const uint8_t aes_rsbox[256] = {
    0x52,0x09,0x6a,0xd5,0x30,0x36,0xa5,0x38,0xbf,0x40,0xa3,0x9e,0x81,0xf3,0xd7,0xfb,
    0x7c,0xe3,0x39,0x82,0x9b,0x2f,0xff,0x87,0x34,0x8e,0x43,0x44,0xc4,0xde,0xe9,0xcb,
    0x54,0x7b,0x94,0x32,0xa6,0xc2,0x23,0x3d,0xee,0x4c,0x95,0x0b,0x42,0xfa,0xc3,0x4e,
    0x08,0x2e,0xa1,0x66,0x28,0xd9,0x24,0xb2,0x76,0x5b,0xa2,0x49,0x6d,0x8b,0xd1,0x25,
    0x72,0xf8,0xf6,0x64,0x86,0x68,0x98,0x16,0xd4,0xa4,0x5c,0xcc,0x5d,0x65,0xb6,0x92,
    0x6c,0x70,0x48,0x50,0xfd,0xed,0xb9,0xda,0x5e,0x15,0x46,0x57,0xa7,0x8d,0x9d,0x84,
    0x90,0xd8,0xab,0x00,0x8c,0xbc,0xd3,0x0a,0xf7,0xe4,0x58,0x05,0xb8,0xb3,0x45,0x06,
    0xd0,0x2c,0x1e,0x8f,0xca,0x3f,0x0f,0x02,0xc1,0xaf,0xbd,0x03,0x01,0x13,0x8a,0x6b,
    0x3a,0x91,0x11,0x41,0x4f,0x67,0xdc,0xea,0x97,0xf2,0xcf,0xce,0xf0,0xb4,0xe6,0x73,
    0x96,0xac,0x74,0x22,0xe7,0xad,0x35,0x85,0xe2,0xf9,0x37,0xe8,0x1c,0x75,0xdf,0x6e,
    0x47,0xf1,0x1a,0x71,0x1d,0x29,0xc5,0x89,0x6f,0xb7,0x62,0x0e,0xaa,0x18,0xbe,0x1b,
    0xfc,0x56,0x3e,0x4b,0xc6,0xd2,0x79,0x20,0x9a,0xdb,0xc0,0xfe,0x78,0xcd,0x5a,0xf4,
    0x1f,0xdd,0xa8,0x33,0x88,0x07,0xc7,0x31,0xb1,0x12,0x10,0x59,0x27,0x80,0xec,0x5f,
    0x60,0x51,0x7f,0xa9,0x19,0xb5,0x4a,0x0d,0x2d,0xe5,0x7a,0x9f,0x93,0xc9,0x9c,0xef,
    0xa0,0xe0,0x3b,0x4d,0xae,0x2a,0xf5,0xb0,0xc8,0xeb,0xbb,0x3c,0x83,0x53,0x99,0x61,
    0x17,0x2b,0x04,0x7e,0xba,0x77,0xd6,0x26,0xe1,0x69,0x14,0x63,0x55,0x21,0x0c,0x7d};
static const uint8_t aes_Rcon[255] = {
    0x8d,0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1b,0x36,0x6c,0xd8,0xab,0x4d,0x9a,
    0x2f,0x5e,0xbc,0x63,0xc6,0x97,0x35,0x6a,0xd4,0xb3,0x7d,0xfa,0xef,0xc5,0x91,0x39,
    0x72,0xe4,0xd3,0xbd,0x61,0xc2,0x9f,0x25,0x4a,0x94,0x33,0x66,0xcc,0x83,0x1d,0x3a,
    0x74,0xe8,0xcb,0x8d,0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1b,0x36,0x6c,0xd8};

__host__ __device__ inline uint8_t xtime(uint8_t x) { return ((x << 1) ^ (((x >> 7) & 1) * 0x1b)); }
__host__ __device__ inline uint8_t mul_9(uint8_t x)  { return xtime(xtime(xtime(x))) ^ x; }
__host__ __device__ inline uint8_t mul_11(uint8_t x) { return xtime(xtime(xtime(x))) ^ xtime(x) ^ x; }
__host__ __device__ inline uint8_t mul_13(uint8_t x) { return xtime(xtime(xtime(x))) ^ xtime(xtime(x)) ^ x; }
__host__ __device__ inline uint8_t mul_14(uint8_t x) { return xtime(xtime(xtime(x))) ^ xtime(xtime(x)) ^ xtime(x); }

__host__ __device__ inline void InvMixColumns(uint8_t* s) {
    for (int i = 0; i < 4; i++) {
        uint8_t a = s[i*4+0], b = s[i*4+1], c = s[i*4+2], d = s[i*4+3];
        s[i*4+0] = mul_14(a) ^ mul_11(b) ^ mul_13(c) ^ mul_9(d);
        s[i*4+1] = mul_9(a) ^ mul_14(b) ^ mul_11(c) ^ mul_13(d);
        s[i*4+2] = mul_13(a) ^ mul_9(b) ^ mul_14(c) ^ mul_11(d);
        s[i*4+3] = mul_11(a) ^ mul_13(b) ^ mul_9(c) ^ mul_14(d);
    }
}
__host__ __device__ inline void InvSubBytes(uint8_t* s) { for (int i = 0; i < 16; i++) s[i] = aes_rsbox[s[i]]; }
__host__ __device__ inline void InvShiftRows(uint8_t* s) {
    uint8_t t;
    t = s[13]; s[13] = s[9]; s[9] = s[5]; s[5] = s[1]; s[1] = t;
    t = s[2]; s[2] = s[10]; s[10] = t; t = s[6]; s[6] = s[14]; s[14] = t;
    t = s[3]; s[3] = s[7]; s[7] = s[11]; s[11] = s[15]; s[15] = t;
}
__host__ __device__ inline void AddRoundKey(uint8_t round, uint8_t* s, const uint8_t* rk) {
    for (int i = 0; i < 16; i++) s[i] ^= rk[round*16 + i];
}
__host__ __device__ inline void InvCipher(uint8_t* s, const uint8_t* rk, uint8_t rounds) {
    AddRoundKey(rounds, s, rk);
    for (int8_t round = rounds - 1; ; round--) {
        InvShiftRows(s);
        InvSubBytes(s);
        AddRoundKey(round, s, rk);
        if (round == 0) break;
        InvMixColumns(s);
    }
}
__host__ __device__ inline void KeyExpansion(uint8_t* rk, const uint8_t* key, uint8_t rounds, uint8_t Nk) {
    unsigned i, ksWords = 4 * (rounds + 1);
    uint32_t w[256];
    for (i = 0; i < Nk; i++)
        w[i] = ((uint32_t)key[4*i]<<24) | ((uint32_t)key[4*i+1]<<16) | ((uint32_t)key[4*i+2]<<8) | ((uint32_t)key[4*i+3]);
    for (i = Nk; i < ksWords; i++) {
        uint32_t temp = w[i-1];
        if (i % Nk == 0) {
            temp = (temp << 8) | (temp >> 24);
            temp = ((uint32_t)aes_sbox[(temp>>24)&0xff]<<24) | ((uint32_t)aes_sbox[(temp>>16)&0xff]<<16) |
                   ((uint32_t)aes_sbox[(temp>>8)&0xff]<<8) | ((uint32_t)aes_sbox[temp&0xff]);
            temp ^= ((uint32_t)aes_Rcon[i/Nk]) << 24;
        } else if (Nk > 6 && (i % Nk) == 4) {
            temp = ((uint32_t)aes_sbox[(temp>>24)&0xff]<<24) | ((uint32_t)aes_sbox[(temp>>16)&0xff]<<16) |
                   ((uint32_t)aes_sbox[(temp>>8)&0xff]<<8) | ((uint32_t)aes_sbox[temp&0xff]);
        }
        w[i] = w[i-Nk] ^ temp;
    }
    for (i = 0; i < ksWords; i++) {
        rk[4*i+0] = (w[i] >> 24) & 0xff;
        rk[4*i+1] = (w[i] >> 16) & 0xff;
        rk[4*i+2] = (w[i] >> 8) & 0xff;
        rk[4*i+3] = w[i] & 0xff;
    }
}

// ---------------------------------------------------------------------------
// Config & kernel
// ---------------------------------------------------------------------------
typedef struct {
    uint8_t  keys_num;
    uint16_t words_num[MAX_KEYS];
    uint16_t words_len[MAX_KEYS][MAX_WORDS];
    uint8_t  words_pool[MAX_KEYS][MAX_WORDS][MAX_WORD_LEN];
    uint8_t  salt[8];
    uint8_t  ciphertext[MAX_MSG];
    uint32_t ciphertext_size;
    uint8_t  tag[16];
    uint8_t  tag_len;
} BruteConfig;

typedef struct {
    uint32_t found_flag;
    uint8_t  phrase[MAX_PHRASE];
    uint32_t phrase_len;
    uint8_t  wallet[MAX_MSG];
    uint32_t wallet_len;
} BruteResult;

__host__ __device__ uint16_t index_to_phrase(uint64_t index, const BruteConfig* cfg,
                                             const uint64_t* digitsWeights, uint8_t* phrase) {
    int16_t indices[MAX_KEYS];
    for (int i = 0; i < cfg->keys_num - 1; i++) {
        indices[i] = index / digitsWeights[i];
        index -= (uint64_t)indices[i] * digitsWeights[i];
    }
    indices[cfg->keys_num - 1] = (int16_t)index;
    uint16_t plen = 0;
    for (int i = 0; i < cfg->keys_num; i++)
        for (int j = 0; j < cfg->words_len[i][indices[i]]; j++)
            phrase[plen++] = cfg->words_pool[i][indices[i]][j];
    return plen;
}

__host__ __device__ void derive_key_iv(const uint8_t* pw, uint16_t pw_len, const BruteConfig* cfg,
                                       uint8_t* kdf_key_iv) {
    char hex[129];
    iterative_sha512(pw, pw_len, hex, 11513);
    int kdf_len = 0;
    uint8_t md5_prev[16] = {0};
    bool has_prev = false;
    while (kdf_len < 128 + 16) {
        uint8_t block[160];
        int blen = 0;
        if (has_prev) for (int i = 0; i < 16; i++) block[blen++] = md5_prev[i];
        for (int i = 0; i < 128; i++) block[blen++] = (uint8_t)hex[i];
        for (int i = 0; i < 8; i++) block[blen++] = cfg->salt[i];
        md5_cuda(block, blen, md5_prev);
        for (int i = 1; i < 10000; i++) md5_cuda(md5_prev, 16, md5_prev);
        int copy_len = 16;
        if (kdf_len + copy_len > 144) copy_len = 144 - kdf_len;
        for (int i = 0; i < copy_len; i++) kdf_key_iv[kdf_len + i] = md5_prev[i];
        kdf_len += copy_len;
        has_prev = true;
    }
}

__host__ __device__ void aes_cbc_decrypt(const BruteConfig* cfg, const uint8_t* key_iv,
                                         uint8_t* out, uint32_t nbytes) {
    uint8_t round_key[4 * (38+1) * 16];
    uint8_t local_iv[16];
    KeyExpansion(round_key, key_iv, 38, 32);
    for (int i = 0; i < 16; i++) local_iv[i] = key_iv[128 + i];
    for (uint32_t i = 0; i < nbytes; i++) out[i] = cfg->ciphertext[i];
    for (uint32_t i = 0; i + 16 <= nbytes; i += 16) {
        uint8_t next_iv[16];
        for (int j = 0; j < 16; j++) next_iv[j] = out[i+j];
        InvCipher(out + i, round_key, 38);
        for (int j = 0; j < 16; j++) out[i+j] ^= local_iv[j];
        for (int j = 0; j < 16; j++) local_iv[j] = next_iv[j];
    }
}

// fast tag check: decrypt only the first 2 CBC blocks
__host__ __device__ uint8_t verify_candidate(const uint8_t* pw, uint16_t pw_len, const BruteConfig* cfg) {
    uint8_t kdf_key_iv[144];
    derive_key_iv(pw, pw_len, cfg, kdf_key_iv);

    uint8_t round_key[4 * (38+1) * 16];
    uint8_t local_iv[16];
    KeyExpansion(round_key, kdf_key_iv, 38, 32);
    for (int i = 0; i < 16; i++) local_iv[i] = kdf_key_iv[128 + i];

    uint8_t buf[32];
    uint32_t nblocks = cfg->ciphertext_size / 16;
    if (nblocks > 2) nblocks = 2;
    for (uint32_t i = 0; i < nblocks; i++) {
        uint8_t ct_block[16], next_iv[16];
        for (int j = 0; j < 16; j++) { ct_block[j] = cfg->ciphertext[i*16+j]; next_iv[j] = ct_block[j]; }
        InvCipher(ct_block, round_key, 38);
        for (int j = 0; j < 16; j++) ct_block[j] ^= local_iv[j];
        for (int j = 0; j < 16; j++) { buf[i*16+j] = ct_block[j]; local_iv[j] = next_iv[j]; }
    }

    uint32_t search_len = nblocks * 16;
    for (uint32_t i = 0; i + cfg->tag_len <= search_len; i++) {
        bool match = true;
        for (uint8_t j = 0; j < cfg->tag_len; j++)
            if (buf[i+j] != cfg->tag[j]) { match = false; break; }
        if (match) return 1;
    }
    return 0;
}

#ifndef SOLVER_HOST_ONLY
__global__ void brute_kernel(const BruteConfig* cfg, const uint64_t index_begin,
                             const uint64_t index_end, BruteResult* ret) {
    uint64_t idx = index_begin + (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= index_end || ret->found_flag) return;

    uint64_t digitsWeights[MAX_KEYS];
    for (int i1 = 1; i1 < cfg->keys_num; i1++) {
        digitsWeights[i1-1] = 1;
        for (int i2 = i1; i2 < cfg->keys_num; i2++)
            digitsWeights[i1-1] *= cfg->words_num[i2];
    }

    uint8_t phrase[MAX_PHRASE];
    uint16_t plen = index_to_phrase(idx, cfg, digitsWeights, phrase);
    if (verify_candidate(phrase, plen, cfg)) {
        ret->found_flag = 1;
        ret->phrase_len = plen;
        for (uint16_t i = 0; i < plen; i++) ret->phrase[i] = phrase[i];
        // full decrypt for the wallet
        uint8_t key_iv[144];
        derive_key_iv(phrase, plen, cfg, key_iv);
        aes_cbc_decrypt(cfg, key_iv, ret->wallet, cfg->ciphertext_size);
        ret->wallet_len = cfg->ciphertext_size;
    }
}

#endif // !SOLVER_HOST_ONLY

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------
static void check_cuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(e));
        exit(1);
    }
}

static int b64decode(const char* in, uint8_t* out) {
    static int8_t tbl[256];
    static int init = 0;
    if (!init) {
        for (int i = 0; i < 256; i++) tbl[i] = -1;
        for (int i = 0; i < 26; i++) { tbl['A'+i] = i; tbl['a'+i] = 26+i; }
        for (int i = 0; i < 10; i++) tbl['0'+i] = 52+i;
        tbl['+'] = 62; tbl['/'] = 63;
        init = 1;
    }
    int len = strlen(in);
    int olen = 0;
    uint32_t acc = 0;
    int nbits = 0;
    for (int i = 0; i < len; i++) {
        if (in[i] == '=' || in[i] == '\n' || in[i] == '\r') continue;
        int v = tbl[(uint8_t)in[i]];
        if (v < 0) continue;
        acc = (acc << 6) | v;
        nbits += 6;
        if (nbits >= 8) {
            nbits -= 8;
            out[olen++] = (acc >> nbits) & 0xFF;
        }
    }
    return olen;
}

static int parse_keys(const char* path, BruteConfig* cfg) {
    FILE* f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open keys file: %s\n", path); return -1; }
    char line[4096];
    int k = 0;
    while (fgets(line, sizeof(line), f) && k < MAX_KEYS) {
        line[strcspn(line, "\r\n")] = 0;
        if (!line[0]) continue;
        char* tok = strtok(line, ",");
        int w = 0;
        while (tok && w < MAX_WORDS) {
            int len = strlen(tok);
            if (len > MAX_WORD_LEN) len = MAX_WORD_LEN;
            cfg->words_len[k][w] = len;
            for (int i = 0; i < len; i++) cfg->words_pool[k][w][i] = (uint8_t)tok[i];
            cfg->words_num[k] = w + 1;
            tok = strtok(NULL, ",");
            w++;
        }
        k++;
    }
    fclose(f);
    cfg->keys_num = k;
    return 0;
}

static int load_message(const char* msg_file, BruteConfig* cfg) {
    FILE* f = fopen(msg_file, "r");
    if (!f) { fprintf(stderr, "cannot open msg file: %s\n", msg_file); return -1; }
    char b64[20000];
    size_t rlen = fread(b64, 1, sizeof(b64)-1, f);
    b64[rlen] = 0;
    fclose(f);
    uint8_t raw[MAX_MSG];
    int raw_len = b64decode(b64, raw);
    if (raw_len < 32 || memcmp(raw, "Salted__", 8) != 0) {
        fprintf(stderr, "bad message (len=%d, not Salted__)\n", raw_len);
        return -1;
    }
    for (int i = 0; i < 8; i++) cfg->salt[i] = raw[8 + i];
    cfg->ciphertext_size = raw_len - 16;
    memcpy(cfg->ciphertext, raw + 16, cfg->ciphertext_size);
    return 0;
}

#ifndef SOLVER_HOST_ONLY
int main(int argc, char* argv[]) {
    const char* msg_file = NULL;
    const char* keys_file = NULL;
    const char* selftest_pw = NULL;
    const char* selftest_gpu_pw = NULL;
    const char* solution_tag = "\"kty\":\"RSA\"";
    int block = 256, grid = 128, verbose = 1;

    for (int i = 1; i < argc; i++) {
        if (!strncmp(argv[i], "--msg_file=", 11)) msg_file = argv[i] + 11;
        else if (!strncmp(argv[i], "--keys_file=", 12)) keys_file = argv[i] + 12;
        else if (!strncmp(argv[i], "--selftest-pw=", 14)) selftest_pw = argv[i] + 14;
        else if (!strncmp(argv[i], "--selftest-gpu=", 15)) selftest_gpu_pw = argv[i] + 15;
        else if (!strncmp(argv[i], "--solution_tag=", 15)) solution_tag = argv[i] + 15;
        else if (!strncmp(argv[i], "--block=", 8)) block = atoi(argv[i] + 8);
        else if (!strncmp(argv[i], "--grid=", 7)) grid = atoi(argv[i] + 7);
        else if (!strncmp(argv[i], "--verbose=", 10)) verbose = atoi(argv[i] + 10);
    }
    if (!msg_file) {
        fprintf(stderr, "usage: %s --msg_file=... (--keys_file=... | --selftest-pw=... | --selftest-gpu=...)\n", argv[0]);
        return 2;
    }

    BruteConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    if (load_message(msg_file, &cfg) != 0) return 2;
    cfg.tag_len = strlen(solution_tag);
    if (cfg.tag_len > 15) cfg.tag_len = 15;
    for (int i = 0; i < cfg.tag_len; i++) cfg.tag[i] = (uint8_t)solution_tag[i];

    // ---- CPU SELF-TEST GATE ----
    if (selftest_pw) {
        uint8_t pw[64];
        int pw_len = strlen(selftest_pw);
        for (int i = 0; i < pw_len; i++) pw[i] = (uint8_t)selftest_pw[i];
        uint8_t ok = verify_candidate(pw, pw_len, &cfg);
        if (ok) {
            printf("SELFTEST_OK tag_found=true pw='%s'\n", selftest_pw);
            return 0;
        } else {
            printf("SELFTEST_FAIL tag_found=false pw='%s'\n", selftest_pw);
            return 1;
        }
    }

    if (selftest_gpu_pw) {
        // GPU self-test: single position, single word = known-good password.
        // Exercises the REAL kernel: table access, key feeding, tag search.
        int len = strlen(selftest_gpu_pw);
        if (len > MAX_WORD_LEN) len = MAX_WORD_LEN;
        cfg.keys_num = 1;
        cfg.words_num[0] = 1;
        cfg.words_len[0][0] = len;
        for (int i = 0; i < len; i++) cfg.words_pool[0][0][i] = (uint8_t)selftest_gpu_pw[i];
    } else {
        if (!keys_file) {
            fprintf(stderr, "need --keys_file=...\n");
            return 2;
        }
        if (parse_keys(keys_file, &cfg) != 0) return 2;
    }

    uint64_t total = 1;
    for (int i = 0; i < cfg.keys_num; i++) total *= cfg.words_num[i];

    if (verbose) {
        printf("keys_num=%d total_combos=%llu ciphertext=%u bytes tag='%s'\n",
               cfg.keys_num, (unsigned long long)total, cfg.ciphertext_size, solution_tag);
        for (int i = 0; i < cfg.keys_num; i++) {
            printf("  pos %d (%u words): ", i, cfg.words_num[i]);
            for (int w = 0; w < cfg.words_num[i] && w < 12; w++)
                printf("%.*s ", cfg.words_len[i][w], cfg.words_pool[i][w]);
            if (cfg.words_num[i] > 12) printf("...");
            printf("\n");
        }
        fflush(stdout);
    }

    check_cuda(cudaSetDevice(0), "cudaSetDevice");
    cudaDeviceProp prop;
    check_cuda(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    if (verbose) printf("GPU: %s\n", prop.name);
    if (selftest_gpu_pw) {
        printf("GPU_SELFTEST running kernel with 1 known-good candidate...\n");
        fflush(stdout);
    }

    BruteConfig* d_cfg;
    BruteResult* d_ret;
    check_cuda(cudaMalloc(&d_cfg, sizeof(BruteConfig)), "cudaMalloc cfg");
    check_cuda(cudaMalloc(&d_ret, sizeof(BruteResult)), "cudaMalloc ret");
    check_cuda(cudaMemcpy(d_cfg, &cfg, sizeof(BruteConfig), cudaMemcpyHostToDevice), "memcpy cfg");
    check_cuda(cudaMemset(d_ret, 0, sizeof(BruteResult)), "memset ret");

    const uint64_t chunk = (uint64_t)grid * block;
    uint64_t done = 0;
    cudaEvent_t start_ev, end_ev;
    cudaEventCreate(&start_ev); cudaEventCreate(&end_ev);
    cudaEventRecord(start_ev);

    while (done < total) {
        uint64_t begin = done;
        uint64_t end = done + chunk;
        if (end > total) end = total;
        brute_kernel<<<grid, block>>>(d_cfg, begin, end, d_ret);
        check_cuda(cudaGetLastError(), "kernel launch");
        check_cuda(cudaDeviceSynchronize(), "kernel sync");
        done = end;

        BruteResult host_ret;
        check_cuda(cudaMemcpy(&host_ret, d_ret, sizeof(BruteResult), cudaMemcpyDeviceToHost), "memcpy ret");
        if (host_ret.found_flag) {
            if (selftest_gpu_pw) {
                // GPU self-test: success means the DEVICE pipeline is correct
                printf("GPU_SELFTEST_OK tag_found=true pw='%s'\n", selftest_gpu_pw);
                return 0;
            }
            printf("FOUND %llu/%llu\n", (unsigned long long)end, (unsigned long long)total);
            fwrite(host_ret.phrase, 1, host_ret.phrase_len, stdout);
            printf("\n");
            FILE* wf = fopen("SOLVED_wallet.json", "wb");
            if (wf) { fwrite(host_ret.wallet, 1, host_ret.wallet_len, wf); fclose(wf); }
            fprintf(stderr, "SOLVED_WALLET_SAVED len=%u\n", host_ret.wallet_len);
            return 0;
        }
        if (verbose && (end % (chunk * 100) < chunk || end == total)) {
            cudaEventRecord(end_ev); cudaEventSynchronize(end_ev);
            float ms; cudaEventElapsedTime(&ms, start_ev, end_ev);
            double rate = (double)end / (ms / 1000.0);
            fprintf(stderr, "  ...%llu/%llu (%.0f/s)\n", (unsigned long long)end, (unsigned long long)total, rate);
        }
    }

    if (selftest_gpu_pw) {
        fprintf(stderr, "GPU_SELFTEST_FAIL tag_found=false pw='%s'\n", selftest_gpu_pw);
        return 1;
    }
    fprintf(stderr, "NO_MATCH %llu combos\n", (unsigned long long)total);
    return 1;
}
#endif // !SOLVER_HOST_ONLY
