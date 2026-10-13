// host_selftest.cpp — compile the EXACT crypto code from our_solver.cu on CPU
// (CUDA stubbed out) and run the self-test gate against real known-good pairs.
//
// Build:
//   g++ -O2 -std=c++11 -I cuda_stub -o host_selftest host_selftest.cpp
// Run:
//   ./host_selftest <msg.b64> <known-good-password> [solution_tag]
//   solution_tag defaults to '"kty":"RSA"' (AR wallet); pass '{"version":3,"id'
//   for ETH-wallet puzzles. exit 0 => tag found (crypto pipeline correct)

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cstddef>

// ---- CUDA stubs so the .cu crypto compiles as plain C++ ----
#define __host__
#define __device__
#define __constant__ static const
#define __global__
// exclude the GPU-only kernel + main() (real CUDA syntax g++ can't parse)
#define SOLVER_HOST_ONLY
// rename the .cu's main() so it doesn't clash with ours
#define main solver_cu_main_unused

#include "our_solver.cu"

#undef main

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <msg.b64> <password> [solution_tag]\n", argv[0]);
        return 2;
    }

    // ---- gate 0: PKCS#7 strip — SOLVED_wallet.json must come out valid JSON ----
    {
        uint8_t buf[32]; uint32_t len;
        memset(buf, 0x41, 32);                       // 'A' payload
        for (int i = 27; i < 32; i++) buf[i] = 5;    // 5 bytes of 5 -> 27 + 5 = 32
        len = 32; strip_pkcs7(buf, &len);
        if (len != 27) { fprintf(stderr, "PKCS7_STRIP FAIL: padded len=%u want 27\n", len); return 1; }
        buf[31] = 9;                                 // pad-value mismatch must NOT strip
        len = 32; strip_pkcs7(buf, &len);
        if (len != 32) { fprintf(stderr, "PKCS7_STRIP FAIL: mismatched pad stripped, len=%u\n", len); return 1; }
        buf[31] = 0;                                 // out-of-range value must NOT strip
        len = 32; strip_pkcs7(buf, &len);
        if (len != 32) { fprintf(stderr, "PKCS7_STRIP FAIL: v=0 stripped, len=%u\n", len); return 1; }
        printf("PKCS7_STRIP OK (wallet padding removed; invalid padding preserved)\n");
    }
    BruteConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    if (load_message(argv[1], &cfg) != 0) return 2;
    const char* tag_s = (argc >= 4) ? argv[3] : "\"kty\":\"RSA\"";
    size_t tag_len = strlen(tag_s);
    if (tag_len == 0 || tag_len > sizeof(cfg.tag)) {
        fprintf(stderr, "solution_tag must be 1..%zu bytes (got %zu)\n", sizeof(cfg.tag), tag_len);
        return 2;
    }
    cfg.tag_len = (uint8_t)tag_len;
    memcpy(cfg.tag, tag_s, tag_len);

    const char* pw_s = argv[2];
    uint8_t pw[64];
    int pw_len = strlen(pw_s);
    for (int i = 0; i < pw_len; i++) pw[i] = (uint8_t)pw_s[i];

    uint8_t ok = verify_candidate(pw, pw_len, &cfg);
    printf("SELFTEST %s tag_found=%s pw='%s' ciphertext=%u bytes\n",
           ok ? "OK" : "FAIL", ok ? "true" : "false", pw_s, cfg.ciphertext_size);
    return ok ? 0 : 1;
}
