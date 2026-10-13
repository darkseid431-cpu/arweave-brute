# arweave-brute — Arweave-puzzle brute-force solver (CUDA) with a CPU self-test gate.
#
# Targets:
#   make test         CPU gate (needs only g++): compile host_selftest, run all
#                     known-answer vectors. MUST pass before trusting anything.
#   make solver       build the CUDA solver -> ./our_solver  (needs nvcc)
#   make solver-gate  gates the nvcc-built binary: CPU --selftest-pw on every
#                     fixture + a real-kernel --selftest-gpu run (needs a GPU)
#   make clean
#
# ARCH: sm_75 = T4/Turing (default). Override: make solver ARCH=sm_80

NVCC      ?= nvcc
CXX       ?= g++
ARCH      ?= sm_75
CXXFLAGS  ?= -O2 -std=c++11
NVCCFLAGS ?= -O3 -std=c++11 -arch=$(ARCH)

# Known-answer vectors — passwords of REAL solved puzzles (see README).
PW_AR8 := RasputinWilhelmAlekhine
PW_AR5 := *48GCEErisUmberCastlePicasso
PW_ETH7 := Vivaldi227PermanentArnheimBulgakov1157Ulysses143176176209
TAG_ETH := {"version":3,"id

.PHONY: all test solver solver-gate clean

all: test

host_selftest: host_selftest.cpp our_solver.cu cuda_stub/cuda_runtime.h
	$(CXX) $(CXXFLAGS) -I cuda_stub -o $@ host_selftest.cpp

test: host_selftest
	@echo "== gate 1/3: pzl8 (AR wallet, tag \"kty\":\"RSA\") =="
	./host_selftest fixtures/pzl8_msg.b64 '$(PW_AR8)'
	@echo "== gate 2/3: pzl5 (AR wallet, star included in password) =="
	./host_selftest fixtures/pzl5_msg.b64 '$(PW_AR5)'
	@echo "== gate 3/3: pzl7 (ETH wallet, tag {\"version\":3,\"id) =="
	./host_selftest fixtures/pzl7_msg.b64 '$(PW_ETH7)' '$(TAG_ETH)'
	@echo "ALL KNOWN-ANSWER GATES PASS"

solver: our_solver.cu
	$(NVCC) $(NVCCFLAGS) -o our_solver our_solver.cu

solver-gate: solver
	./our_solver --msg_file=fixtures/pzl8_msg.b64 --selftest-pw='$(PW_AR8)'
	./our_solver --msg_file=fixtures/pzl5_msg.b64 --selftest-pw='$(PW_AR5)'
	./our_solver --msg_file=fixtures/pzl7_msg.b64 --solution_tag='$(TAG_ETH)' --selftest-pw='$(PW_ETH7)'
	@echo "CPU gates passed — now the real-kernel GPU gate (needs a CUDA device)"
	./our_solver --msg_file=fixtures/pzl8_msg.b64 --selftest-gpu='$(PW_AR8)' --block=256 --grid=256
	@echo "ALL BINARY GATES PASS"

clean:
	rm -f our_solver host_selftest *.o SOLVED_wallet.json
