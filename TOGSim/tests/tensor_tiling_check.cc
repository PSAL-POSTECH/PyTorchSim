// One (8,128) fp32 tile of a 512x512 tensor, loaded by a MOVIN: row-major it is eight
// 512 B runs at a 2 KiB pitch; in (8,128) tiles it is the tensor's first 4 KiB, contiguous.
// Fails to build without TensorTiling and fails its asserts if the tiling is not applied.
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <set>
#include <vector>

#include "Instruction.h"

static std::set<addr_type> requests(std::shared_ptr<const TensorTiling> tiling) {
  Instruction inst(Opcode::MOVIN, 0, 0, 0x10000000, {8, 128}, {512, 1}, 32, {0}, {1}, {});
  inst.set_tensor_tiling(std::move(tiling));
  return *inst.get_dram_address(32);
}

static void expect(bool ok, const char* what) {
  std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
  if (!ok) std::exit(1);
}

int main() {
  const auto row = requests(nullptr);
  const auto tiled = requests(std::make_shared<TensorTiling>(TensorTiling{0x10000000, 512, 512, 4, 8, 128}));
  std::set<addr_type> run, pitched;
  for (addr_type a = 0; a < 4096; a += 32) run.insert(0x10000000 + a);
  for (addr_type r = 0; r < 8; ++r)
    for (addr_type a = 0; a < 512; a += 32) pitched.insert(0x10000000 + r * 2048 + a);
  expect(row == pitched, "row-major: eight 512 B runs at a 2 KiB pitch");
  expect(tiled == run, "(8,128) tiles: one contiguous 4 KiB run");
  expect(TensorTiling{0, 9, 130, 2, 16, 128}.footprint() == 16 * 256 * 2, "footprint pads to whole tiles");
  return 0;
}
