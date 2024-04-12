#pragma once

#include <robin_hood.h>
#include <spdlog/fmt/ranges.h>
#include <spdlog/spdlog.h>

#include <cassert>
#include <cstdint>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <vector>

#include "SimulationConfig.h"
#include "Instruction.h"
#include "nlohmann/json.hpp"
#include "onnx/defs/schema.h"
#include "onnx/onnx-operators_pb.h"
#include "onnx/onnx_pb.h"

#define MIN(x, y) (((x) > (y)) ? (y) : (x))
#define MIN3(x, y, z) MIN(MIN(x, y), z)
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

#define KB *1024

#define PAGE_SIZE 4096

using json = nlohmann::json;

typedef uint64_t addr_type;
typedef uint64_t cycle_type;

typedef struct {
  enum class Status {
    INITIALIZED,
    RUNNING,
    FINISH,
    EMPTY,
  };
  Status status = Status::EMPTY;
  uint32_t required_sram_size;
  size_t num_parent_tiles = 0;
  std::deque<std::unique_ptr<Instruction>> instructions;
  std::vector<size_t*> child_tiles_counter;
} Tile;

uint32_t generate_id();
SimulationConfig initialize_config(json config);