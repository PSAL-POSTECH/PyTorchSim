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
#include "helper/HelperFunctions.h"
#include "nlohmann/json.hpp"
#include "onnx/defs/schema.h"
#include "onnx/onnx-operators_pb.h"
#include "onnx/onnx_pb.h"

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
    BAR,
    EMPTY,
  };
  Status status = Status::EMPTY;
  uint32_t required_sram_size;
  std::deque<std::unique_ptr<Instruction>> instructions;
} Tile;

uint32_t generate_id();
SimulationConfig initialize_config(json config);