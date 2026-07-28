#pragma once

#include <robin_hood.h>
#include <spdlog/fmt/ranges.h>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include <cassert>
#include <cstdint>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <vector>

#include "SimulationConfig.h"
#include "Instruction.h"

#define MIN(x, y) (((x) > (y)) ? (y) : (x))
#define MIN3(x, y, z) MIN(MIN(x, y), z)
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

#define KB *1024

#define PAGE_SIZE 4096

typedef uint64_t addr_type;
typedef uint64_t cycle_type;

bool loadConfig(const std::string& config_path, YAML::Node& config_yaml);
SimulationConfig initialize_config(const YAML::Node& config,
                                     const std::string& config_file_path = {});