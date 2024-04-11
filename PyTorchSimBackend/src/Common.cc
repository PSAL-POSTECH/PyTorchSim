#include "Common.h"

uint32_t generate_id() {
  static uint32_t id_counter{0};
  return id_counter++;
}
uint32_t generate_mem_access_id() {
  static uint32_t id_counter{0};
  return id_counter++;
}

SimulationConfig initialize_config(json config) {
  SimulationConfig parsed_config;

  /* Core configs */
  parsed_config.num_cores = config["num_cores"];
  parsed_config.core_freq = config["core_freq"];

  /* DRAM config */
  if ((std::string)config["dram_type"] == "simple")
    parsed_config.dram_type = DramType::SIMPLE;
  else if ((std::string)config["dram_type"] == "ramulator")
    parsed_config.dram_type = DramType::RAMULATOR;
  else
    throw std::runtime_error(fmt::format("Not implemented dram type {} ",
                                         (std::string)config["dram_type"]));
  parsed_config.dram_freq = config["dram_freq"];
  if (config.contains("dram_latency"))
    parsed_config.dram_latency = config["dram_latency"];
  if (config.contains("dram_config_path"))
    parsed_config.dram_config_path = config["dram_config_path"];
  parsed_config.dram_channels = config["dram_channels"];
  if (config.contains("dram_req_size"))
    parsed_config.dram_req_size = config["dram_req_size"];
  if (config.contains("dram_print_interval"))
    parsed_config.dram_print_interval = config["dram_print_interval"];

  /* Icnt config */
  if ((std::string)config["icnt_type"] == "simple")
    parsed_config.icnt_type = IcntType::SIMPLE;
  else if ((std::string)config["icnt_type"] == "booksim2")
    parsed_config.icnt_type = IcntType::BOOKSIM2;
  else
    throw std::runtime_error(fmt::format("Not implemented icnt type {} ",
                                         (std::string)config["icnt_type"]));
  parsed_config.icnt_freq = config["icnt_freq"];
  if (config.contains("icnt_latency"))
    parsed_config.icnt_latency = config["icnt_latency"];
  if (config.contains("icnt_config_path"))
    parsed_config.icnt_config_path = config["icnt_config_path"];

  parsed_config.scheduler_type = config["scheduler"];
  parsed_config.precision = config["precision"];
  parsed_config.layout = config["layout"];

  if (config.contains("partition")) {
    for (int i=0; i<parsed_config.num_cores; i++) {
      std::string core_partition = "core_" + std::to_string(i);
      uint32_t partition_id = uint32_t(config["partition"][core_partition]);
      parsed_config.partiton_map[partition_id].push_back(i);
      spdlog::info("CPU {}: Partition {}", i, partition_id);
    }
  } else {
    /* Default: all partition 0 */
    for (int i=0; i<parsed_config.num_cores; i++) {
      parsed_config.partiton_map[0].push_back(i);
      spdlog::info("CPU {}: Partition {}", i, 0);
    }
  }
  return parsed_config;
}