#include "Common.h"

uint32_t generate_id() {
  static uint32_t id_counter{0};
  return id_counter++;
}

template <typename T>
T get_config_value(json config, std::string key) {
  if (config.contains(key)) {
    return config[key];
  } else {
    throw std::runtime_error(fmt::format("Config key {} not found", key));
  }
}

SimulationConfig initialize_config(json config) {
  SimulationConfig parsed_config;

  /* Core configs */
  parsed_config.num_cores = config["num_cores"];
  parsed_config.core_freq = config["core_freq"];
  parsed_config.sram_size = config["sram_size"];
  if (config.contains("num_systolic_array_per_core"))
    parsed_config.num_systolic_array_per_core = config["num_systolic_array_per_core"];
  parsed_config.core_print_interval = get_config_value<uint32_t>(config, "core_print_interval");

  /* DRAM config */
  if ((std::string)config["dram_type"] == "simple")
    parsed_config.dram_type = DramType::SIMPLE;
  else if ((std::string)config["dram_type"] == "ramulator")
    parsed_config.dram_type = DramType::RAMULATOR1;
  else if ((std::string)config["dram_type"] == "ramulator2")
    parsed_config.dram_type = DramType::RAMULATOR2;
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
  if (config.contains("dram_num_partitions"))
    parsed_config.dram_num_partitions = config["dram_num_partitions"];

   /* L2D config */
  if (config.contains("l2d_type")) {
    if ((std::string)config["l2d_type"] == "nocache")
      parsed_config.l2d_type = L2CacheType::NOCACHE;
    else if ((std::string)config["l2d_type"] == "readonly")
      parsed_config.l2d_type = L2CacheType::READONLY;
    else
      throw std::runtime_error(fmt::format("Not implemented l2 cache type {} ",
                                          (std::string)config["l2d_type"]));
  } else {
    parsed_config.l2d_type = L2CacheType::NOCACHE;
  }

  if (config.contains("l2d_config"))
    parsed_config.l2d_config_str = config["l2d_config"];
  if (config.contains("l2d_hit_latency"))
    parsed_config.l2d_config_str = config["l2d_hit_latency"];

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
  if (config.contains("icnt_print_interval"))
    parsed_config.icnt_print_interval = config["icnt_print_interval"];
  if (config.contains("icnt_node_per_core"))
    parsed_config.icnt_node_per_core = config["icnt_node_per_core"];

  parsed_config.scheduler_type = config["scheduler"];
  if (config.contains("num_partition"))
    parsed_config.num_patition = config["num_partition"];
  if (config.contains("partition")) {
    for (int i=0; i<parsed_config.num_cores; i++) {
      std::string core_partition = "core_" + std::to_string(i);
      uint32_t partition_id = uint32_t(config["partition"][core_partition]);
      parsed_config.partiton_map[i] = partition_id;
      spdlog::info("[Config/Core] CPU {}: Partition {}", i, partition_id);
    }
  } else {
    /* Default: all partition 0 */
    for (int i=0; i<parsed_config.num_cores; i++) {
      parsed_config.partiton_map[i] = 0;
      spdlog::info("[Config/Core] CPU {}: Partition {}", i, 0);
    }
  }
  return parsed_config;
}
