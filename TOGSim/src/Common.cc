#include "Common.h"

#include "Dram.h"

#include <optional>

bool loadConfig(const std::string& config_path, YAML::Node& config_yaml) {
  try {
    config_yaml = YAML::LoadFile(config_path);
    spdlog::info("[LoadConfig] Loaded configuration file \"{}\"", config_path);
    return true;
  } catch (const YAML::BadFile& e) {
    spdlog::error("[LoadConfig] Failed to open \"{}\" (File not found or inaccessible)", config_path);
    return false;
  } catch (const YAML::ParserException& e) {
    spdlog::error("[LoadConfig] Failed to parse YAML file \"{}\": {}", config_path, e.what());
    return false;
  } catch (const std::exception& e) {
    spdlog::error("[LoadConfig] Unknown error loading \"{}\": {}", config_path, e.what());
    return false;
  }
}

template <typename T>
T get_config_value(const YAML::Node& config, std::string key) {
  if (config[key]) {
    return config[key].as<T>();
  } else {
    throw std::runtime_error(fmt::format("Config key {} not found", key));
  }
}

SimulationConfig initialize_config(const YAML::Node& config,
                                     const std::string& config_file_path) {
  SimulationConfig parsed_config;
  parsed_config.config_file_path = config_file_path;
  YAML::Emitter emitter;
  emitter << config;
  spdlog::info("PyTorchSim config:\n{}", emitter.c_str());

  /* Core configs */
  parsed_config.num_cores = get_config_value<uint32_t>(config, "num_cores");
  if (config["core_type"]) {
    std::vector<std::string> core_types = config["core_type"].as<std::vector<std::string>>();

    if (core_types.size() != parsed_config.num_cores)
      throw std::runtime_error("Mismatch between num_cores and core_type list size");

    for (const auto& core_type : core_types) {
      if (core_type == "ws_mesh") {
        parsed_config.core_type.push_back(CoreType::WS_MESH);
      } else if (core_type == "stonne") {
        parsed_config.core_type.push_back(CoreType::STONNE);
      } else {
        throw std::runtime_error(fmt::format("Not implemented core type: {}", core_type));
      }
    }
  } else {
    /* Used WS as default */
    for (int i=0; i<parsed_config.num_cores; i++)
      parsed_config.core_type.push_back(CoreType::WS_MESH);
  }

  parsed_config.core_freq_mhz = get_config_value<uint32_t>(config, "core_freq_mhz");
  if (config["num_systolic_array_per_core"])
    parsed_config.num_systolic_array_per_core = config["num_systolic_array_per_core"].as<uint32_t>();
  if (config["core_spad_size_kb"])
    parsed_config.core_spad_size_kb = config["core_spad_size_kb"].as<uint32_t>();
  if (config["sa_weight_buffer_depth"])
    parsed_config.sa_weight_buffer_depth = config["sa_weight_buffer_depth"].as<uint32_t>();
  if (config["num_stonne_per_core"])
    parsed_config.num_stonne_per_core = config["num_stonne_per_core"].as<uint32_t>();
  if (config["num_stonne_port"])
    parsed_config.num_stonne_port = config["num_stonne_port"].as<uint32_t>();
  parsed_config.core_print_interval = get_config_value<uint32_t>(config, "core_stats_print_period_cycles");

  /* Stonne config */
  if (config["stonne_config_path"])
    parsed_config.stonne_config_path = config["stonne_config_path"].as<std::string>();

  /* DRAM config */
  std::string dram_type_str = get_config_value<std::string>(config, "dram_type");

  if (dram_type_str == "simple") {
    parsed_config.dram_type = DramType::SIMPLE;
  } else if (dram_type_str == "ramulator2") {
    parsed_config.dram_type = DramType::RAMULATOR2;
    const std::string ramulator_config_rel =
        get_config_value<std::string>(config, "ramulator_config_path");
    parsed_config.dram_config_path =
        parsed_config.resolve_against_simulation_config(ramulator_config_rel);
  } else {
    throw std::runtime_error(fmt::format("Not implemented dram type {} ", dram_type_str));
  }

  parsed_config.dram_channels = get_config_value<uint32_t>(config, "dram_channels");

  if (parsed_config.dram_type == DramType::RAMULATOR2) {
    DramRamulator2::apply_ramulator_config_to_simulation_config(
        parsed_config, parsed_config.dram_config_path,
        config["dram_freq_mhz"] ? std::optional<uint32_t>(config["dram_freq_mhz"].as<uint32_t>()) : std::nullopt);
  } else {
    SimpleDRAM::apply_yaml_to_simulation_config(config, parsed_config);
  }

  if (config["dram_stats_print_period_cycles"])
    parsed_config.dram_print_interval = config["dram_stats_print_period_cycles"].as<uint32_t>();
  if (config["dram_num_partitions"]) {
    parsed_config.dram_num_partitions = config["dram_num_partitions"].as<uint32_t>();
    if (parsed_config.dram_channels % parsed_config.dram_num_partitions != 0) {
      throw std::runtime_error("[Config] DRAM channels must be divisible by dram_num_partitions");
    }
  }

  if (parsed_config.dram_num_partitions != 0) {
      parsed_config.dram_channels_per_partitions =
        parsed_config.dram_channels / parsed_config.dram_num_partitions;
  } else {
      parsed_config.dram_channels_per_partitions = parsed_config.dram_channels;
  }

   /* L2D config */
  if (config["l2d_type"]) {
    std::string l2d_type_str = config["l2d_type"].as<std::string>();
    if (l2d_type_str == "nocache")
      parsed_config.l2d_type = L2CacheType::NOCACHE;
    else if (l2d_type_str == "datacache") {
      parsed_config.l2d_type = L2CacheType::DATACACHE;
      parsed_config.l2d_config_str = get_config_value<std::string>(config, "l2d_config");
      if (config["l2d_hit_latency"])
        parsed_config.l2d_hit_latency = config["l2d_hit_latency"].as<uint32_t>();
    } else
      throw std::runtime_error(fmt::format("Not implemented l2 cache type {} ", l2d_type_str));
  } else {
    parsed_config.l2d_type = L2CacheType::NOCACHE;
  }

  /* Icnt config */
  std::string icnt_type_str = config["icnt_type"].as<std::string>();
  if (icnt_type_str == "simple") {
    parsed_config.icnt_type = IcntType::SIMPLE;
    if (config["icnt_latency_cycles"])
      parsed_config.icnt_latency = config["icnt_latency_cycles"].as<uint32_t>();
  } else if (icnt_type_str == "booksim2") {
    parsed_config.icnt_type = IcntType::BOOKSIM2;
    const std::string booksim_config_rel =
        get_config_value<std::string>(config, "booksim_config_path");
    parsed_config.icnt_config_path =
        parsed_config.resolve_against_simulation_config(booksim_config_rel);
  } else
    throw std::runtime_error(fmt::format("Not implemented icnt type {} ", icnt_type_str));

  parsed_config.icnt_freq_mhz = config["icnt_freq_mhz"].as<double>();
  if (config["icnt_stats_print_period_cycles"])
    parsed_config.icnt_stats_print_period_cycles = config["icnt_stats_print_period_cycles"].as<uint32_t>();
  if (config["icnt_injection_ports_per_core"])
    parsed_config.icnt_injection_ports_per_core = config["icnt_injection_ports_per_core"].as<uint32_t>();

  /* Energy config */
  if (config["energy_cost_table_path"]) {
    const std::string cost_table_rel = config["energy_cost_table_path"].as<std::string>();
    parsed_config.dram_energy_costs =
        load_dram_energy_costs(parsed_config.resolve_against_simulation_config(cost_table_rel));
    parsed_config.energy_model_enabled = true;
  }

  if (config["scheduler"])
    parsed_config.scheduler_type = config["scheduler"].as<std::string>();
  if (config["num_partition"])
    parsed_config.num_partition = config["num_partition"].as<uint32_t>();
  if (config["partition"]) {
    for (int i=0; i<parsed_config.num_cores; i++) {
      std::string core_partition = "core_" + std::to_string(i);
      if (config["partition"][core_partition]) {
          uint32_t partition_id = config["partition"][core_partition].as<uint32_t>();
          parsed_config.partiton_map[i] = partition_id;
          spdlog::info("[Config/Core] core_id: {}, partition_id: {}", i, partition_id);
      } else {
          spdlog::warn("[Config/Core] core_id: {}, partition: missing in config, using partition_id 0", i);
          parsed_config.partiton_map[i] = 0;
      }
    }
  } else {
    for (int i=0; i<parsed_config.num_cores; i++) {
      parsed_config.partiton_map[i] = 0;
      spdlog::info("[Config/Core] core_id: {}, partition_id: 0 (no partition section)", i);
    }
  }
  return parsed_config;
}
