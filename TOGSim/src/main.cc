#include <fstream>
#include <chrono>
#include <filesystem>
#include <sstream>
#include <thread>
#include <atomic>

#include "Simulator.h"
#include "TileGraphParser.h"
#include "helper/CommandLineParser.h"
#include "togsim_loader.h"        // P3 trace pipeline: run a compiled producer .so
#include "togsim_trace_bridge.h"  // ... and bridge its trace to a TileGraph

namespace fs = std::filesystem;
namespace po = boost::program_options;


void launchKernel(Simulator* simulator, unsigned int kernel_id, std::string onnx_path, std::string attribute_path, const YAML::Node& config_yaml, cycle_type request_time=0, int partition_id=0, int device_id=0) {
  auto graph_praser = TileGraphParser(onnx_path, attribute_path, config_yaml);
  std::unique_ptr<TileGraph>& tile_graph = graph_praser.get_tile_graph();
  tile_graph->set_arrival_time(request_time ? request_time : simulator->get_core_cycle());
  tile_graph->set_kernel_id(kernel_id);
  spdlog::info("[Scheduler {}] Enqueued kernel_id: {}, tog_path: {}, operation: {}, request_time_cycles: {}",
               partition_id, kernel_id, onnx_path, tile_graph->get_name(), request_time);
  simulator->enqueue_graph(partition_id, std::move(tile_graph));
}

void process_trace_file(Simulator* simulator, std::string trace_file_path, const YAML::Node& config_yaml) {
  // Open trace file (can be FIFO or regular file)
  std::ifstream trace_file;
  trace_file.open(trace_file_path);
  if (!trace_file.is_open()) {
    spdlog::error("[TOGSim] Failed to open trace file: {}", trace_file_path);
    return;
  }
  spdlog::info("[TOGSim] Reading trace file: {}", trace_file_path);

  // Read all available commands and process them
  std::string line;
  while (std::getline(trace_file, line)) {
    if (line.empty()) {
      continue;
    }

    // Skip comment lines starting with #
    if (line[0] == '#') {
      continue;
    }

    // Parse command: command_type,kernel_id,device_index,stream_index,tog_path,attribute_path,timestamp
    std::istringstream iss(line);
    std::string token;
    std::vector<std::string> tokens;

    while (std::getline(iss, token, ',')) {
      tokens.push_back(token);
    }

    if (tokens.size() != 7) {
      spdlog::error("[TOGSim] Invalid command format. Expected: command_type,kernel_id,device_index,stream_index,tog_path,attribute_path,timestamp. Got: {} ({} tokens)", line, tokens.size());
      continue;
    }

    std::string command_type = tokens[0];
    unsigned int kernel_id = std::stoul(tokens[1]);
    int device_index = std::stoi(tokens[2]);
    int stream_index = std::stoi(tokens[3]);
    std::string tog_path = tokens[4];
    std::string attribute_path = tokens[5];
    int timestamp = std::stoi(tokens[6]);
    // timestamp (tokens[6]) is available but not used in current implementation

    try {
      if (command_type == "LAUNCH_KERNEL") {
        launchKernel(simulator, kernel_id, tog_path, attribute_path, config_yaml, timestamp, stream_index, device_index);
      } else if (command_type == "DEVICE_SYNC") {
        simulator->cycle();
        spdlog::info("[Device {}] Device synchronization completed", device_index);
      } else {
        spdlog::error("[TOGSim] Unknown command type: {}", command_type);
      }
    } catch (const std::exception& e) {
      spdlog::error("[TOGSim] Error processing command {} (type: {}): {}", kernel_id, command_type, e.what());
    }
  }
  trace_file.close();
  simulator->cycle();
}

Simulator* create_simulator(const std::string& config_path) {
  YAML::Node config_yaml;
  if (!loadConfig(config_path, config_yaml)) {
    return nullptr;
  }
  SimulationConfig config = initialize_config(config_yaml, config_path);
  return new Simulator(config, std::move(config_yaml));
}

int main(int argc, char** argv) {
  auto start = std::chrono::high_resolution_clock::now();
  // parse command line argumnet
  CommandLineParser cmd_parser = CommandLineParser();
  cmd_parser.add_command_line_option<std::string>(
      "config", "Path for hardware configuration file (.yml)");
  cmd_parser.add_command_line_option<std::string>(
      "models_list", "Path for the trace file (.trace)");
  cmd_parser.add_command_line_option<std::string>(
      "log_level", "Set for log level [trace, debug, info], default = info");
  cmd_parser.add_command_line_option<std::string>(
      "trace_so", "Path to a compiled trace producer .so (P3 trace pipeline)");
  cmd_parser.add_command_line_option<std::string>(
      "cycle_table", "Path to a 'cycle<TAB>overlapping' per-tile_id sidecar (TSV) "
                     "for --trace_so; falls back to a flat stub if omitted");
  try {
    cmd_parser.parse(argc, argv);
  } catch (const CommandLineParser::ParsingError& e) {
    spdlog::error(
        "Command line argument parsing error captured. Error message: {}",
        e.what());
    std::cerr << std::endl;
    cmd_parser.print_help_message();
    exit(1);
  }
  
  // Check if help was requested
  cmd_parser.print_help_message_if_required();

  // Dump full command for copy-paste re-run
  std::ostringstream cmd_oss;
  for (int i = 0; i < argc; ++i) {
    if (i > 0) cmd_oss << " ";
    cmd_oss << argv[i];
  }
  spdlog::info("[TOGSim] Command line: {}", cmd_oss.str());

  std::string level = "info";
  cmd_parser.set_if_defined("log_level", &level);
  if (level == "trace")
    spdlog::set_level(spdlog::level::trace);
  else if (level == "debug")
    spdlog::set_level(spdlog::level::debug);
  else if (level == "info")
    spdlog::set_level(spdlog::level::info);

  std::string config_path;
  std::string trace_file_path;

  /* Create simulator */
  cmd_parser.set_if_defined("config", &config_path);

  auto simulator = create_simulator(config_path);
  if (!simulator) {
    spdlog::error("[TOGSim] Failed to load config file: {}", config_path);
    exit(1);
  }

  // P3 trace pipeline: if a compiled producer .so is given, run it, bridge the
  // recorded trace to a TileGraph, and run the existing Simulator on it.
  std::string trace_so_path;
  cmd_parser.set_if_defined("trace_so", &trace_so_path);
  if (!trace_so_path.empty()) {
    const auto& cfg = simulator->get_hardware_config_yaml();
    int num_cores = cfg["num_cores"] ? cfg["num_cores"].as<int>() : 1;
    // First cut: stub tensor bases (real per-tensor addresses come later).
    std::vector<uint64_t> bases(16);
    for (size_t i = 0; i < bases.size(); ++i) bases[i] = 0x100000ull * (i + 1);
    // Cycle table: load the per-tile_id TSV sidecar if given, else a flat stub.
    std::vector<int64_t> cyc, ovl;
    std::string cycle_table_path;
    cmd_parser.set_if_defined("cycle_table", &cycle_table_path);
    if (!cycle_table_path.empty()) {
      std::ifstream ct(cycle_table_path);
      if (!ct.is_open()) { spdlog::error("[TOGSim] cannot open cycle_table {}", cycle_table_path); exit(1); }
      int64_t c, o;
      while (ct >> c >> o) { cyc.push_back(c); ovl.push_back(o); }
      spdlog::info("[TOGSim-trace] loaded cycle table: {} tiles from {}", cyc.size(), cycle_table_path);
    } else {
      cyc.assign(256, 128);
      ovl.assign(256, 0);
    }
    auto run = togsim::run_producer(trace_so_path.c_str(), nullptr, 0,
                                    bases.data(), (int)bases.size(),
                                    cyc.data(), ovl.data(), (int)cyc.size(),
                                    num_cores);
    if (!run.ok) { spdlog::error("[TOGSim] trace producer run failed"); exit(1); }
    spdlog::info("[TOGSim-trace] recorded {} instructions", run.trace.size());
    auto tg = trace_to_tilegraph(run, "trace_kernel");
    tg->set_arrival_time(simulator->get_core_cycle());
    tg->set_kernel_id(0);
    simulator->enqueue_graph(0, std::move(tg));
    simulator->run_simulator();
    spdlog::info("[TOGSim-trace] Total cycles: {}", simulator->get_core_cycle());
    spdlog::info("Simulation finished");
    simulator->print_core_stat();
    return 0;
  }

  // Get trace file path
  cmd_parser.set_if_defined("models_list", &trace_file_path);

  if (!trace_file_path.empty()) {
    // Process trace file (unified mode: supports both FIFO and regular file)
    process_trace_file(simulator, trace_file_path,
                       simulator->get_hardware_config_yaml());
    spdlog::info("Simulation finished");
    simulator->print_core_stat();
  } else {
    spdlog::error("No trace file provided. Use --models_list to specify trace file path.");
    exit(1);
  }
  delete simulator;

  /* Simulation time measurement */
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  spdlog::info("Wall-clock time for simulation: {:2f} seconds", duration.count());
  return 0;
}
