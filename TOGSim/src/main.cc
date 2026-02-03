#include <fstream>
#include <chrono>
#include <filesystem>
#include <sstream>
#include <thread>
#include <atomic>

#include "Simulator.h"
#include "TileGraphParser.h"
#include "helper/CommandLineParser.h"

namespace fs = std::filesystem;
namespace po = boost::program_options;


void launchKernel(Simulator* simulator, unsigned int kernel_id, std::string onnx_path, std::string attribute_path, const YAML::Node& config_yaml, cycle_type request_time=0, int partiton_id=0, int device_id=0) {
  auto graph_praser = TileGraphParser(onnx_path, attribute_path, config_yaml);
  std::unique_ptr<TileGraph>& tile_graph = graph_praser.get_tile_graph();
  tile_graph->set_arrival_time(request_time ? request_time : simulator->get_core_cycle());
  tile_graph->set_kernel_id(kernel_id);
  spdlog::info("[Scheduler {}] launch kernel {} tog: {} operation: {} at {}", device_id, kernel_id, partiton_id, onnx_path, tile_graph->get_name(), simulator->get_core_cycle());
  simulator->schedule_graph(partiton_id, std::move(tile_graph));
}

void process_trace_file(Simulator* simulator, std::string trace_file_path, const YAML::Node& config_yaml) {
  // Open trace file (can be FIFO or regular file)
  std::ifstream trace_file;
  trace_file.open(trace_file_path);
  if (!trace_file.is_open()) {
    spdlog::error("[TOGSim] Failed to open trace file: {}", trace_file_path);
    return;
  }
  spdlog::info("[TOGSim] Reading from trace file: {}", trace_file_path);

  // Read all available commands and launch kernels
  std::string line;
  while (std::getline(trace_file, line)) {
    if (line.empty()) {
      continue;
    }

    // Parse command: id,device_index,stream_index,tog_path,attribute_path
    std::istringstream iss(line);
    std::string token;
    std::vector<std::string> tokens;

    while (std::getline(iss, token, ',')) {
      tokens.push_back(token);
    }

    if (tokens.size() != 5) {
      spdlog::error("[TOGSim] Invalid command format. Expected: id,device_index,stream_index,tog_path,attribute_path. Got: {}", line);
      continue;
    }

    unsigned int kernel_id = std::stoul(tokens[0]);
    int device_index = std::stoi(tokens[1]);
    int stream_index = std::stoi(tokens[2]);
    std::string tog_path = tokens[3];
    std::string attribute_path = tokens[4];

    try {
      launchKernel(simulator, kernel_id, tog_path, attribute_path, config_yaml, 0, stream_index, device_index);
    } catch (const std::exception& e) {
      spdlog::error("[TOGSim] Error processing kernel {}: {}", kernel_id, e.what());
    }
  }
  trace_file.close();
  simulator->cycle();
}

Simulator* create_simulator(std::string config_path) {
  YAML::Node config_yaml;
  if (!loadConfig(config_path, config_yaml))
    exit(1);
  SimulationConfig config = initialize_config(config_yaml);

  auto simulator = new Simulator(config);
  return simulator;
}

int main(int argc, char** argv) {
  auto start = std::chrono::high_resolution_clock::now();
  // parse command line argumnet
  CommandLineParser cmd_parser = CommandLineParser();
  cmd_parser.add_command_line_option<std::string>(
      "config", "Path for hardware configuration file");
  cmd_parser.add_command_line_option<std::string>(
      "models_list", "Path for the models list file (can be FIFO or regular file)");
  cmd_parser.add_command_line_option<std::string>(
      "log_level", "Set for log level [trace, debug, info], default = info");
  try {
    cmd_parser.parse(argc, argv);
  } catch (const CommandLineParser::ParsingError& e) {
    spdlog::error(
        "Command line argument parrsing error captured. Error message: {}",
        e.what());
    throw(e);
  }

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
  
  // Load config once for reuse
  YAML::Node config_yaml;
  if (!loadConfig(config_path, config_yaml)) {
    spdlog::error("[TOGSim] Failed to load config file: {}", config_path);
    exit(1);
  }
  
  auto simulator = create_simulator(config_path);

  // Get trace file path
  cmd_parser.set_if_defined("models_list", &trace_file_path);

  if (!trace_file_path.empty()) {
    // Process trace file (unified mode: supports both FIFO and regular file)
    process_trace_file(simulator, trace_file_path, config_yaml);
    simulator->print_core_stat();
  } else {
    spdlog::error("No trace file provided. Use --models_list to specify trace file path.");
  }
  delete simulator;

  /* Simulation time measurement */
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  spdlog::info("Wall-clock time for simulation: {:2f} seconds", duration.count());
  return 0;
}
