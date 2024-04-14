#include <fstream>
#include <chrono>
#include <filesystem>

#include "Simulator.h"
#include "TileGraphParser.h"
#include "helper/CommandLineParser.h"

namespace fs = std::filesystem;
namespace po = boost::program_options;

int main(int argc, char** argv) {
  auto start = std::chrono::high_resolution_clock::now();
  // parse command line argumnet
  CommandLineParser cmd_parser = CommandLineParser();
  cmd_parser.add_command_line_option<std::string>(
      "config", "Path for hardware configuration file");
  cmd_parser.add_command_line_option<std::string>(
      "models_list", "Path for the models list file");
  cmd_parser.add_command_line_option<std::string>(
      "log_level", "Set for log level [trace, debug, info], default = info");
  cmd_parser.add_command_line_option<std::string>(
      "mode", "choose one_model or two_model");

  try {
    cmd_parser.parse(argc, argv);
  } catch (const CommandLineParser::ParsingError& e) {
    spdlog::error(
        "Command line argument parrsing error captured. Error message: {}",
        e.what());
    throw(e);
  }
  char* onnxim_path_env = std::getenv("ONNXIM_HOME");
  std::string onnxim_path = onnxim_path_env != NULL?
    std::string(onnxim_path_env) : std::string("./");

  std::string model_base_path = fs::path(onnxim_path).append("models");
  std::string level = "info";
  cmd_parser.set_if_defined("log_level", &level);
  if (level == "trace")
    spdlog::set_level(spdlog::level::trace);
  else if (level == "debug")
    spdlog::set_level(spdlog::level::debug);
  else if (level == "info")
    spdlog::set_level(spdlog::level::info);

  std::string config_path;
  cmd_parser.set_if_defined("config", &config_path);

  json config_json;
  std::ifstream config_file(config_path);
  config_file >> config_json;
  config_file.close();
  SimulationConfig config = initialize_config(config_json);

  std::string onnx_path;
  cmd_parser.set_if_defined("models_list", &onnx_path);

  auto simulator = std::make_unique<Simulator>(config);
  auto graph_praser = TileGraphParser(onnx_path);
  std::unique_ptr<TileGraph>& tile_graph = graph_praser.get_tile_graph();
  spdlog::info("Register graph: {}", onnx_path);

  simulator->schedule_graph(std::move(tile_graph));
  simulator->run_simulator();
  /* Simulation time measurement */
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  spdlog::info("Simulation time: {:2f} seconds", duration.count());
  return 0;
}
