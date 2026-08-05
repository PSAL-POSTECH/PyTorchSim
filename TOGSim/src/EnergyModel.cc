#include "EnergyModel.h"

#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include <array>
#include <cmath>
#include <stdexcept>

#include "fmt/core.h"

double DramEnergyCosts::transfer_pj_per_bit_total() const {
  double total = 0.0;
  for (const auto& [name, pj] : transfer_pj_per_bit)
    total += pj;
  return total;
}

std::string DramEnergyCosts::transfer_breakdown() const {
  std::string out;
  for (const auto& [name, pj] : transfer_pj_per_bit) {
    if (!out.empty())
      out += " + ";
    out += fmt::format("{} {:.2f}", name, pj);
  }
  return out;
}

DramEnergyCosts load_dram_energy_costs(const std::string& cost_table_path) {
  YAML::Node table;
  try {
    table = YAML::LoadFile(cost_table_path);
  } catch (const std::exception& e) {
    throw std::runtime_error(
        fmt::format("[Config/Energy] Failed to load energy cost table \"{}\": {}", cost_table_path, e.what()));
  }

  DramEnergyCosts parsed;
  parsed.path = cost_table_path;
  parsed.name = table["name"] ? table["name"].as<std::string>() : "unnamed";

  const YAML::Node dram = table["offchip_dram"];
  if (!dram)
    throw std::runtime_error(
        fmt::format("[Config/Energy] Energy cost table \"{}\" has no offchip_dram section", cost_table_path));

  if (!dram["row_activation_pj"])
    throw std::runtime_error(
        fmt::format("[Config/Energy] Energy cost table \"{}\": offchip_dram.row_activation_pj is required", cost_table_path));
  parsed.row_activation_pj = dram["row_activation_pj"].as<double>();

  const YAML::Node per_bit = dram["transfer_pj_per_bit"];
  if (!per_bit || !per_bit.IsMap() || per_bit.size() == 0)
    throw std::runtime_error(fmt::format(
        "[Config/Energy] Energy cost table \"{}\": offchip_dram.transfer_pj_per_bit must be a non-empty map", cost_table_path));
  for (const auto& term : per_bit)
    parsed.transfer_pj_per_bit.emplace_back(term.first.as<std::string>(), term.second.as<double>());

  spdlog::info("[Config/Energy] Loaded energy cost table \"{}\" from {}", parsed.name, cost_table_path);
  return parsed;
}

DramEnergy compute_dram_energy(const DramEnergyCosts& costs, const DramEnergyCounters& counters) {
  DramEnergy energy;
  energy.activation_pj = static_cast<double>(counters.row_activations) * costs.row_activation_pj;
  energy.transfer_pj = static_cast<double>(counters.transferred_bits()) * costs.transfer_pj_per_bit_total();
  energy.total_pj = energy.activation_pj + energy.transfer_pj;
  return energy;
}

namespace {
/** Pick the unit whose scale keeps `value / scale` in [1, 1000), or the smallest unit. */
std::string scale_units(double value, const std::vector<std::pair<double, const char*>>& units) {
  for (const auto& [scale, suffix] : units) {
    if (std::fabs(value) >= scale)
      return fmt::format("{:.3f} {}", value / scale, suffix);
  }
  return fmt::format("{:.3f} {}", value / units.back().first, units.back().second);
}
}  // namespace

std::string format_energy(double pj) {
  return scale_units(pj, {{1e12, "J"}, {1e9, "mJ"}, {1e6, "uJ"}, {1e3, "nJ"}, {1.0, "pJ"}});
}

std::string format_power(double watts) {
  return scale_units(watts, {{1.0, "W"}, {1e-3, "mW"}, {1e-6, "uW"}});
}

std::string format_time(double seconds) {
  return scale_units(seconds, {{1.0, "s"}, {1e-3, "ms"}, {1e-6, "us"}, {1e-9, "ns"}});
}
