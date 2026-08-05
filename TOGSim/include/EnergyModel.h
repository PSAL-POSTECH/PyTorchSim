#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

/**
 * Off-chip DRAM energy constants, read from the `offchip_dram` section of the
 * energy cost table selected by the simulation config's
 * `energy_cost_table_path`. All values are in pJ.
 */
struct DramEnergyCosts {
  std::string name;
  std::string path;
  double row_activation_pj = 0.0;
  /* Per-bit terms kept separate so the breakdown stays visible in the report. */
  std::vector<std::pair<std::string, double>> transfer_pj_per_bit;

  double transfer_pj_per_bit_total() const;
  /** "dram 1.51 + io 1.17 + phy 0.80" */
  std::string transfer_breakdown() const;
};

/** Raw activity a DRAM model reports for energy accounting, aggregated over all channels. */
struct DramEnergyCounters {
  uint64_t row_activations = 0;
  /* Column commands served. Each moves one burst, never a partial one. */
  uint64_t transactions = 0;
  /* Burst size, i.e. the DRAM model's transaction granularity. */
  uint32_t bytes_per_transaction = 0;
  /* False when the DRAM model tracks no row state, so activation energy is unknown. */
  bool available = false;

  uint64_t transferred_bits() const {
    return transactions * static_cast<uint64_t>(bytes_per_transaction) * 8ull;
  }
};

struct DramEnergy {
  double activation_pj = 0.0;
  double transfer_pj = 0.0;
  double total_pj = 0.0;
};

DramEnergyCosts load_dram_energy_costs(const std::string& cost_table_path);

DramEnergy compute_dram_energy(const DramEnergyCosts& costs, const DramEnergyCounters& counters);

/** Scale to the largest unit that keeps the value >= 1, e.g. "1.131 mJ". */
std::string format_energy(double pj);
std::string format_power(double watts);
std::string format_time(double seconds);
