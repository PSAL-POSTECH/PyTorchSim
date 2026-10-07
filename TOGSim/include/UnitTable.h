#pragma once
// UnitTable -- the per-tile unit admission table (--unit_table): what each unit admitted, at
// its primary port, during ONE run of a compute tile. Core sums it per completion and prints
// each unit as utilized cycles = admitted / capacity_per_cycle over its cycles.
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

class UnitTable {
 public:
  struct Unit {
    std::string name;
    std::string unit_of_work;
    uint64_t capacity;
  };

  // Parse and validate `path`; on any error, log the offending line and exit(1).
  static std::unique_ptr<UnitTable> load(const std::string& path);

  size_t num_units() const { return _units.size(); }
  // Add one run of `tile_id` into `acc` (indexed like the units); unknown tiles add nothing.
  void accumulate(int64_t tile_id, std::vector<uint64_t>& acc) const {
    if (tile_id < 0 || (size_t)tile_id >= _rows.size()) return;
    for (auto& [u, admitted] : _rows[tile_id]) acc[u] += admitted;
  }
  // Print '<who> : <unit> utilization(%): ..., active_cycles: U, idle_cycles: I' for unit `u`;
  // false if `check` and the unit admitted more than its capacity allows in `cycles`.
  bool print(const std::string& who, size_t u, uint64_t admitted, uint64_t cycles, bool check) const;

 private:
  std::vector<Unit> _units;   // in first-seen order
  std::vector<std::vector<std::pair<size_t, uint64_t>>> _rows;   // tile_id -> (unit, admitted)
};
