#pragma once
// UnitTable -- the per-tile Port admission table (--unit_table): what each port of
// each unit admitted during ONE run of a compute tile. Core sums it per completion;
// report() turns the sums into admitted / (capacity_per_cycle * cycles).
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

class UnitTable {
 public:
  struct Port {
    std::string unit;
    std::string port;
    bool primary;
    std::string unit_of_work;
    uint64_t capacity;
  };

  // Parse and validate `path`; on any error, log the offending line and exit(1).
  static std::unique_ptr<UnitTable> load(const std::string& path);

  size_t num_ports() const { return _ports.size(); }
  // Add one run of `tile_id` into `acc` (indexed like the ports); unknown tiles add nothing.
  void accumulate(int64_t tile_id, std::vector<uint64_t>& acc) const {
    if (tile_id < 0 || (size_t)tile_id >= _rows.size()) return;
    for (auto& [p, admitted] : _rows[tile_id]) acc[p] += admitted;
  }
  // Print one line per core and port, then a total per port; false if any line exceeds capacity.
  bool report(const std::vector<std::vector<uint64_t>>& per_core, uint64_t cycles) const;

 private:
  std::vector<Port> _ports;   // primary ports first, then the rest, each in first-seen order
  std::vector<std::vector<std::pair<size_t, uint64_t>>> _rows;   // tile_id -> (port, admitted)
};
