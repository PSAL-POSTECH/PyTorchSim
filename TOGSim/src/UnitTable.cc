#include "UnitTable.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <spdlog/spdlog.h>

namespace {

[[noreturn]] void reject(const std::string& path, size_t lineno, const std::string& what) {
  spdlog::error("[TOGSim] --unit_table {} line {}: {}", path, lineno, what);
  exit(EXIT_FAILURE);
}

bool parse_u64(const std::string& s, uint64_t& out) {
  auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), out);
  return !s.empty() && ec == std::errc() && p == s.data() + s.size();
}

}  // namespace

std::unique_ptr<UnitTable> UnitTable::load(const std::string& path) {
  std::ifstream in(path);
  if (!in.is_open()) {
    spdlog::error("[TOGSim] --unit_table {}: cannot open", path);
    exit(EXIT_FAILURE);
  }
  auto table = std::make_unique<UnitTable>();
  std::map<std::string, std::pair<size_t, size_t>> index;
  std::set<std::pair<uint64_t, size_t>> tile_unit;
  std::string line;
  size_t lineno = 0;
  while (std::getline(in, line)) {
    lineno++;
    if (line.empty()) continue;
    std::vector<std::string> f;
    std::stringstream ss(line);
    std::string tok;
    while (std::getline(ss, tok, '\t')) f.push_back(tok);
    if (line.back() == '\t') f.push_back("");
    if (f.size() != 5)
      reject(path, lineno, fmt::format("expected 5 tab-separated fields "
             "(tile_id unit unit_of_work capacity_per_cycle admitted), got {}", f.size()));
    for (int k : {1, 2})
      if (f[k].empty() || f[k].find_first_of(" \t\r\v\f") != std::string::npos)
        reject(path, lineno, fmt::format("field {} '{}' must be a non-empty name without whitespace", k + 1, f[k]));
    uint64_t tile, capacity, admitted;
    if (!parse_u64(f[0], tile)) reject(path, lineno, fmt::format("tile_id '{}' is not a non-negative integer", f[0]));
    if (!parse_u64(f[3], capacity)) reject(path, lineno, fmt::format("capacity_per_cycle '{}' is not a non-negative integer", f[3]));
    if (capacity == 0) reject(path, lineno, "capacity_per_cycle must be > 0");
    if (!parse_u64(f[4], admitted)) reject(path, lineno, fmt::format("admitted '{}' is not a non-negative integer", f[4]));
    auto it = index.find(f[1]);
    if (it == index.end()) {
      it = index.emplace(f[1], std::make_pair(table->_units.size(), lineno)).first;
      table->_units.push_back({f[1], f[2], capacity});
    } else {
      const Unit& q = table->_units[it->second.first];
      if (q.unit_of_work != f[2] || q.capacity != capacity)
        reject(path, lineno, fmt::format("unit {} disagrees with line {} on unit_of_work/capacity",
               f[1], it->second.second));
    }
    const size_t u = it->second.first;
    if (!tile_unit.emplace(tile, u).second)
      reject(path, lineno, fmt::format("second row for tile {} unit {}", tile, f[1]));
    if (tile >= table->_rows.size()) table->_rows.resize(tile + 1);
    table->_rows[tile].emplace_back(u, admitted);
  }
  return table;
}

bool UnitTable::print(const std::string& who, size_t u, uint64_t admitted, uint64_t cycles, bool check) const {
  const Unit& unit = _units[u];
  const double utilized = (double)admitted / (double)unit.capacity;
  const double pct = cycles ? 100.0 * utilized / (double)cycles : 0.0;
  const int64_t active = std::llround(utilized);
  spdlog::info("{} : {} utilization(%): {:.2f}, active_cycles: {}, idle_cycles: {}",
               who, unit.name, pct, active, (int64_t)cycles - active);
  if (!check || (unsigned __int128)admitted <= (unsigned __int128)unit.capacity * cycles) return true;
  spdlog::error("{} : {} admitted {} {} exceeds capacity {} per cycle x {} cycles",
                who, unit.name, admitted, unit.unit_of_work, unit.capacity, cycles);
  return false;
}

bool UnitTable::print_spread(const std::string& who, size_t u, double admitted, uint64_t cycles) const {
  const Unit& unit = _units[u];
  const double utilized = admitted / (double)unit.capacity;
  const double pct = cycles ? 100.0 * utilized / (double)cycles : 0.0;
  const int64_t active = std::llround(utilized);
  spdlog::info("{} : {} utilization(%): {:.2f}, active_cycles: {}, idle_cycles: {}",
               who, unit.name, pct, active, std::max<int64_t>((int64_t)cycles - active, 0));
  if (utilized <= (double)cycles * (1.0 + 1e-9)) return true;
  spdlog::error("{} : {} spread {} {} over {} cycles exceeds capacity {} per cycle",
                who, unit.name, admitted, unit.unit_of_work, cycles, unit.capacity);
  return false;
}
