#include "UnitTable.h"

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <tuple>
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
  struct Row { uint64_t tile; size_t port; uint64_t admitted; };
  std::vector<Port> seen;
  std::map<std::pair<std::string, std::string>, std::pair<size_t, size_t>> index;
  std::vector<Row> rows;
  std::set<std::tuple<uint64_t, size_t>> tile_port;
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
    if (f.size() != 7)
      reject(path, lineno, fmt::format("expected 7 tab-separated fields "
             "(tile_id unit port primary unit_of_work capacity_per_cycle admitted), got {}", f.size()));
    for (int k : {1, 2, 4})
      if (f[k].empty() || f[k].find_first_of(" \t\r\v\f") != std::string::npos)
        reject(path, lineno, fmt::format("field {} '{}' must be a non-empty name without whitespace", k + 1, f[k]));
    uint64_t tile, primary, capacity, admitted;
    if (!parse_u64(f[0], tile)) reject(path, lineno, fmt::format("tile_id '{}' is not a non-negative integer", f[0]));
    if (!parse_u64(f[3], primary) || primary > 1) reject(path, lineno, fmt::format("primary '{}' must be 0 or 1", f[3]));
    if (!parse_u64(f[5], capacity)) reject(path, lineno, fmt::format("capacity_per_cycle '{}' is not a non-negative integer", f[5]));
    if (capacity == 0) reject(path, lineno, "capacity_per_cycle must be > 0");
    if (!parse_u64(f[6], admitted)) reject(path, lineno, fmt::format("admitted '{}' is not a non-negative integer", f[6]));
    Port p{f[1], f[2], primary == 1, f[4], capacity};
    auto key = std::make_pair(p.unit, p.port);
    auto it = index.find(key);
    if (it == index.end()) {
      it = index.emplace(key, std::make_pair(seen.size(), lineno)).first;
      seen.push_back(p);
    } else {
      const Port& q = seen[it->second.first];
      if (q.primary != p.primary || q.unit_of_work != p.unit_of_work || q.capacity != p.capacity)
        reject(path, lineno, fmt::format("unit {} port {} disagrees with line {} on primary/unit_of_work/capacity",
               p.unit, p.port, it->second.second));
    }
    if (!tile_port.emplace(tile, it->second.first).second)
      reject(path, lineno, fmt::format("second row for tile {} unit {} port {}", tile, p.unit, p.port));
    rows.push_back({tile, it->second.first, admitted});
  }
  std::map<std::string, std::pair<int, size_t>> primaries;
  for (auto& [key, at] : index) {
    auto& u = primaries.try_emplace(key.first, 0, at.second).first->second;
    u.first += seen[at.first].primary;
    u.second = std::min(u.second, at.second);
  }
  for (auto& [unit, u] : primaries)
    if (u.first != 1)
      reject(path, u.second, fmt::format("unit {} (first seen here) has {} primary ports; exactly one is required",
             unit, u.first));

  auto table = std::make_unique<UnitTable>();
  std::vector<size_t> order(seen.size());
  for (bool want : {true, false})
    for (size_t i = 0; i < seen.size(); i++)
      if (seen[i].primary == want) { order[i] = table->_ports.size(); table->_ports.push_back(seen[i]); }
  for (auto& r : rows) {
    if (r.tile >= table->_rows.size()) table->_rows.resize(r.tile + 1);
    table->_rows[r.tile].emplace_back(order[r.port], r.admitted);
  }
  return table;
}

bool UnitTable::report(const std::vector<std::vector<uint64_t>>& per_core, uint64_t cycles) const {
  bool ok = true;
  auto line = [&](const std::string& who, const Port& p, uint64_t admitted, unsigned __int128 budget) {
    double pct = budget ? 100.0 * (double)admitted / (double)budget : 0.0;
    spdlog::info("{} : Unit {} port {} utilization(%): {:.2f}, admitted: {} {}, capacity: {} per cycle, primary: {}",
                 who, p.unit, p.port, pct, admitted, p.unit_of_work, p.capacity, p.primary ? 1 : 0);
    if ((unsigned __int128)admitted > budget) {
      spdlog::error("{} : Unit {} port {} admitted {} {} exceeds capacity {} per cycle x {} cycles",
                    who, p.unit, p.port, admitted, p.unit_of_work, p.capacity,
                    (uint64_t)(budget / p.capacity));
      ok = false;
    }
  };
  for (size_t c = 0; c < per_core.size(); c++)
    for (size_t i = 0; i < _ports.size(); i++)
      line(fmt::format("Core [{}]", c), _ports[i], per_core[c][i],
           (unsigned __int128)_ports[i].capacity * cycles);
  for (size_t i = 0; i < _ports.size(); i++) {
    uint64_t total = 0;
    for (auto& acc : per_core) total += acc[i];
    line("Total", _ports[i], total,
         (unsigned __int128)_ports[i].capacity * cycles * per_core.size());
  }
  return ok;
}
