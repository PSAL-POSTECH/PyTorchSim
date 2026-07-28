#ifndef INTERCONNECT_H
#define INTERCONNECT_H
#include "DMA.h"
#include "booksim2/Interconnect.hpp"
#include <cmath>
#include <iostream>
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;

class Interconnect {
 public:
  virtual ~Interconnect() = default;
  virtual bool running() = 0;
  virtual void cycle() = 0;
  virtual void push(uint32_t src, uint32_t dest, mem_fetch* request) = 0;
  virtual bool is_full(uint32_t src, mem_fetch* request) = 0;
  virtual bool is_empty(uint32_t nid) = 0;
  virtual mem_fetch* top(uint32_t nid) = 0;
  virtual void pop(uint32_t nid) = 0;
  virtual void print_stats() = 0;

 protected:
  SimulationConfig _config;
  uint32_t _n_nodes;
  uint64_t _cycles;
};

// Simple without conflict interconnect
class SimpleInterconnect : public Interconnect {
 public:
  SimpleInterconnect(SimulationConfig config);
  virtual bool running() override;
  virtual void cycle() override;
  virtual void push(uint32_t src, uint32_t dest,
                    mem_fetch* request) override;
  virtual bool is_full(uint32_t src, mem_fetch* request) override;
  virtual bool is_empty(uint32_t nid) override;
  virtual mem_fetch* top(uint32_t nid) override;
  virtual void pop(uint32_t nid) override;
  virtual void print_stats() override {}

 private:
  uint32_t _latency;
  double _bandwidth;
  uint32_t _rr_start;
  uint32_t _buffer_size;

  struct Entity {
    cycle_type finish_cycle;
    uint32_t dest;
    mem_fetch* access;
  };

  std::vector<std::vector<std::queue<Entity>>> _in_buffers;
  std::vector<std::queue<mem_fetch*>> _out_buffers;
  std::vector<int> _rr_next_src;
  std::vector<bool> _busy_node;
};

class Booksim2Interconnect : public Interconnect {
 public:
  Booksim2Interconnect(SimulationConfig config);
  virtual bool running() override;
  virtual void cycle() override;
  virtual void push(uint32_t src, uint32_t dest,
                    mem_fetch* request) override;
  virtual bool is_full(uint32_t src, mem_fetch* request) override;
  virtual bool is_empty(uint32_t nid) override;
  virtual mem_fetch* top(uint32_t nid) override;
  virtual void pop(uint32_t nid) override;
  virtual void print_stats() override;
  void print_config(std::string config_path);

 private:
  uint32_t _ctrl_size;
  std::string _config_path;
  std::unique_ptr<booksim2::Interconnect> _booksim;

  booksim2::Interconnect::Type get_booksim_type(mem_fetch* access);
  uint32_t get_packet_size(mem_fetch* access);
};
#endif