#ifndef INSTRUCTION_H
#define INSTRUCTION_H

#include "Common.h"
#include "helper/HelperFunctions.h"

class Model {
  public:
    Model() {}
    std::string get_name() { return _name; }

  protected:
    std::string _name;
};

#endif