#ifndef INSTRUCTION_H
#define INSTRUCTION_H


class Model {
  public:
    Model() {}
    std::string get_name() { return _name; }

  protected:
    std::string _name;
};

#endif