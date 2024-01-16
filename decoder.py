import os

BRANCHES = ["bne", "beq", "blt", "bge", "bltu", "bgeu",
            "ble", "bgt" "bleu", "bgtu", "beqz", "bnez",
            "j", "jr", "jal", "jalr"]
LOADS = ["vle32.v", "vle64.v", "vle8.v", "vle16.v",
         "vl2re32.v", "vl2re64.v", "vl2re8.v", "vl2re16.v"]
STORES = ["vse32.v", "vse64.v", "vse8.v", "vse16.v",
          "vs2r.v", "vs2re32.v", "vs2re64.v", "vs2re8.v", "vs2re16.v"]

dummy_gem5 = lambda x: 5 # FIXME. dummy function

class node:
  def __init__(self, type, inst, num, cycle=0):
    self.type = type # Load, Compute, Store
    self.parents = []
    self.children = []
    self.cycle = cycle

    self.inst = inst
    self.id = num
    self.name = "node" + str(self.id)
    self.visited = False # for BFS

  def add_child(self, child):
    self.children.append(child)

  def search_reg(self, reg): # TODO: extend to mutiple instruction
    if reg in self.inst[1]:
      return True
    else:
      return False


class Decoder:
  def __init__(self):
    self.dir = "./tmp"
    self.node_list = []
    self.node_num = 1
    self.root = node("Dummy", ["nop"], 0) # dummy node for root nodes

  def __call__(self, file):
    self.decode(file)

  # remove loops (branch) from the code to make single tile
  def preprocess(self, file):
    with open(os.path.join(self.dir, file), 'r') as asm:
      asm_code = asm.read()
    asm_code = asm_code.split('\n')
    # remove branch lines
    asm_code = [line for line in asm_code if not any(branch in line for branch in BRANCHES)]
    asm_code = '\n'.join(asm_code)
    asm.close()
    return asm_code

  def node_gen(self, code, cycle=0):
    code = code.split('\n')
    code = [line.split() for line in code]
    code = [line for line in code if "nop" not in line]
    code = [line for line in code if "ret" not in line]
    for line in code:
      cmd = line[0]
      if cmd in LOADS:
        self.node_list.append(node("Load", line, self.node_num))
        self.node_num += 1
      elif cmd in STORES:
        self.node_list.append(node("Store", line, self.node_num))
        self.node_num += 1
      elif ".v" in cmd:
        self.node_list.append(node("Compute", line, self.node_num, cycle))
        self.node_num += 1

  # This function assumes that the instructions are in order (load -> compute -> store)
  def connect_node(self, node1, node2):
    node1.add_child(node2)
    node2.parents.append(node1)

  def build_nodes(self):
    for node in self.node_list:
      if (node.type != "Store"):
        for node2 in self.node_list[node.id:]:
            if node2.type == "Load":
              if node.search_reg(node2.inst[2]):
                self.connect_node(node, node2)
            elif node2.type == "Store":
                if node.search_reg(node2.inst[1]):
                  self.connect_node(node, node2)
            else: # node2.type == "Compute"
              if node.search_reg(node2.inst[2]) or node.search_reg(node2.inst[3]):
                self.connect_node(node, node2)
            if (node2.type != "Store" and node.inst[1] == node2.inst[1]):
              break
      if (len(node.parents) == 0):
        self.connect_node(self.root, node) # connect root nodes to dummy node
    self.node_list.append(self.root)

  def dump_graph_info(self, file, node):
    file.write(str(node.name) + " type: " + str(node.type) + " cycle: " + str(node.cycle))
    file.write(" parent: " + str(len(node.parents)) + " ")
    file.write(", ".join([str(parent.name) for parent in node.parents]))
    file.write(" child: " + str(len(node.children)) + " ")
    file.write(", ".join([str(child.name) for child in node.children]))
    file.write("\n")

  def BFS(self, file):
    queue = [self.root]
    self.root.visited = True
    while queue:
      s = queue.pop(0)
      self.dump_graph_info(file, s)
      for node in s.children:
        if node.visited == False:
          queue.append(node)
          node.visited = True

  def decode(self, file):
    code = self.preprocess(file)
    cycle = dummy_gem5(code) # FIXME. dummy function and binary
    self.node_gen(code, cycle)
    self.build_nodes()
    self.graph_dump()
    self.print_graph() # FIXME. for debug

  def graph_dump(self):
    with open(os.path.join(self.dir, "graph.g"), 'w') as f:
      self.BFS(f)

  def print_graph(self):
    for node in self.node_list:
      print(node.name, "type:", node.type, "inst:", " ".join(node.inst))
      print("[parents]", ", ".join([parent.name for parent in node.parents]))
      print("[children]", ", ".join([child.name for child in node.children]))

if __name__ == '__main__':
  decoder = Decoder()
  decoder('vectoradd.s')