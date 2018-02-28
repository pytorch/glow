#include <fstream>
#include <iostream>
#include <sstream>

void genBackendFwdDecl(std::string backendName, std::string backendPreproc,
                       std::string whichGen, std::ofstream &strm) {
  strm << "#if " + backendPreproc + " == 1\n";
  strm << "void add" << whichGen << "For" + backendName + "(Builder &BB);\n";
  strm << "#endif // " + backendPreproc + "\n\n";
}

void genBackendAdd(std::string backendName, std::string backendPreproc,
                   std::string whichGen, std::ofstream &strm) {
  strm << "#if " + backendPreproc + " == 1\n";
  strm << "  add" << whichGen << "For" + backendName + "(BB);\n";
  strm << "#endif // " + backendPreproc + "\n\n";
}

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr
        << "Usage: " << argv[0]
        << " [path/to/output/node_file.cpp] [path/to/output/instr_file.cpp]\n";
    return -1;
  }

  std::cout << "Writing backend-specific node hooks to:\n\t" << argv[1] << "\n";
  std::cout << "Writing backend-specific instr hooks to:\n\t" << argv[2]
            << "\n";

  std::ofstream nodeStream(argv[1]);
  std::ofstream instrStream(argv[2]);

  nodeStream << "class Builder;\n\n";
  instrStream << "class Builder;\n\n";

#define DEF_BACKEND(be_name_, be_preproc_)                                     \
  genBackendFwdDecl(#be_name_, #be_preproc_, "Nodes", nodeStream);             \
  genBackendFwdDecl(#be_name_, #be_preproc_, "Instrs", instrStream);
#include "Backends.def"
#undef DEF_BACKEND

  nodeStream << "\nvoid addBackendSpecificNodes(Builder &BB) {\n\n";
  instrStream << "\nvoid addBackendSpecificInstrs(Builder &BB) {\n\n";

#define DEF_BACKEND(be_name_, be_preproc_)                                     \
  genBackendAdd(#be_name_, #be_preproc_, "Nodes", nodeStream);                 \
  genBackendAdd(#be_name_, #be_preproc_, "Instrs", instrStream);
#include "Backends.def"
#undef DEF_BACKEND

  nodeStream << "}\n";
  instrStream << "}\n";

  return 0;
}
