/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "gtest/gtest.h"

#include "peglib.h"
#include <assert.h>
#include <iostream>
#include <vector>

class InstructionParser {
public:
  virtual void parseInstruction() = 0;
  virtual void debug() = 0;
};

class ConvolutionParser : public InstructionParser {
public:
  ConvolutionParser(std::string &str) : instr_str_(str) {
    syntax_ = R"(
      ROOT      <- INSTR _ '=' _ 'convolution' _
                   '[' KERNEL _ STRIDE _ PAD _ GROUP ']' _ 
                   '@out' _ OUT ',' _ '@in' _ IN1 ',' _
                   '@in' _ IN2 ',' _ '@in' _ IN3
      INSTR     <- '%'[a-z0-9]+
      OUT       <- '%'[a-z0-9]+
      IN1       <- '%'[a-z0-9]+
      IN2       <- '%'[a-z0-9]+
      IN3       <- '%'[a-z0-9]+
      KERNEL    <- [0-9]+
      STRIDE    <- [0-9]+
      PAD       <- [0-9]+
      GROUP     <- [0-9]+
      _         <- [ \t\r\n]
    )";
  }

  ~ConvolutionParser() {}

  void parseInstruction() override {
    peg::parser pg(syntax_.c_str());

    pg["INSTR"] = [&](const peg::SemanticValues &sv) {
      auto token = sv.token();
      instr_name_ = token;
    };

    pg["OUT"] = [&](const peg::SemanticValues &sv) {
      auto token = sv.token();
      out_ = token;
    };

    pg["IN1"] = [&](const peg::SemanticValues &sv) {
      auto token = sv.token();
      in1_ = token;
    };

    pg["IN2"] = [&](const peg::SemanticValues &sv) {
      auto token = sv.token();
      in2_ = token;
    };

    pg["IN3"] = [&](const peg::SemanticValues &sv) {
      auto token = sv.token();
      in3_ = token;
    };

    pg["KERNEL"] = [&](const peg::SemanticValues &sv) {
      auto token = sv.token();
      kernel_ = token;
    };

    pg["STRIDE"] = [&](const peg::SemanticValues &sv) {
      auto token = sv.token();
      stride_ = token;
    };

    pg["PAD"] = [&](const peg::SemanticValues &sv) {
      auto token = sv.token();
      pad_ = token;
    };

    pg["GROUP"] = [&](const peg::SemanticValues &sv) {
      auto token = sv.token();
      group_ = token;
    };

    pg.enable_packrat_parsing();

    pg.parse(instr_str_.c_str());
  }

  void debug() override {
    std::cout << "INSTR " << instr_name_ << std::endl;
    std::cout << "OUT " << out_ << std::endl;
    std::cout << "IN1 " << in1_ << std::endl;
    std::cout << "IN2 " << in2_ << std::endl;
    std::cout << "IN3 " << in3_ << std::endl;
    std::cout << "KERNEL " << kernel_ << std::endl;
    std::cout << "STRIDE " << stride_ << std::endl;
    std::cout << "PAD " << pad_ << std::endl;
    std::cout << "GROUP " << group_ << std::endl;
  }

private:
  std::string instr_str_;
  std::string syntax_;

  std::string instr_name_;
  std::string out_;
  std::string in1_;
  std::string in2_;
  std::string in3_;

  std::string kernel_;
  std::string stride_;
  std::string pad_;
  std::string group_;
};

class AllocParser : public InstructionParser {
public:
  AllocParser(std::string &str) : instr_str_(str) {
    syntax_ = R"(
      ROOT      <- INSTR _ '=' _ 'alloc' _ ELEMTYPE '[' DIM ']'
      INSTR     <- '%'[a-z0-9]+
      ELEMTYPE  <- 'float16' / 'float' / 'i8' /
                   'i16' / 'i32' / 'index32' / 'index64'
      DIM       <- [0-9]+ (_ 'x' _ [0-9]+)* 
      _         <- [ \t\r\n]
    )";
  }

  ~AllocParser() {}

  void parseInstruction() override {
    peg::parser pg(syntax_.c_str());

    pg["INSTR"] = [&](const peg::SemanticValues &sv) {
      auto token = sv.token();
      instr_name_ = token;
    };

    pg["ELEMTYPE"] = [&](const peg::SemanticValues &sv) {
      auto token = sv.token();
      elem_type_ = token;
    };

    pg["DIM"] = [&](const peg::SemanticValues &sv) {
      auto token = sv.token();
      dim_ = token;
    };

    pg.enable_packrat_parsing();

    pg.parse(instr_str_.c_str());
  }

  void debug() override {
    std::cout << "INSTR " << instr_name_ << std::endl;
    std::cout << "ELEMTYPE " << elem_type_ << std::endl;
    std::cout << "DIM " << dim_ << std::endl;
  }

private:
  std::string instr_str_;
  std::string syntax_;

  std::string instr_name_;
  std::string elem_type_;
  std::string dim_;
};

class InstructionTypeParser : public InstructionParser {
public:
  InstructionTypeParser(std::string &str) : instr_str_(str) {
    syntax_ = R"(
      ROOT  <- INSTR _ '=' _ TYPE _ *
      INSTR <- '%'[a-z0-9]+
      TYPE  <- [a-z]+
      _     <- [ \t\r\n]
    )";
  }

  ~InstructionTypeParser() {}

  void parseInstruction() override {
    peg::parser pg(syntax_.c_str());

    pg["INSTR"] = [&](const peg::SemanticValues &sv) {
      auto token = sv.token();
      instr_name_ = token;
    };

    pg["TYPE"] = [&](const peg::SemanticValues &sv) {
      auto token = sv.token();
      instr_type_ = token;
    };

    pg.enable_packrat_parsing();

    pg.parse(instr_str_.c_str());
  }

  std::string type() { return instr_type_; }

  void debug() override {
    std::cout << "INSTR " << instr_name_ << std::endl;
    std::cout << "TYPE " << instr_type_ << std::endl;
  }

private:
  std::string instr_str_;
  std::string syntax_;

  std::string instr_name_;
  std::string instr_type_;
};

TEST(LowLevelIRLoader, test) {
  std::vector<std::string> input = {
      "%conv = convolution [5 1 2 16] @out %alloc, @in %input, @in %filter, "
      "@in %bias0",
      "%alloc = alloc float[8 x 28 x 28 x 16]"};

  for (auto &s : input) {
    InstructionTypeParser type_parser(s);
    type_parser.parseInstruction();
    if (type_parser.type() == "convolution") {
      ConvolutionParser conv_parser(s);
      conv_parser.parseInstruction();
      conv_parser.debug();
    } else if (type_parser.type() == "alloc") {
      AllocParser alloc_parser(s);
      alloc_parser.parseInstruction();
      alloc_parser.debug();
    }
    std::cout << std::endl;
  }
}
