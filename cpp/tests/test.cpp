// Copyright © 2023 Apple Inc.

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest/doctest.h"

#include <cstdlib>

int main(int argc, char** argv) {
  doctest::Context context;
  context.applyCommandLine(argc, argv);
  return context.run();
}