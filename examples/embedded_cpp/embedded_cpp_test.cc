// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#include "security/ise_memory_safety/safer_cffi/examples/embedded_cpp/embedded_cpp.h"

#include "testing/base/public/gunit.h"

namespace {

TEST(EmbeddedCppTest, Fibonacci) {
  EXPECT_EQ(fibonacci(0), 0);
  EXPECT_EQ(fibonacci(1), 1);
  EXPECT_EQ(fibonacci(2), 1);
  EXPECT_EQ(fibonacci(10), 55);
  EXPECT_EQ(fibonacci(20), 6765);
}

TEST(EmbeddedCppTest, IsFibonacci) {
  EXPECT_TRUE(is_fibonacci(0));
  EXPECT_TRUE(is_fibonacci(1));
  EXPECT_TRUE(is_fibonacci(2));
  EXPECT_TRUE(is_fibonacci(3));
  EXPECT_FALSE(is_fibonacci(4));
  EXPECT_TRUE(is_fibonacci(5));
  EXPECT_FALSE(is_fibonacci(6));
  EXPECT_TRUE(is_fibonacci(55));
  EXPECT_FALSE(is_fibonacci(56));
  EXPECT_TRUE(is_fibonacci(6765));
  EXPECT_FALSE(is_fibonacci(-1));
}

}  // namespace
