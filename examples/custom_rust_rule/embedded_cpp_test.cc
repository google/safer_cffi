// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#include "security/ise_memory_safety/safer_cffi/examples/custom_rust_rule/embedded_cpp.h"

#include "testing/base/public/gunit.h"

namespace {

TEST(EmbeddedCppTest, BitCeil) { EXPECT_EQ(bit_ceil(5), 8); }

}  // namespace
