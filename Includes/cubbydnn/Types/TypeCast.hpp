//
// Created by jwkim98 on 19/02/19.
//

#ifndef CUBBYDNN_TYPECAST_HPP
#define CUBBYDNN_TYPECAST_HPP

#include <cstdio>
#include <cstdint>
#include "Endian.hpp"
namespace CubbyDNN
{
void TruncateFloatToBfloat16(const float* floatInput, uint16_t* Bfloat16Output, size_t count = 1);

void TruncateBfloat16ToFloat(const uint16_t* Bfloat16Input, float* floatOutput, size_t count = 1);
}  // namespace CubbyDNN

#endif  // CUBBYDNN_TYPECAST_HPP
