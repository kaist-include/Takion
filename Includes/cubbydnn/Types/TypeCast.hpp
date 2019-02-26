//
// Created by jwkim98 on 19/02/19.
//

#ifndef CUBBYDNN_TYPECAST_HPP
#define CUBBYDNN_TYPECAST_HPP

#include <cstdio>
#include "Endian.hpp"
#include "bFloat16.hpp"
namespace CubbyDNN
{
void FloatTobFloat16(const float* val, bFloat16* result, size_t count = 1);

void bFloat16ToFloat(const bFloat16* val, float* result, size_t count = 1);
}  // namespace CubbyDNN

#endif  // CUBBYDNN_TYPECAST_HPP
