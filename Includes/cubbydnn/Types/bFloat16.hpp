//
// Created by jwkim98 on 19/02/19.
//

#ifndef CUBBYDNN_FLOAT16_HPP
#define CUBBYDNN_FLOAT16_HPP

#include <cstdint>

namespace CubbyDNN
{
struct bFloat16
{
    uint16_t m_value = 0;
};

struct int8{
    uint8_t m_value;
};


}  // namespace CubbyDNN

#endif  // CUBBYDNN_FLOAT16_HPP
