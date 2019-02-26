//
// Created by jwkim98 on 19/02/19.
//

#include "cubbydnn/Types/TypeCast.hpp"

namespace CubbyDNN
{
void TruncateFloatToBfloat16(const float* floatInput, uint16_t* Bfloat16Output,
                             size_t count)
{
    const uint16_t* temp = reinterpret_cast<const uint16_t*>(floatInput);
    uint16_t* tempResult = Bfloat16Output;
    size_t Count16 = 0;
    size_t Count32 = 0;
    if (ByteOrder::LittleEndian)
    {
        for (; count > 0; Count16 += 1, Count32 += 2, count -= 1)
        {
            *(tempResult + Count16) = (temp + Count32)[1];
        }
    }
    else
    {
        for (; count > 0; Count16 += 1, Count32 += 2, count -= 1)
        {
            *(tempResult + Count16) = (temp + Count32)[0];
        }
    }
}

void TruncateBfloat16ToFloat(const uint16_t* Bfloat16Input, float* floatOutput,
                             size_t count)
{
    uint16_t* tempResult = reinterpret_cast<uint16_t*>(floatOutput);
    const uint16_t* temp = Bfloat16Input;
    size_t Count16 = 0;
    size_t Count32 = 0;
    if (ByteOrder::LittleEndian)
    {
        for (; count > 0; Count16 += 1, Count32 += 2, count -= 1)
        {
            (tempResult + Count32)[1] = *(temp + Count16);
            (tempResult + Count32)[0] = 0;
        }
    }
    else
    {
        for (; count > 0; Count16 += 1, Count32 += 2, count -= 1)
        {
            (tempResult + Count32)[0] = *(temp + Count16);
            (tempResult + Count32)[1] = 0;
        }
    }
}
}  // namespace CubbyDNN
