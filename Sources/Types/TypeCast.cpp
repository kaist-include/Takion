//
// Created by jwkim98 on 19/02/19.
//

#include "cubbydnn/Types/TypeCast.hpp"

namespace CubbyDNN{
    void FloatTobFloat16(const float* val, bFloat16* result, size_t count = 1)
    {
        const uint16_t* temp = reinterpret_cast<const uint16_t*>(val);
        uint16_t* tempResult = reinterpret_cast<uint16_t*>(result);
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

    void bFloat16ToFloat(const bFloat16* val, float* result, size_t count = 1){
        uint16_t* tempResult = reinterpret_cast<uint16_t*>(result);
        const uint16_t* temp = reinterpret_cast<const uint16_t*>(val);
        size_t Count16 = 0;
        size_t Count32 = 0;
        if(ByteOrder::LittleEndian){
            for(; count > 0; Count16 += 1, Count32 += 2, count -= 1){
                (tempResult+Count32)[1] = *(temp + Count16);
                (tempResult+Count32)[0] = 0;
            }
        }
        else{
            for(; count > 0; Count16 += 1, Count32 += 2, count -= 1){
                (tempResult+Count32)[0] = *(temp + Count16);
                (tempResult+Count32)[1] = 0;
            }
        }
    }
}
