//
// Created by jwkim98 on 19/02/19.
//
#include "../../Includes/cubbydnn/Types/bFloat16.hpp"

namespace CubbyDNN
{
bFloat16 bFloat16::operator+(bFloat16 rhs)
{
    float returnVal32 = this->getFloat() + rhs.getFloat();
    uint16_t returnVal16;
    TruncateFloatToBfloat16(&returnVal32, &returnVal16);
    return bFloat16{ returnVal16 };
}

bFloat16 bFloat16::operator-(bFloat16 rhs)
{
    float returnVal32 = this->getFloat() - rhs.getFloat();
    uint16_t returnVal16;
    TruncateFloatToBfloat16(&returnVal32, &returnVal16);
    return bFloat16{ returnVal16 };
}

bFloat16 bFloat16::operator*(bFloat16 rhs)
{
    float returnVal32 = this->getFloat() * rhs.getFloat();
    uint16_t returnVal16;
    TruncateFloatToBfloat16(&returnVal32, &returnVal16);
    return bFloat16{ returnVal16 };
}

bFloat16 bFloat16::operator/(bFloat16 rhs)
{
    float returnVal32 = this->getFloat() / rhs.getFloat();
    uint16_t returnVal16;
    TruncateFloatToBfloat16(&returnVal32, &returnVal16);
    return bFloat16{ returnVal16 };
}

float bFloat16::getFloat()
{
    float floatVal;
    TruncateBfloat16ToFloat(&m_value, &floatVal);
    return floatVal;
}
}  // namespace CubbyDNN